import argparse
import asyncio
import json
import multiprocessing as mp
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoConfig, AutoTokenizer
import uvicorn

from nanovllm.engine.cudagraph import CUDAGraphMode
from nanovllm.serve.engine import (
    EngineClient,
    EngineRequest,
    EngineRequestError,
    EngineUnavailableError,
    RequestQueueFullError,
    run_engine_proc,
)
from nanovllm.serve.protocol import MessageType
from nanovllm.serve.tokenizer import normalize_token_ids


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def validate_token_limits(self):
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens must match when both are provided")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self

    @property
    def resolved_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 64


@dataclass(slots=True)
class ServerSettings:
    model: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    engine_host: str = "127.0.0.1"
    engine_port: int = 5555
    startup_timeout: float = 300.0
    max_pending_requests: int = 1024
    max_model_len: int = 4096
    engine_kwargs: dict | None = None

    @property
    def engine_endpoint(self) -> str:
        return f"tcp://{self.engine_host}:{self.engine_port}"


class ServingRuntime:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model, use_fast=True)
        hf_config = AutoConfig.from_pretrained(settings.model)
        self.max_model_len = min(settings.max_model_len, hf_config.max_position_embeddings)
        self._tokenizer_lock = threading.Lock()
        self.process: mp.Process | None = None
        self.client: EngineClient | None = None

    async def start(self):
        context = mp.get_context("spawn")
        self.process = context.Process(
            target=run_engine_proc,
            args=(self.settings.engine_endpoint, self.settings.model, self.settings.engine_kwargs or {}),
            daemon=False,
            name="nanovllm-engine",
        )
        self.process.start()
        self.client = EngineClient(
            self.settings.engine_endpoint,
            max_pending_requests=self.settings.max_pending_requests,
            process=self.process,
        )
        await self.client.start()
        try:
            await self.client.wait_until_ready(self.settings.startup_timeout)
        except BaseException:
            await self.close(graceful=False)
            raise

    async def close(self, graceful: bool = True):
        if self.client is not None:
            await self.client.close(shutdown_engine=graceful)
        if self.process is not None:
            await asyncio.to_thread(self.process.join, 10 if graceful else 0)
            if self.process.is_alive():
                self.process.terminate()
                await asyncio.to_thread(self.process.join, 5)

    def encode_messages(self, messages: list[ChatMessage]) -> list[int]:
        payload = [message.model_dump() for message in messages]
        with self._tokenizer_lock:
            token_ids = self.tokenizer.apply_chat_template(
                payload,
                tokenize=True,
                add_generation_prompt=True,
            )
        return normalize_token_ids(token_ids)

    def decode(self, token_ids: list[int]) -> str:
        with self._tokenizer_lock:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    async def healthy(self) -> bool:
        if self.client is None or not self.client.process_alive:
            return False
        try:
            await self.client.ping(timeout=0.5)
            return True
        except Exception:
            return False


def _completion_base(completion_id: str, created: int, model: str, object_type: str):
    return {
        "id": completion_id,
        "object": object_type,
        "created": created,
        "model": model,
    }


def _sse(message: dict) -> str:
    return f"data: {json.dumps(message, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def _stream_chat_completion(
    runtime: ServingRuntime,
    engine_request: EngineRequest,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int = 0,
    include_usage: bool = False,
):
    base = _completion_base(completion_id, created, model, "chat.completion.chunk")
    role_chunk = {
        **base,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    }
    token_ids: list[int] = []
    cached_tokens = 0
    previous_text = ""
    try:
        yield _sse(role_chunk)
        async for event in engine_request.events():
            if event["type"] == MessageType.TOKEN:
                token_ids.append(event["token_id"])
                text = runtime.decode(token_ids)
                delta = text[len(previous_text):] if text.startswith(previous_text) else text
                previous_text = text
                if delta:
                    yield _sse({
                        **base,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }],
                    })
            elif event["type"] == MessageType.FINISHED:
                cached_tokens = event.get("cached_tokens", 0)
                yield _sse({
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": event.get("finish_reason") or "stop",
                    }],
                })
        if include_usage:
            yield _sse({
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(token_ids),
                    "total_tokens": prompt_tokens + len(token_ids),
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                },
            })
        yield "data: [DONE]\n\n"
    except EngineRequestError as exc:
        yield _sse({"error": {"message": str(exc), "type": "engine_error"}})
        yield "data: [DONE]\n\n"
    finally:
        await engine_request.abort()


async def _collect_chat_completion(
    runtime: ServingRuntime,
    engine_request: EngineRequest,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
):
    token_ids: list[int] = []
    finish_reason = "stop"
    cached_tokens = 0
    async for event in engine_request.events():
        if event["type"] == MessageType.TOKEN:
            token_ids.append(event["token_id"])
        elif event["type"] == MessageType.FINISHED:
            finish_reason = event.get("finish_reason") or "stop"
            cached_tokens = event.get("cached_tokens", 0)
    text = runtime.decode(token_ids)
    return {
        **_completion_base(completion_id, created, model, "chat.completion"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def create_app(settings: ServerSettings, runtime: ServingRuntime | None = None) -> FastAPI:
    owns_runtime = runtime is None
    runtime = runtime or ServingRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if owns_runtime:
            await runtime.start()
        try:
            yield
        finally:
            if owns_runtime:
                await runtime.close()

    app = FastAPI(title="nano-vLLM OpenAI API", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        if not await runtime.healthy():
            raise HTTPException(status_code=503, detail="engine is unavailable")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": settings.served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-vllm",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        if body.model != settings.served_model_name:
            raise HTTPException(status_code=404, detail=f"model not found: {body.model}")
        prompt_token_ids = await asyncio.to_thread(runtime.encode_messages, body.messages)
        max_tokens = body.resolved_max_tokens
        if len(prompt_token_ids) + max_tokens > runtime.max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt ({len(prompt_token_ids)} tokens) and completion ({max_tokens} tokens) "
                    f"exceed max_model_len={runtime.max_model_len}"
                ),
            )
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="client disconnected")
        if runtime.client is None:
            raise HTTPException(status_code=503, detail="engine is unavailable")
        try:
            engine_request = await runtime.client.submit(
                prompt_token_ids,
                {
                    "temperature": body.temperature,
                    "max_tokens": max_tokens,
                    "ignore_eos": False,
                },
            )
        except RequestQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except EngineUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())
        if body.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    runtime,
                    engine_request,
                    completion_id,
                    created,
                    settings.served_model_name,
                    len(prompt_token_ids),
                    bool(body.stream_options and body.stream_options.include_usage),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await _collect_chat_completion(
                runtime,
                engine_request,
                completion_id,
                created,
                settings.served_model_name,
                len(prompt_token_ids),
            )
        except EngineRequestError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _parse_device_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device IDs must be comma-separated integers") from exc


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Serve nano-vLLM with an OpenAI-compatible API")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-host", default="127.0.0.1")
    parser.add_argument("--engine-port", type=int, default=5555)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--max-pending-requests", type=int, default=1024)
    parser.add_argument("--quantization", choices=["fp8", "gptq"])
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8_e4m3"], default="auto")
    parser.add_argument(
        "--speculative-method", choices=["none", "mtp"], default="none"
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--mtp-model")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--device-ids", type=_parse_device_ids)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--shm-name")
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cudagraph-mode",
        choices=[mode.value for mode in CUDAGraphMode],
        default=CUDAGraphMode.FULL_AND_PIECEWISE.value,
    )
    parser.add_argument("--piecewise-max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if not Path(args.model).is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not 1 <= args.port <= 65535 or not 1 <= args.engine_port <= 65535:
        parser.error("ports must be in [1, 65535]")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1, 65535]")
    if not 1 <= args.tensor_parallel_size <= 8:
        parser.error("--tensor-parallel-size must be in [1, 8]")
    if args.device_ids is not None and len(args.device_ids) != args.tensor_parallel_size:
        parser.error("--device-ids count must match --tensor-parallel-size")
    if args.max_pending_requests <= 0:
        parser.error("--max-pending-requests must be positive")
    if args.speculative_method == "mtp":
        if args.num_speculative_tokens not in (1, 2):
            parser.error("--num-speculative-tokens must be 1 or 2")
        if args.mtp_model is not None and not Path(args.mtp_model).is_dir():
            parser.error(f"MTP model directory does not exist: {args.mtp_model}")
    if args.piecewise_max_tokens <= 0:
        parser.error("--piecewise-max-tokens must be positive")
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    served_model_name = args.served_model_name or Path(args.model).name
    shm_name = args.shm_name or f"nanovllm-{uuid4().hex}"
    engine_kwargs = {
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "device_ids": args.device_ids,
        "master_port": args.master_port,
        "shm_name": shm_name,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "cudagraph_mode": args.cudagraph_mode,
        "piecewise_max_tokens": args.piecewise_max_tokens,
    }
    settings = ServerSettings(
        model=args.model,
        served_model_name=served_model_name,
        host=args.host,
        port=args.port,
        engine_host=args.engine_host,
        engine_port=args.engine_port,
        startup_timeout=args.startup_timeout,
        max_pending_requests=args.max_pending_requests,
        max_model_len=args.max_model_len,
        engine_kwargs=engine_kwargs,
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()