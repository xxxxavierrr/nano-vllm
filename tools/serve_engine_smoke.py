import argparse
import asyncio
import multiprocessing as mp
from uuid import uuid4

from transformers import AutoTokenizer

from nanovllm.serve.engine import EngineClient, run_engine_proc
from nanovllm.serve.protocol import MessageType
from nanovllm.serve.tokenizer import normalize_token_ids


async def smoke(args):
    endpoint = f"tcp://127.0.0.1:{args.engine_port}"
    process = mp.get_context("spawn").Process(
        target=run_engine_proc,
        args=(
            endpoint,
            args.model,
            {
                "enforce_eager": True,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "master_port": args.master_port,
                "shm_name": f"nanovllm-smoke-{uuid4().hex}",
            },
        ),
        daemon=False,
    )
    process.start()
    client = EngineClient(endpoint, process=process)
    await client.start()
    try:
        await client.wait_until_ready(timeout=args.startup_timeout)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        prompt_token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        request = await client.submit(
            normalize_token_ids(prompt_token_ids),
            {
                "temperature": 1.0,
                "max_tokens": args.max_tokens,
                "ignore_eos": False,
            },
        )
        output_token_ids = []
        finish_reason = None
        async for event in request.events():
            if event["type"] == MessageType.TOKEN:
                output_token_ids.append(event["token_id"])
                print(f"TOKEN {event['token_id']}", flush=True)
            elif event["type"] == MessageType.FINISHED:
                finish_reason = event["finish_reason"]
        print(f"FINISHED {finish_reason}", flush=True)
        print(tokenizer.decode(output_token_ids, skip_special_tokens=True), flush=True)
    finally:
        await client.close()
        await asyncio.to_thread(process.join, 10)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test EngineClient -> ZMQ -> EngineProc -> GPU")
    parser.add_argument("model")
    parser.add_argument("--prompt", default="Reply with exactly: hello")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--engine-port", type=int, default=5560)
    parser.add_argument("--master-port", type=int, default=2334)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(smoke(args))


if __name__ == "__main__":
    main()
