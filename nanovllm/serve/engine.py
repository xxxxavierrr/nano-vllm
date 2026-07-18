import asyncio
import json
import multiprocessing as mp
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

import zmq
import zmq.asyncio

from nanovllm.serve.protocol import MessageType, TERMINAL_MESSAGE_TYPES


class EngineUnavailableError(RuntimeError):
    pass


class RequestQueueFullError(RuntimeError):
    pass


class EngineRequestError(RuntimeError):
    pass


@dataclass(slots=True)
class EngineRequest:
    request_id: str
    _queue: asyncio.Queue
    _client: "EngineClient"
    _closed: bool = False

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                message = await self._queue.get()
                message_type = message["type"]
                if message_type == MessageType.ERROR:
                    raise EngineRequestError(message.get("message", "engine request failed"))
                yield message
                if message_type == MessageType.FINISHED:
                    break
        finally:
            await self.abort()

    async def abort(self):
        if self._closed:
            return
        self._closed = True
        await self._client.abort(self.request_id)


class EngineClient:
    def __init__(
        self,
        endpoint: str,
        max_pending_requests: int = 1024,
        process: mp.Process | None = None,
    ):
        self.endpoint = endpoint
        self.max_pending_requests = max_pending_requests
        self.process = process
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._receiver_task: asyncio.Task | None = None
        self._requests: dict[str, asyncio.Queue] = {}
        self._pings: dict[str, asyncio.Future] = {}
        self._shutdown_future: asyncio.Future | None = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def pending_requests(self) -> int:
        return len(self._requests)

    @property
    def process_alive(self) -> bool:
        return self.process is None or self.process.is_alive()

    async def start(self):
        if self._socket is not None:
            return
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, f"api-{uuid4().hex}".encode())
        self._socket.connect(self.endpoint)
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def _send(self, message: dict[str, Any]):
        if self._closed or self._socket is None:
            raise EngineUnavailableError("engine client is not running")
        if not self.process_alive:
            raise EngineUnavailableError("engine process is not running")
        async with self._send_lock:
            await self._socket.send_json(message)

    async def submit(
        self,
        prompt_token_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str | None = None,
    ) -> EngineRequest:
        if len(self._requests) >= self.max_pending_requests:
            raise RequestQueueFullError("engine request queue is full")
        request_id = request_id or uuid4().hex
        if request_id in self._requests:
            raise ValueError(f"duplicate request id: {request_id}")
        queue: asyncio.Queue = asyncio.Queue()
        self._requests[request_id] = queue
        try:
            await self._send({
                "type": MessageType.ADD_REQUEST,
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
            })
        except BaseException:
            self._requests.pop(request_id, None)
            raise
        return EngineRequest(request_id, queue, self)

    async def abort(self, request_id: str):
        queue = self._requests.pop(request_id, None)
        if queue is None or self._closed:
            return
        with suppress(EngineUnavailableError, zmq.ZMQError):
            await self._send({
                "type": MessageType.ABORT_REQUEST,
                "request_id": request_id,
            })

    async def ping(self, timeout: float = 1.0) -> dict[str, Any]:
        ping_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pings[ping_id] = future
        try:
            await self._send({"type": MessageType.PING, "ping_id": ping_id})
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pings.pop(ping_id, None)

    async def wait_until_ready(self, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if not self.process_alive:
                raise EngineUnavailableError("engine process exited during startup")
            try:
                await self.ping(timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                return
            except (TimeoutError, EngineUnavailableError, zmq.ZMQError) as exc:
                last_error = exc
                await asyncio.sleep(0.1)
        raise EngineUnavailableError("timed out waiting for engine readiness") from last_error

    async def _receive_loop(self):
        assert self._socket is not None
        try:
            while True:
                message = await self._socket.recv_json()
                message_type = message.get("type")
                if message_type == MessageType.PONG:
                    future = self._pings.get(message.get("ping_id"))
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue
                if message_type == MessageType.SHUTDOWN_ACK:
                    if self._shutdown_future is not None and not self._shutdown_future.done():
                        self._shutdown_future.set_result(message)
                    continue
                request_id = message.get("request_id")
                queue = self._requests.get(request_id)
                if queue is None:
                    continue
                queue.put_nowait(message)
                if message_type in TERMINAL_MESSAGE_TYPES:
                    self._requests.pop(request_id, None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = {
                "type": MessageType.ERROR,
                "message": f"engine connection failed: {exc}",
            }
            for request_id, queue in list(self._requests.items()):
                queue.put_nowait({**error, "request_id": request_id})
            self._requests.clear()

    async def close(self, shutdown_engine: bool = True):
        if self._closed:
            return
        if shutdown_engine and self._socket is not None and self.process_alive:
            with suppress(Exception):
                self._shutdown_future = asyncio.get_running_loop().create_future()
                await self._send({"type": MessageType.SHUTDOWN})
                await asyncio.wait_for(self._shutdown_future, timeout=5.0)
        self._closed = True
        for request_id, queue in list(self._requests.items()):
            queue.put_nowait({
                "type": MessageType.ERROR,
                "request_id": request_id,
                "message": "engine client closed",
            })
        self._requests.clear()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._receiver_task
        if self._socket is not None:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()


def _encode_message(message: dict[str, Any]) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def _send_router(socket: zmq.Socket, identity: bytes, message: dict[str, Any]):
    socket.send_multipart([identity, _encode_message(message)])


def run_engine_proc(endpoint: str, model: str, engine_kwargs: dict[str, Any]):
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    engine: Any | None = None
    request_to_seq: dict[str, int] = {}
    seq_to_request: dict[int, tuple[str, bytes]] = {}
    running = True

    try:
        engine = LLMEngine(model, **engine_kwargs)
        while running:
            timeout_ms = 100 if engine.is_finished() else 0
            if socket in dict(poller.poll(timeout_ms)):
                for _ in range(64):
                    message: dict[str, Any] = {}
                    try:
                        identity, payload = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        message = json.loads(payload)
                        message_type = message.get("type")
                        if message_type == MessageType.PING:
                            _send_router(socket, identity, {
                                "type": MessageType.PONG,
                                "ping_id": message.get("ping_id"),
                            })
                        elif message_type == MessageType.ADD_REQUEST:
                            request_id = message["request_id"]
                            if request_id in request_to_seq:
                                raise ValueError(f"duplicate request id: {request_id}")
                            params = SamplingParams(**message["sampling_params"])
                            seq_id = engine.add_request(message["prompt_token_ids"], params)
                            request_to_seq[request_id] = seq_id
                            seq_to_request[seq_id] = (request_id, identity)
                        elif message_type == MessageType.ABORT_REQUEST:
                            request_id = message["request_id"]
                            seq_id = request_to_seq.pop(request_id, None)
                            if seq_id is not None:
                                engine.abort_request(seq_id)
                                seq_to_request.pop(seq_id, None)
                        elif message_type == MessageType.SHUTDOWN:
                            _send_router(socket, identity, {"type": MessageType.SHUTDOWN_ACK})
                            running = False
                            break
                        else:
                            raise ValueError(f"unsupported message type: {message_type}")
                    except Exception as exc:
                        _send_router(socket, identity, {
                            "type": MessageType.ERROR,
                            "request_id": message.get("request_id"),
                            "message": str(exc),
                        })

            if running and not engine.is_finished():
                outputs, _ = engine.step()
                for output in outputs:
                    owner = seq_to_request.get(output.seq_id)
                    if owner is None:
                        continue
                    request_id, identity = owner
                    _send_router(socket, identity, {
                        "type": MessageType.TOKEN,
                        "request_id": request_id,
                        "token_id": output.token_id,
                    })
                    if output.finished:
                        _send_router(socket, identity, {
                            "type": MessageType.FINISHED,
                            "request_id": request_id,
                            "finish_reason": output.finish_reason,
                            "cached_tokens": output.cached_tokens,
                        })
                        request_to_seq.pop(request_id, None)
                        seq_to_request.pop(output.seq_id, None)
    except BaseException as exc:
        message = f"engine process failed: {exc}"
        for seq_id, (request_id, identity) in list(seq_to_request.items()):
            with suppress(Exception):
                _send_router(socket, identity, {
                    "type": MessageType.ERROR,
                    "request_id": request_id,
                    "message": message,
                })
        traceback.print_exc()
        raise
    finally:
        if engine is not None:
            with suppress(Exception):
                engine.exit()
        socket.close(linger=0)
        context.term()
