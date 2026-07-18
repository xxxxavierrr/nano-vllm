import asyncio
import json

import zmq
import zmq.asyncio

from nanovllm.serve.engine import EngineClient, RequestQueueFullError
from nanovllm.serve.protocol import MessageType


def run(coro):
    return asyncio.run(coro)


async def send(router, identity, message):
    await router.send_multipart([identity, json.dumps(message).encode()])


def test_routes_concurrent_requests_by_request_id():
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        client = EngineClient(f"tcp://127.0.0.1:{port}")
        await client.start()
        try:
            first = await client.submit([1], {"temperature": 1.0, "max_tokens": 1}, "first")
            second = await client.submit([2], {"temperature": 1.0, "max_tokens": 1}, "second")
            identities = {}
            for _ in range(2):
                identity, payload = await router.recv_multipart()
                message = json.loads(payload)
                identities[message["request_id"]] = identity

            await send(router, identities["second"], {
                "type": MessageType.TOKEN,
                "request_id": "second",
                "token_id": 22,
            })
            await send(router, identities["first"], {
                "type": MessageType.TOKEN,
                "request_id": "first",
                "token_id": 11,
            })
            for request_id in ("second", "first"):
                await send(router, identities[request_id], {
                    "type": MessageType.FINISHED,
                    "request_id": request_id,
                    "finish_reason": "length",
                })

            first_events, second_events = await asyncio.gather(
                _collect(first.events()),
                _collect(second.events()),
            )
            assert first_events[0]["token_id"] == 11
            assert second_events[0]["token_id"] == 22
            assert client.pending_requests == 0
        finally:
            await client.close(shutdown_engine=False)
            router.close(linger=0)
            context.term()

    run(scenario())


def test_rejects_requests_when_queue_is_full():
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        client = EngineClient(f"tcp://127.0.0.1:{port}", max_pending_requests=1)
        await client.start()
        try:
            await client.submit([1], {"temperature": 1.0, "max_tokens": 1}, "first")
            try:
                await client.submit([2], {"temperature": 1.0, "max_tokens": 1}, "second")
            except RequestQueueFullError:
                pass
            else:
                raise AssertionError("expected RequestQueueFullError")
        finally:
            await client.close(shutdown_engine=False)
            router.close(linger=0)
            context.term()

    run(scenario())


async def _collect(iterator):
    return [event async for event in iterator]

