import asyncio
from types import SimpleNamespace

from nanovllm.serve.engine import (
    DataParallelEngineClient,
    RequestQueueFullError,
)


class FakeReplica:
    def __init__(self, replica_id, pending=0, alive=True, ready=True):
        self.replica_id = replica_id
        self.pending_requests = pending
        self.process_alive = alive
        self.ready = ready
        self.submitted = []

    async def submit(self, prompt_token_ids, sampling_params, request_id=None):
        self.pending_requests += 1
        self.submitted.append(request_id)
        return SimpleNamespace(replica_id=self.replica_id, request_id=request_id)

    async def ping(self, timeout=0.5):
        if not self.ready:
            raise TimeoutError
        return {"type": "pong"}

    async def close(self, shutdown_engine=True):
        return None


def run(coro):
    return asyncio.run(coro)


def test_routes_to_least_loaded_replica_and_rotates_ties():
    async def scenario():
        replicas = [FakeReplica(0), FakeReplica(1)]
        pool = DataParallelEngineClient(replicas, max_pending_requests=8)

        first = await pool.submit([1], {}, "first")
        second = await pool.submit([2], {}, "second")
        third = await pool.submit([3], {}, "third")

        assert [first.replica_id, second.replica_id, third.replica_id] == [0, 1, 0]
        assert pool.pending_requests == 3

    run(scenario())


def test_skips_dead_replica_and_reports_health():
    async def scenario():
        replicas = [
            FakeReplica(0, alive=False, ready=False),
            FakeReplica(1),
        ]
        pool = DataParallelEngineClient(replicas)

        request = await pool.submit([1], {}, "request")
        pong = await pool.ping()

        assert request.replica_id == 1
        assert pong["healthy_replicas"] == 1
        assert pong["replicas"][0]["ready"] is False
        assert pong["replicas"][1]["ready"] is True

    run(scenario())


def test_global_pending_limit_applies_across_replicas():
    async def scenario():
        replicas = [FakeReplica(0, pending=1), FakeReplica(1, pending=1)]
        pool = DataParallelEngineClient(replicas, max_pending_requests=2)

        try:
            await pool.submit([1], {}, "overflow")
        except RequestQueueFullError:
            pass
        else:
            raise AssertionError("expected global DP queue limit")

    run(scenario())
