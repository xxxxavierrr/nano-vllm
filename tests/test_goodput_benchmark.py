import asyncio
from time import perf_counter

from benchmarks.load_generator import run_load
from benchmarks.models import ChatRequest, RequestResult
from benchmarks.sweep import run_offered_load_sweep
from benchmarks.telemetry import GpuSample, GpuTelemetryMonitor


class FakeBackend:
    async def run(self, request, scheduled_s):
        started = perf_counter()
        await asyncio.sleep(0)
        finished = perf_counter()
        return RequestResult(
            request_id=request.request_id,
            scheduled_s=scheduled_s,
            started_s=started,
            first_content_s=started,
            finished_s=finished,
            chunk_times_s=[started],
            status_code=200,
            completion_tokens=1,
            saw_done=True,
        )


def test_load_generator_accepts_implementation_independent_fake_backend():
    requests = [
        ChatRequest(f"r{index}", [{"role": "user", "content": "x"}], 1, 0.1)
        for index in range(3)
    ]
    results = asyncio.run(run_load(FakeBackend(), requests, 2, float("inf"), 0))
    assert [result.request_id for result in results] == ["r0", "r1", "r2"]
    assert all(result.succeeded for result in results)


def test_sweep_grows_then_refines_maximum_passing_rate():
    async def scenario():
        async def run_point(rate):
            attainment = 1.0 if rate <= 6 else 0.8
            return {
                "metrics": {
                    "slo": {"attainment": attainment},
                    "requests": {"error_rate": 0.0},
                }
            }

        return await run_offered_load_sweep(
            run_point,
            start_rate=2,
            growth_factor=2,
            max_rate=16,
            refine_steps=2,
            min_attainment=0.99,
            max_error_rate=0.01,
        )

    points, selected = asyncio.run(scenario())
    assert [point["sweep"]["offered_request_rate"] for point in points] == [2, 4, 6, 7, 8]
    assert selected["sweep"] == {"offered_request_rate": 6, "passed": True}


def test_gpu_telemetry_supports_fake_reader_without_nvidia_smi():
    counter = 0

    async def reader():
        nonlocal counter
        counter += 1
        return GpuSample(perf_counter(), 50 + counter, 100, 200, 150)

    async def scenario():
        monitor = GpuTelemetryMonitor(0.001, reader)
        async with monitor:
            await asyncio.sleep(0.004)
        return monitor.report()

    report = asyncio.run(scenario())
    assert report["available"] is True
    assert report["source"] == "nvidia-smi"
    assert report["sample_count"] >= 1
