import asyncio
import math
import random
from time import perf_counte

from benchmarks.backends.openai_chat import OpenAIChatBackend
from benchmarks.models import ChatRequest, RequestResult


async def run_warmups(
    backend: OpenAIChatBackend,
    requests: list[ChatRequest],
    count: int,
) -> None:
    for index in range(count):
        source = requests[index % len(requests)]
        request = ChatRequest(
            request_id=f"warmup-{index}",
            messages=source.messages,
            max_tokens=source.max_tokens,
            temperature=source.temperature,
        )
        result = await backend.run(request, perf_counter())
        if not result.succeeded:
            raise RuntimeError(f"warmup failed: {result.error}")


async def run_load(
    backend: OpenAIChatBackend,
    requests: list[ChatRequest],
    max_concurrency: int,
    request_rate: float,
    seed: int,
) -> list[RequestResult]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")

    semaphore = asyncio.Semaphore(max_concurrency)
    rng = random.Random(seed + 1)
    benchmark_started = perf_counter()
    offsets = []
    offset = 0.0
    for index in range(len(requests)):
        if index and not math.isinf(request_rate):
            offset += rng.expovariate(request_rate)
        offsets.append(offset)

    async def execute(request: ChatRequest, arrival_offset: float):
        scheduled_s = benchmark_started + arrival_offset
        delay = scheduled_s - perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            return await backend.run(request, scheduled_s)

    tasks = [
        asyncio.create_task(execute(request, arrival_offset))
        for request, arrival_offset in zip(requests, offsets)
    ]
    return await asyncio.gather(*tasks)
