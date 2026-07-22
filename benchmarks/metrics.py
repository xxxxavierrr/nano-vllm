import math
from collections import Counter
from dataclasses import dataclass
from statistics import fmean

from benchmarks.models import EngineSnapshot, RequestResult


@dataclass(frozen=True, slots=True)
class SLOThresholds:
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None

    @property
    def enabled(self) -> bool:
        return any(value is not None for value in (self.ttft_ms, self.tpot_ms, self.e2e_ms))


@dataclass(frozen=True, slots=True)
class RequestTiming:
    result: RequestResult
    ttft_s: float | None
    service_ttft_s: float | None
    tpot_s: float | None
    e2e_s: float
    service_e2e_s: float
    client_queue_s: float
    slo_good: bool


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float], scale: float = 1.0) -> dict[str, float]:
    scaled = [value * scale for value in values]
    if not scaled:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": fmean(scaled),
        "p50": percentile(scaled, 50),
        "p90": percentile(scaled, 90),
        "p95": percentile(scaled, 95),
        "p99": percentile(scaled, 99),
        "max": max(scaled),
    }


def _max_concurrency(results: list[RequestResult]) -> int:
    events = []
    for result in results:
        events.append((result.started_s, 1))
        events.append((result.finished_s, -1))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _time_weighted_request_count(
    results: list[RequestResult],
    *,
    use_scheduled_start: bool,
) -> float:
    if not results:
        return 0.0
    events: list[tuple[float, int]] = []
    for result in results:
        start = result.scheduled_s if use_scheduled_start else result.started_s
        events.append((start, 1))
        events.append((result.finished_s, -1))
    ordered = sorted(events, key=lambda event: (event[0], -event[1]))
    beginning = ordered[0][0]
    ending = max(result.finished_s for result in results)
    duration = max(ending - beginning, 1e-12)
    area = 0.0
    active = 0
    previous = beginning
    for timestamp, delta in ordered:
        area += active * max(timestamp - previous, 0.0)
        active += delta
        previous = timestamp
    return area / duration


def _summarize_engine_snapshots(
    snapshots: list[EngineSnapshot] | None,
    started: float,
    finished: float,
) -> dict:
    if not snapshots:
        return {
            "source": None,
            "average_running_requests": None,
            "scheduled_actual_tokens": None,
            "scheduled_padded_tokens": None,
            "padding_ratio": None,
            "accepted_tokens": None,
        }
    ordered = sorted(snapshots, key=lambda sample: sample.timestamp_s)
    boundaries = [sample for sample in ordered if sample.timestamp_s <= finished]
    if not boundaries:
        return {
            "source": "engine-snapshots",
            "average_running_requests": None,
            "scheduled_actual_tokens": None,
            "scheduled_padded_tokens": None,
            "padding_ratio": None,
            "accepted_tokens": None,
        }
    area = 0.0
    previous = started
    current = boundaries[0].running_requests
    for sample in boundaries:
        timestamp = min(max(sample.timestamp_s, started), finished)
        area += current * max(timestamp - previous, 0.0)
        current = sample.running_requests
        previous = timestamp
    area += current * max(finished - previous, 0.0)
    duration = max(finished - started, 1e-12)
    actual_values = [
        item.scheduled_actual_tokens
        for item in boundaries
        if item.scheduled_actual_tokens is not None
    ]
    padded_values = [
        item.scheduled_padded_tokens
        for item in boundaries
        if item.scheduled_padded_tokens is not None
    ]
    accepted_values = [
        item.accepted_tokens
        for item in boundaries
        if item.accepted_tokens is not None
    ]
    actual = sum(actual_values) if actual_values else None
    padded = sum(padded_values) if padded_values else None
    return {
        "source": "engine-snapshots",
        "average_running_requests": area / duration,
        "scheduled_actual_tokens": actual,
        "scheduled_padded_tokens": padded,
        "padding_ratio": (
            (padded - actual) / padded
            if padded is not None and actual is not None and padded
            else None
        ),
        "accepted_tokens": sum(accepted_values) if accepted_values else None,
    }


def _request_timing(result: RequestResult, slo: SLOThresholds) -> RequestTiming:
    ttft = (
        result.first_content_s - result.scheduled_s
        if result.first_content_s is not None
        else None
    )
    service_ttft = (
        result.first_content_s - result.started_s
        if result.first_content_s is not None
        else None
    )
    has_tpot = (
        result.first_content_s is not None
        and bool(result.chunk_times_s)
        and result.completion_tokens > 1
    )
    tpot = (
        (result.chunk_times_s[-1] - result.first_content_s)
        / (result.completion_tokens - 1)
        if has_tpot
        else (0.0 if result.completion_tokens <= 1 else None)
    )
    e2e = result.finished_s - result.scheduled_s
    good = (
        (slo.ttft_ms is None or (ttft is not None and ttft * 1000 <= slo.ttft_ms))
        and (slo.tpot_ms is None or (tpot is not None and tpot * 1000 <= slo.tpot_ms))
        and (slo.e2e_ms is None or e2e * 1000 <= slo.e2e_ms)
    )
    return RequestTiming(
        result=result,
        ttft_s=ttft,
        service_ttft_s=service_ttft,
        tpot_s=tpot,
        e2e_s=e2e,
        service_e2e_s=result.finished_s - result.started_s,
        client_queue_s=result.started_s - result.scheduled_s,
        slo_good=good,
    )


def _request_summary(results: list[RequestResult], successful: list[RequestResult]) -> dict:
    failed = len(results) - len(successful)
    status_codes = Counter(
        str(result.status_code or "transport")
        for result in results
        if not result.succeeded
    )
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": failed,
        "error_rate": failed / len(results),
        "status_codes": dict(status_codes),
        "max_observed_concurrency": _max_concurrency(results),
        "average_inflight_requests": _time_weighted_request_count(
            results, use_scheduled_start=True
        ),
        "average_transport_requests": _time_weighted_request_count(
            results, use_scheduled_start=False
        ),
    }


def _token_summary(successful: list[RequestResult], engine: dict) -> dict:
    prompt = sum(result.prompt_tokens or 0 for result in successful)
    completion = sum(result.completion_tokens for result in successful)
    cached = sum(result.cached_tokens or 0 for result in successful)
    request_accepted = [
        result.accepted_tokens
        for result in successful
        if result.accepted_tokens is not None
    ]
    accepted = engine["accepted_tokens"]
    if accepted is None and request_accepted:
        accepted = sum(request_accepted)
    return {
        "prompt": prompt,
        "completion": completion,
        "accepted": accepted,
        "cached": cached,
        "prefix_cache_hit_rate": cached / prompt if prompt else None,
        "count_sources": dict(Counter(result.token_count_source for result in successful)),
    }


def _latency_summary(timings: list[RequestTiming], results: list[RequestResult]) -> dict:
    intervals = [
        current - previous
        for timing in timings
        for previous, current in zip(
            timing.result.chunk_times_s, timing.result.chunk_times_s[1:]
        )
    ]
    present = lambda values: [value for value in values if value is not None]
    return {
        "ttft": distribution(present([item.ttft_s for item in timings]), 1000),
        "service_ttft": distribution(
            present([item.service_ttft_s for item in timings]), 1000
        ),
        "tpot": distribution(
            present([item.tpot_s for item in timings if item.result.completion_tokens > 1]),
            1000,
        ),
        "inter_chunk": distribution(intervals, 1000),
        "e2e": distribution([item.e2e_s for item in timings], 1000),
        "service_e2e": distribution([item.service_e2e_s for item in timings], 1000),
        "client_queue": distribution(
            [result.started_s - result.scheduled_s for result in results], 1000
        ),
    }


def _slo_summary(
    timings: list[RequestTiming],
    slo: SLOThresholds,
    duration: float,
    total_requests: int,
) -> dict:
    good = [timing for timing in timings if timing.slo_good]
    good_tokens = sum(item.result.completion_tokens for item in good)
    return {
        "enabled": slo.enabled,
        "ttft_ms": slo.ttft_ms,
        "tpot_ms": slo.tpot_ms,
        "e2e_ms": slo.e2e_ms,
        "good_requests": len(good) if slo.enabled else None,
        "goodput_request_per_s": len(good) / duration if slo.enabled else None,
        "good_output_tokens": good_tokens if slo.enabled else None,
        "goodput_output_token_per_s": good_tokens / duration if slo.enabled else None,
        "attainment": len(good) / total_requests if slo.enabled else None,
    }


def summarize(
    results: list[RequestResult],
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
    engine_snapshots: list[EngineSnapshot] | None = None,
) -> dict:
    if not results:
        raise ValueError("at least one benchmark result is required")
    successful = [result for result in results if result.succeeded]
    started = min(result.scheduled_s for result in results)
    finished = max(result.finished_s for result in results)
    duration = max(finished - started, 1e-12)
    slo = SLOThresholds(ttft_slo_ms, tpot_slo_ms, e2e_slo_ms)
    timings = [_request_timing(result, slo) for result in successful]
    engine = _summarize_engine_snapshots(engine_snapshots, started, finished)
    tokens = _token_summary(successful, engine)
    return {
        "duration_s": duration,
        "requests": _request_summary(results, successful),
        "tokens": tokens,
        "throughput": {
            "request_per_s": len(successful) / duration,
            "output_token_per_s": tokens["completion"] / duration,
            "total_token_per_s": (tokens["prompt"] + tokens["completion"]) / duration,
            "accepted_token_per_s": (
                tokens["accepted"] / duration if tokens["accepted"] is not None else None
            ),
        },
        "latency_ms": _latency_summary(timings, results),
        "slo": _slo_summary(timings, slo, duration, len(results)),
        "engine": engine,
    }
