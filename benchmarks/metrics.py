import math
from collections import Counter
from statistics import fmean

from benchmarks.models import EngineSnapshot, RequestResult


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

    ttfts = [
        result.first_content_s - result.scheduled_s
        for result in successful
        if result.first_content_s is not None
    ]
    service_ttfts = [
        result.first_content_s - result.started_s
        for result in successful
        if result.first_content_s is not None
    ]
    e2es = [result.finished_s - result.scheduled_s for result in successful]
    service_e2es = [result.finished_s - result.started_s for result in successful]
    tpots = [
        (result.chunk_times_s[-1] - result.first_content_s) / (result.completion_tokens - 1)
        for result in successful
        if result.first_content_s is not None and result.chunk_times_s and result.completion_tokens > 1
    ]
    chunk_intervals = [
        current - previous
        for result in successful
        for previous, current in zip(result.chunk_times_s, result.chunk_times_s[1:])
    ]
    client_queue = [result.started_s - result.scheduled_s for result in results]

    good_requests = 0
    slo_enabled = any(value is not None for value in (ttft_slo_ms, tpot_slo_ms, e2e_slo_ms))
    for result in successful:
        ttft_ms = (
            (result.first_content_s - result.scheduled_s) * 1000
            if result.first_content_s is not None
            else math.inf
        )
        tpot_ms = (
            (result.chunk_times_s[-1] - result.first_content_s) * 1000 / (result.completion_tokens - 1)
            if result.first_content_s is not None and result.chunk_times_s and result.completion_tokens > 1
            else 0.0
        )
        e2e_ms = (result.finished_s - result.scheduled_s) * 1000
        good_requests += int(
            (ttft_slo_ms is None or ttft_ms <= ttft_slo_ms)
            and (tpot_slo_ms is None or tpot_ms <= tpot_slo_ms)
            and (e2e_slo_ms is None or e2e_ms <= e2e_slo_ms)
        )

    status_codes = Counter(str(result.status_code or "transport") for result in results if not result.succeeded)
    token_sources = Counter(result.token_count_source for result in successful)
    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    completion_tokens = sum(result.completion_tokens for result in successful)
    accepted_values = [
        result.accepted_tokens
        for result in successful
        if result.accepted_tokens is not None
    ]
    request_accepted_tokens = sum(accepted_values) if accepted_values else None
    cached_tokens = sum(result.cached_tokens or 0 for result in successful)
    engine = _summarize_engine_snapshots(engine_snapshots, started, finished)
    accepted_tokens = (
        engine["accepted_tokens"]
        if engine["accepted_tokens"] is not None
        else request_accepted_tokens
    )
    good_output_tokens = sum(
        result.completion_tokens
        for result in successful
        if (
            (ttft_slo_ms is None or (
                result.first_content_s is not None
                and (result.first_content_s - result.scheduled_s) * 1000 <= ttft_slo_ms
            ))
            and (tpot_slo_ms is None or (
                result.completion_tokens <= 1
                or (
                    result.first_content_s is not None
                    and bool(result.chunk_times_s)
                    and (result.chunk_times_s[-1] - result.first_content_s)
                    * 1000
                    / (result.completion_tokens - 1)
                    <= tpot_slo_ms
                )
            ))
            and (
                e2e_slo_ms is None
                or (result.finished_s - result.scheduled_s) * 1000 <= e2e_slo_ms
            )
        )
    )
    return {
        "duration_s": duration,
        "requests": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "error_rate": (len(results) - len(successful)) / len(results),
            "status_codes": dict(status_codes),
            "max_observed_concurrency": _max_concurrency(results),
            "average_inflight_requests": _time_weighted_request_count(
                results, use_scheduled_start=True
            ),
            "average_transport_requests": _time_weighted_request_count(
                results, use_scheduled_start=False
            ),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "accepted": accepted_tokens,
            "cached": cached_tokens,
            "prefix_cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else None,
            "count_sources": dict(token_sources),
        },
        "throughput": {
            "request_per_s": len(successful) / duration,
            "output_token_per_s": completion_tokens / duration,
            "total_token_per_s": (prompt_tokens + completion_tokens) / duration,
            "accepted_token_per_s": (
                accepted_tokens / duration if accepted_tokens is not None else None
            ),
        },
        "latency_ms": {
            "ttft": distribution(ttfts, 1000),
            "service_ttft": distribution(service_ttfts, 1000),
            "tpot": distribution(tpots, 1000),
            "inter_chunk": distribution(chunk_intervals, 1000),
            "e2e": distribution(e2es, 1000),
            "service_e2e": distribution(service_e2es, 1000),
            "client_queue": distribution(client_queue, 1000),
        },
        "slo": {
            "enabled": slo_enabled,
            "ttft_ms": ttft_slo_ms,
            "tpot_ms": tpot_slo_ms,
            "e2e_ms": e2e_slo_ms,
            "good_requests": good_requests if slo_enabled else None,
            "goodput_request_per_s": good_requests / duration if slo_enabled else None,
            "good_output_tokens": good_output_tokens if slo_enabled else None,
            "goodput_output_token_per_s": (
                good_output_tokens / duration if slo_enabled else None
            ),
            "attainment": good_requests / len(results) if slo_enabled else None,
        },
        "engine": engine,
    }
