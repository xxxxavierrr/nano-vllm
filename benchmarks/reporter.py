import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.models import RequestResult


def print_distribution(name: str, values: dict[str, float]):
    print(
        f"{name:<12} mean={values['mean']:.2f}ms  p50={values['p50']:.2f}ms  "
        f"p90={values['p90']:.2f}ms  p95={values['p95']:.2f}ms  "
        f"p99={values['p99']:.2f}ms  max={values['max']:.2f}ms"
    )


def print_summary(result: dict):
    metrics = result["metrics"]
    requests = metrics["requests"]
    tokens = metrics["tokens"]
    throughput = metrics["throughput"]
    print("\n============ Online Serving Benchmark ============")
    print(
        f"Endpoint: {result['endpoint']['base_url']}  Model: {result['endpoint']['model']}  "
        f"Profile: {result['workload']['profile']}"
    )
    print(
        f"Requests: {requests['successful']}/{requests['total']} succeeded  "
        f"Duration: {metrics['duration_s']:.3f}s  "
        f"Max concurrency: {requests['max_observed_concurrency']}"
    )
    print("\n--- Throughput ---")
    print(f"Request throughput:      {throughput['request_per_s']:.2f} req/s")
    print(f"Output token throughput: {throughput['output_token_per_s']:.2f} tok/s")
    print(f"Total token throughput:  {throughput['total_token_per_s']:.2f} tok/s")
    print("\n--- Latency ---")
    for name in (
        "ttft",
        "service_ttft",
        "tpot",
        "inter_chunk",
        "e2e",
        "service_e2e",
        "client_queue",
    ):
        print_distribution(name, metrics["latency_ms"][name])
    print("\n--- Tokens / Cache ---")
    cache_rate = tokens["prefix_cache_hit_rate"]
    cache_text = f"{cache_rate:.2%}" if cache_rate is not None else "unavailable"
    print(
        f"Prompt: {tokens['prompt']}  Completion: {tokens['completion']}  "
        f"Cached: {tokens['cached']}  Prefix cache hit: {cache_text}"
    )
    print(f"Token count sources: {tokens['count_sources']}")
    if requests["failed"]:
        print(f"Errors: {requests['status_codes']}")
    if metrics["slo"]["enabled"]:
        print(
            f"SLO goodput: {metrics['slo']['goodput_request_per_s']:.2f} req/s "
            f"/ {metrics['slo']['goodput_output_token_per_s']:.2f} tok/s "
            f"({metrics['slo']['good_requests']}/{requests['total']})"
        )


def request_details(results: list[RequestResult]) -> list[dict]:
    details = []
    for result in results:
        item = asdict(result)
        item["succeeded"] = result.succeeded
        item["ttft_ms"] = (
            (result.first_content_s - result.scheduled_s) * 1000
            if result.first_content_s is not None
            else None
        )
        item["service_ttft_ms"] = (
            (result.first_content_s - result.started_s) * 1000
            if result.first_content_s is not None
            else None
        )
        item["e2e_ms"] = (result.finished_s - result.scheduled_s) * 1000
        item["service_e2e_ms"] = (result.finished_s - result.started_s) * 1000
        item["client_queue_ms"] = (result.started_s - result.scheduled_s) * 1000
        details.append(item)
    return details


def save_json(path: str, result: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
