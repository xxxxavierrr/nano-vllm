import argparse

from benchmarks.reporter import save_json


def parse_metadata(items: list[str]) -> dict[str, str]:
    metadata = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must use KEY=VALUE syntax: {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("metadata key cannot be empty")
        metadata[key] = value
    return metadata


def normalize_offline_result(result: dict, metadata: dict[str, str]) -> dict:
    workload = result["workload"]
    throughput = result["throughput"]
    scheduler = result["scheduler"]
    num_requests = workload["num_requests"]
    normalized = {
        "schema_version": 3,
        "timestamp_utc": result["timestamp_utc"],
        "mode": "offline",
        "metadata": {"label": result["label"], **metadata},
        "backend": {
            "type": "nanovllm-offline",
            "model": result["engine"]["model"],
        },
        "workload": workload,
        "metrics": {
            "duration_s": throughput["benchmark_duration_s"],
            "requests": {
                "total": num_requests,
                "successful": num_requests,
                "failed": 0,
                "error_rate": 0.0,
                "status_codes": {},
                "max_observed_concurrency": workload["max_concurrency"],
                "average_inflight_requests": scheduler.get(
                    "average_running_requests"
                ),
                "average_transport_requests": None,
            },
            "tokens": {
                "prompt": workload["total_input_tokens"],
                "completion": workload["actual_output_tokens"],
                "accepted": result.get("speculative", {}).get(
                    "accepted_tokens", 0
                ),
                "cached": scheduler["cached_prompt_tokens"],
                "prefix_cache_hit_rate": scheduler["prefix_cache_hit_rate"],
                "count_sources": {"engine": num_requests},
            },
            "throughput": {
                "request_per_s": throughput["request_per_s"],
                "output_token_per_s": throughput["output_token_per_s"],
                "total_token_per_s": throughput["total_token_per_s"],
                "accepted_token_per_s": throughput.get(
                    "accepted_token_per_s", 0.0
                ),
            },
            "latency_ms": result["latency_ms"],
            "slo": result["slo"],
            "engine": {
                "source": "nanovllm-offline",
                "average_running_requests": scheduler.get(
                    "average_running_requests"
                ),
                "scheduled_actual_tokens": scheduler.get(
                    "scheduled_actual_tokens"
                ),
                "scheduled_padded_tokens": scheduler.get(
                    "scheduled_padded_tokens"
                ),
                "padding_ratio": scheduler.get("padding_ratio"),
                "accepted_tokens": result.get("speculative", {}).get(
                    "accepted_tokens", 0
                ),
            },
        },
        "engine_metrics": {
            "system": result["system"],
            "engine": result["engine"],
            "phases": result["phases"],
            "execution_modes": result.get("execution_modes", {}),
            "scheduler": scheduler,
            "memory": result["memory"],
            "startup": result["startup"],
        },
    }
    if "requests" in result:
        normalized["request_details"] = result["requests"]
    return normalized


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--output-json")
    common, engine_argv = parser.parse_known_args(argv)
    try:
        metadata = parse_metadata(common.metadata)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Import lazily so offline mode owns the engine dependency while online mode
    # remains a pure HTTP client.
    import bench

    result = bench.main(engine_argv)
    normalized = normalize_offline_result(result, metadata)
    if common.output_json:
        save_json(common.output_json, normalized)
        print(f"\nSaved unified JSON: {common.output_json}")
    return normalized
