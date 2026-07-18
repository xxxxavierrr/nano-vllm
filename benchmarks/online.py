import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from benchmarks.backends.openai_chat import OpenAIChatBackend
from benchmarks.load_generator import run_load, run_warmups
from benchmarks.metrics import summarize
from benchmarks.reporter import print_summary, request_details, save_json
from benchmarks.workloads import make_synthetic_requests


PROFILE_DIR = Path(__file__).with_name("profiles")


def load_profile(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in PROFILE_DIR.glob("*.json")))
        raise ValueError(f"unknown profile {name!r}; available profiles: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Implementation-independent OpenAI Chat serving benchmark"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--shared-prefix-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--min-slo-attainment", type=float)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--output-json")
    parser.add_argument("--request-details", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--no-stream-usage", action="store_true")
    parser.add_argument("--skip-protocol-check", action="store_true")
    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.profile)
        args.metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (
        "num_requests",
        "max_concurrency",
        "request_rate",
        "input_len",
        "output_len",
        "shared_prefix_len",
        "temperature",
        "warmup_requests",
    ):
        if getattr(args, name) is None:
            setattr(args, name, profile[name])
    if isinstance(args.request_rate, str):
        try:
            args.request_rate = float(args.request_rate)
        except ValueError as exc:
            parser.error(f"invalid profile request_rate: {args.request_rate!r}")
    if args.num_requests <= 0 or args.max_concurrency <= 0:
        parser.error("num_requests and max_concurrency must be positive")
    if args.input_len <= 0 or args.output_len <= 0:
        parser.error("input_len and output_len must be positive")
    if not 0 <= args.shared_prefix_len <= args.input_len:
        parser.error("shared_prefix_len must be between zero and input_len")
    if args.request_rate <= 0 or args.temperature <= 0:
        parser.error("request_rate and temperature must be positive")
    if args.min_slo_attainment is not None and not 0 <= args.min_slo_attainment <= 1:
        parser.error("min_slo_attainment must be in [0, 1]")
    return args


async def run(args) -> tuple[dict, list]:
    headers = {"Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    limits = httpx.Limits(
        max_connections=max(args.max_concurrency, 1),
        max_keepalive_connections=max(args.max_concurrency, 1),
    )
    timeout = httpx.Timeout(args.timeout)
    requests = make_synthetic_requests(
        args.num_requests,
        args.input_len,
        args.output_len,
        args.shared_prefix_len,
        args.temperature,
        args.seed,
    )
    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        backend = OpenAIChatBackend(
            client,
            args.base_url,
            args.model,
            include_usage=not args.no_stream_usage,
        )
        if args.profile == "smoke" and not args.skip_protocol_check:
            await backend.validate_protocol(requests[0])
        await run_warmups(backend, requests, args.warmup_requests)
        results = await run_load(
            backend,
            requests,
            args.max_concurrency,
            args.request_rate,
            args.seed,
        )

    metrics = summarize(results, args.ttft_slo_ms, args.tpot_slo_ms, args.e2e_slo_ms)
    result = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "online",
        "metadata": args.metadata,
        "endpoint": {
            "backend": "openai-chat",
            "base_url": args.base_url,
            "model": args.model,
        },
        "backend": {
            "type": "openai-chat",
            "base_url": args.base_url,
            "model": args.model,
        },
        "workload": {
            "profile": args.profile,
            "num_requests": args.num_requests,
            "max_concurrency": args.max_concurrency,
            "request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
            "input_len_approx": args.input_len,
            "output_len": args.output_len,
            "shared_prefix_len_approx": args.shared_prefix_len,
            "temperature": args.temperature,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
        },
        "metrics": metrics,
    }
    if args.request_details:
        result["request_details"] = request_details(results)
    return result, results


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    result, results = asyncio.run(run(args))
    print_summary(result)
    if args.output_json:
        save_json(args.output_json, result)
        print(f"\nSaved JSON: {args.output_json}")

    failed = result["metrics"]["requests"]["failed"]
    if failed and not args.allow_errors:
        raise SystemExit(1)
    if args.min_slo_attainment is not None:
        good = result["metrics"]["slo"]["good_requests"]
        if good is None:
            raise SystemExit("--min-slo-attainment requires at least one SLO threshold")
        if good / len(results) < args.min_slo_attainment:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
