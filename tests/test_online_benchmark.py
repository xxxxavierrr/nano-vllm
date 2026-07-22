import asyncio
import json
from time import perf_counter

import httpx
import pytest

from benchmarks.backends.openai_chat import OpenAIChatBackend
from benchmarks.metrics import summarize
from benchmarks.models import ChatRequest, EngineSnapshot, RequestResult
from benchmarks.online import parse_args, run
from benchmarks.offline import normalize_offline_result


def test_openai_backend_collects_streaming_usage_and_cache_metrics():
    async def handler(request: httpx.Request):
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        content = "".join([
            'data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,',
            '"total_tokens":12,"prompt_tokens_details":{"cached_tokens":8},',
            '"completion_tokens_details":{"accepted_prediction_tokens":1}}}\n\n',
            'data: [DONE]\n\n',
        ])
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            backend = OpenAIChatBackend(client, "http://test", "model")
            request = ChatRequest("r1", [{"role": "user", "content": "hi"}], 2, 0.1)
            return await backend.run(request, perf_counter())

    result = asyncio.run(scenario())
    assert result.succeeded
    assert result.text == "hello"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2
    assert result.cached_tokens == 8
    assert result.accepted_tokens == 1
    assert result.token_count_source == "usage"


def test_openai_backend_protocol_check_covers_models_and_non_streaming():
    async def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        payload = json.loads(request.content)
        assert payload["stream"] is False
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        })

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            backend = OpenAIChatBackend(client, "http://test", "model")
            request = ChatRequest("r1", [{"role": "user", "content": "hi"}], 2, 0.1)
            await backend.validate_protocol(request)

    asyncio.run(scenario())


def test_summary_reports_errors_goodput_and_token_throughput():
    start = perf_counter()
    successful = RequestResult(
        request_id="ok",
        scheduled_s=start,
        started_s=start,
        first_content_s=start + 0.01,
        finished_s=start + 0.03,
        chunk_times_s=[start + 0.01, start + 0.02, start + 0.03],
        status_code=200,
        prompt_tokens=10,
        completion_tokens=3,
        cached_tokens=8,
        token_count_source="usage",
        saw_done=True,
    )
    failed = RequestResult(
        request_id="failed",
        scheduled_s=start,
        started_s=start,
        finished_s=start + 0.02,
        status_code=429,
        error="queue full",
    )
    metrics = summarize([successful, failed], ttft_slo_ms=20, e2e_slo_ms=50)
    assert metrics["requests"]["successful"] == 1
    assert metrics["requests"]["status_codes"] == {"429": 1}
    assert metrics["tokens"]["prefix_cache_hit_rate"] == 0.8
    assert metrics["slo"]["good_requests"] == 1
    assert metrics["slo"]["good_output_tokens"] == 3
    assert metrics["slo"]["goodput_output_token_per_s"] > 0


def test_summary_includes_client_queue_in_slo_and_engine_occupancy():
    start = perf_counter()
    result = RequestResult(
        request_id="queued",
        scheduled_s=start,
        started_s=start + 0.02,
        first_content_s=start + 0.03,
        finished_s=start + 0.05,
        chunk_times_s=[start + 0.03, start + 0.05],
        status_code=200,
        completion_tokens=2,
        saw_done=True,
    )
    snapshots = [
        EngineSnapshot(start, 0, 3, 8, 1),
        EngineSnapshot(start + 0.01, 2, 5, 8, 2),
        EngineSnapshot(start + 0.04, 1, 2, 4, 0),
    ]

    metrics = summarize(
        [result],
        ttft_slo_ms=20,
        engine_snapshots=snapshots,
    )

    assert metrics["latency_ms"]["ttft"]["p50"] == pytest.approx(30)
    assert metrics["latency_ms"]["service_ttft"]["p50"] == pytest.approx(10)
    assert metrics["slo"]["good_requests"] == 0
    assert metrics["engine"]["scheduled_actual_tokens"] == 10
    assert metrics["engine"]["scheduled_padded_tokens"] == 20
    assert metrics["engine"]["average_running_requests"] > 1


def test_smoke_profile_can_be_overridden_from_cli():
    args = parse_args(["--model", "m", "--profile", "smoke", "--num-requests", "3"])
    assert args.num_requests == 3
    assert args.max_concurrency == 2
    assert args.request_rate == float("inf")


def test_complete_online_runner_works_with_mock_transport():
    async def handler(request: httpx.Request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        body = "".join([
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        ])
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    args = parse_args([
        "--model", "m",
        "--profile", "smoke",
        "--num-requests", "2",
        "--warmup-requests", "0",
        "--skip-protocol-check",
        "--ttft-slo-ms", "1000",
    ])
    result, requests = asyncio.run(run(args, transport=httpx.MockTransport(handler)))

    assert result["schema_version"] == 3
    assert result["metrics"]["requests"]["successful"] == 2
    assert result["metrics"]["slo"]["good_output_tokens"] == 2
    assert result["telemetry"]["gpu"]["missing_reason"] == "disabled"
    assert len(requests) == 2


def test_offline_result_uses_the_shared_outer_schema():
    raw = {
        "schema_version": 1,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "label": "nano-vllm",
        "system": {"gpu": "test"},
        "engine": {"model": "/model", "quantization": "none"},
        "workload": {
            "num_requests": 2,
            "max_concurrency": 2,
            "total_input_tokens": 20,
            "actual_output_tokens": 6,
        },
        "throughput": {
            "benchmark_duration_s": 1.0,
            "request_per_s": 2.0,
            "output_token_per_s": 6.0,
            "total_token_per_s": 26.0,
        },
        "latency_ms": {"ttft": {}, "tpot": {}, "itl": {}, "e2e": {}},
        "phases": {"prefill": {}, "decode": {}},
        "scheduler": {
            "cached_prompt_tokens": 8,
            "prefix_cache_hit_rate": 0.4,
        },
        "memory": {},
        "startup": {},
        "slo": {"enabled": False},
    }

    result = normalize_offline_result(raw, {"quantization": "fp8"})

    assert result["schema_version"] == 3
    assert result["mode"] == "offline"
    assert result["metadata"]["quantization"] == "fp8"
    assert result["metrics"]["tokens"]["cached"] == 8
    assert result["engine_metrics"]["engine"]["model"] == "/model"
    assert result["engine_metrics"]["execution_modes"] == {}
