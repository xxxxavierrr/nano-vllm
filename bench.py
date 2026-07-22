import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.cudagraph import CUDAGraphMode, ExecutionMode


DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B"


@dataclass
class RequestRecord:
    seq_id: int
    input_tokens: int
    target_output_tokens: int
    arrival_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


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


def sample_length(rng: random.Random, target: int, range_ratio: float) -> int:
    lower = max(1, int(target * (1 - range_ratio)))
    return rng.randint(lower, target)


def make_workload(args, vocab_size: int):
    rng = random.Random(args.seed)
    shared_prefix = [rng.randrange(vocab_size) for _ in range(args.shared_prefix_len)]
    workload = []
    for _ in range(args.num_requests):
        input_len = sample_length(rng, args.input_len, args.range_ratio)
        output_len = sample_length(rng, args.output_len, args.range_ratio)
        suffix_len = input_len - len(shared_prefix)
        prompt = shared_prefix + [rng.randrange(vocab_size) for _ in range(suffix_len)]
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=output_len,
            ignore_eos=args.ignore_eos,
        )
        workload.append((prompt, sampling_params))
    return workload


def format_distribution(name: str, values: dict[str, float], unit: str = "ms"):
    print(
        f"{name:<8} mean={values['mean']:.2f}{unit}  p50={values['p50']:.2f}{unit}  "
        f"p90={values['p90']:.2f}{unit}  p95={values['p95']:.2f}{unit}  "
        f"p99={values['p99']:.2f}{unit}  max={values['max']:.2f}{unit}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Comprehensive in-process nano-vLLM benchmark")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Local Hugging Face model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--label", default="nano-vllm")
    parser.add_argument("--quantization", choices=["fp8", "gptq"])
    parser.add_argument(
        "--gptq-kernel-backend",
        choices=["auto", "triton", "marlin"],
        default="auto",
    )
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8_e4m3"], default="auto")
    parser.add_argument(
        "--speculative-method", choices=["none", "mtp"], default="none"
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--mtp-model")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--range-ratio", type=float, default=0.0)
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=math.inf, help="Requests/s; inf is offline burst")
    parser.add_argument("--max-concurrency", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument("--warmup-num-requests", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cudagraph-mode",
        choices=[mode.value for mode in CUDAGraphMode],
        default=CUDAGraphMode.FULL_AND_PIECEWISE.value,
    )
    parser.add_argument("--piecewise-max-tokens", type=int, default=512)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--output-json")
    parser.add_argument("--request-details", action="store_true")
    args = parser.parse_args(argv)

    if args.num_requests <= 0 or args.input_len <= 0 or args.output_len <= 0:
        parser.error("request count and token lengths must be positive")
    if not 0 <= args.range_ratio < 1:
        parser.error("--range-ratio must be in [0, 1)")
    min_input_len = max(1, int(args.input_len * (1 - args.range_ratio)))
    if not 0 <= args.shared_prefix_len <= min_input_len:
        parser.error("--shared-prefix-len cannot exceed the minimum sampled input length")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input_len + output_len cannot exceed --max-model-len")
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    if args.max_concurrency < 0:
        parser.error("--max-concurrency cannot be negative")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    if args.piecewise_max_tokens <= 0:
        parser.error("--piecewise-max-tokens must be positive")
    if args.warmup_num_requests <= 0:
        parser.error("--warmup-num-requests must be positive")
    if args.speculative_method == "mtp":
        if args.num_speculative_tokens not in (1, 2, 3):
            parser.error("--num-speculative-tokens must be 1, 2, or 3")
        if args.temperature != 0:
            parser.error("current MTP milestone requires --temperature 0")
        if args.mtp_model is not None and not Path(args.mtp_model).is_dir():
            parser.error(f"MTP model directory does not exist: {args.mtp_model}")
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    init_started = perf_counter()
    llm = LLM(
        args.model,
        quantization=args.quantization,
        gptq_kernel_backend=args.gptq_kernel_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        mtp_model=args.mtp_model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        master_port=args.master_port,
        cudagraph_mode=args.cudagraph_mode,
        piecewise_max_tokens=args.piecewise_max_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    init_seconds = perf_counter() - init_started

    warmup_started = perf_counter()
    warmup_prompt = [1] * min(args.input_len, 32)
    llm.generate(
        [warmup_prompt] * args.warmup_num_requests,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.warmup_output_len,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    warmup_seconds = perf_counter() - warmup_started
    llm.scheduler.num_preemptions = 0
    llm.model_runner.max_active_delta_states = 0

    workload = deque(make_workload(args, llm.model_runner.config.hf_config.vocab_size))
    total_input_tokens = sum(len(prompt) for prompt, _ in workload)
    target_output_tokens = sum(params.max_tokens for _, params in workload)
    max_concurrency = args.max_concurrency or args.num_requests

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    records: dict[int, RequestRecord] = {}
    active_ids: set[int] = set()
    completed = 0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    prefill_steps = 0
    decode_steps = 0
    mixed_seconds = 0.0
    mixed_tokens = 0
    mixed_steps = 0
    scheduled_actual_tokens = 0
    scheduled_padded_tokens = 0
    running_request_seconds = 0.0
    speculative_stats = {
        "drafted_tokens": 0,
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "bonus_tokens": 0,
        "verification_rounds": 0,
        "accepted_position_1": 0,
        "accepted_position_2": 0,
        "accepted_position_3": 0,
    }
    execution_mode_stats = {
        mode.value: {"steps": 0, "seconds": 0.0, "tokens": 0}
        for mode in ExecutionMode
    }
    max_batch_size = 0
    max_batched_tokens = 0
    arrival_rng = random.Random(args.seed + 1)
    benchmark_started = perf_counter()
    next_arrival = benchmark_started

    while completed < args.num_requests:
        now = perf_counter()
        while workload and len(active_ids) < max_concurrency and now >= next_arrival:
            prompt, sampling_params = workload.popleft()
            arrival = perf_counter()
            seq_id = llm.add_request(prompt, sampling_params)
            records[seq_id] = RequestRecord(
                seq_id=seq_id,
                input_tokens=len(prompt),
                target_output_tokens=sampling_params.max_tokens,
                arrival_s=arrival,
            )
            active_ids.add(seq_id)
            if math.isinf(args.request_rate):
                next_arrival = arrival
            else:
                next_arrival = max(next_arrival, arrival) + arrival_rng.expovariate(args.request_rate)
            now = perf_counter()

        if llm.scheduler.is_finished():
            if workload:
                delay = max(0.0, next_arrival - perf_counter())
                if delay:
                    time.sleep(delay)
                continue
            break

        step_started = perf_counter()
        batch = llm.scheduler.schedule()
        seqs = batch.sequences
        scheduled_tokens = batch.total_tokens
        previous_completion_tokens = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        _, runner_metrics = llm.execute_batch(batch)
        step_finished = perf_counter()
        step_seconds = step_finished - step_started
        execution_mode = runner_metrics.execution_mode
        step_speculative = runner_metrics.speculative
        scheduled_actual_tokens += runner_metrics.real_tokens
        scheduled_padded_tokens += runner_metrics.padded_tokens
        running_request_seconds += runner_metrics.num_requests * step_seconds
        for key in speculative_stats:
            runner_key = key.removesuffix("_tokens")
            speculative_stats[key] += getattr(step_speculative, runner_key)
        mode_stats = execution_mode_stats[execution_mode]
        mode_stats["steps"] += 1
        mode_stats["seconds"] += step_seconds
        mode_stats["tokens"] += runner_metrics.real_tokens
        mode_stats.setdefault("padded_tokens", 0)
        mode_stats["padded_tokens"] += runner_metrics.padded_tokens

        max_batch_size = max(max_batch_size, len(seqs))
        max_batched_tokens = max(max_batched_tokens, scheduled_tokens)
        if batch.prefill_tokens:
            prefill_steps += 1
            prefill_seconds += step_seconds
            prefill_tokens += batch.prefill_tokens
        if batch.decode_tokens:
            decode_steps += 1
            decode_seconds += step_seconds
            decode_tokens += batch.decode_tokens
        if batch.prefill_tokens and batch.decode_tokens:
            mixed_steps += 1
            mixed_seconds += step_seconds
            mixed_tokens += scheduled_tokens

        for seq in seqs:
            record = records[seq.seq_id]
            new_tokens = seq.num_completion_tokens - previous_completion_tokens[seq.seq_id]
            if new_tokens:
                record.token_times_s.extend([step_finished] * new_tokens)
                if record.first_token_s is None:
                    record.first_token_s = step_finished
            if seq.is_finished:
                record.finish_s = step_finished
                active_ids.remove(seq.seq_id)
                completed += 1

    benchmark_finished = perf_counter()
    duration = benchmark_finished - benchmark_started
    finished_records = list(records.values())
    if len(finished_records) != args.num_requests or any(record.finish_s is None for record in finished_records):
        raise RuntimeError("benchmark ended before all requests completed")

    actual_output_tokens = sum(len(record.token_times_s) for record in finished_records)
    ttfts = [record.first_token_s - record.arrival_s for record in finished_records]
    e2es = [record.finish_s - record.arrival_s for record in finished_records]
    tpots = [
        (record.finish_s - record.first_token_s) / (len(record.token_times_s) - 1)
        for record in finished_records
        if len(record.token_times_s) > 1
    ]
    itls = [
        current - previous
        for record in finished_records
        for previous, current in zip(record.token_times_s, record.token_times_s[1:])
    ]

    ttft_ms = [value * 1000 for value in ttfts]
    tpot_ms = [value * 1000 for value in tpots]
    e2e_ms = [value * 1000 for value in e2es]
    good_requests = 0
    good_output_tokens = 0
    slo_enabled = any(value is not None for value in (args.ttft_slo_ms, args.tpot_slo_ms, args.e2e_slo_ms))
    for index, record in enumerate(finished_records):
        request_tpot_ms = 0.0
        if len(record.token_times_s) > 1:
            request_tpot_ms = (record.finish_s - record.first_token_s) * 1000 / (len(record.token_times_s) - 1)
        meets_slo = (
            (args.ttft_slo_ms is None or ttft_ms[index] <= args.ttft_slo_ms)
            and (args.tpot_slo_ms is None or request_tpot_ms <= args.tpot_slo_ms)
            and (args.e2e_slo_ms is None or e2e_ms[index] <= args.e2e_slo_ms)
        )
        good_requests += int(meets_slo)
        if meets_slo:
            good_output_tokens += len(record.token_times_s)

    model = llm.model_runner.model
    parameter_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    config = llm.model_runner.config
    gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
    cached_prompt_tokens = max(0, total_input_tokens - prefill_tokens)

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": args.tensor_parallel_size,
            "gpu_memory_mib": gpu_total_bytes / 2**20,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "engine": {
            "model": str(Path(args.model).resolve()),
            "model_dtype": str(config.hf_config.dtype).removeprefix("torch."),
            "model_family": config.model_family,
            "quantization": config.quantization or "none",
            "gptq_kernel_backend": config.gptq_kernel_backend,
            "kv_cache_dtype": config.kv_cache_dtype,
            "speculative_method": config.speculative_method,
            "num_speculative_tokens": config.num_speculative_tokens,
            "mtp_model": config.mtp_model,
            "kv_cache_storage_dtype": config.kvcache_storage_dtype,
            "kv_cache_scale_mode": (
                "per_token_per_kv_head"
                if config.kv_cache_dtype == "fp8_e4m3"
                else "none"
            ),
            "tensor_parallel_size": args.tensor_parallel_size,
            "master_port": args.master_port,
            "enforce_eager": args.enforce_eager,
            "cudagraph_mode": config.cudagraph_mode.value,
            "piecewise_max_tokens": config.piecewise_max_tokens,
            "requested_max_num_seqs": args.max_num_seqs,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "workload": {
            "num_requests": args.num_requests,
            "request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
            "max_concurrency": max_concurrency,
            "seed": args.seed,
            "range_ratio": args.range_ratio,
            "shared_prefix_len": args.shared_prefix_len,
            "ignore_eos": args.ignore_eos,
            "input_tokens": distribution([record.input_tokens for record in finished_records]),
            "output_tokens": distribution([len(record.token_times_s) for record in finished_records]),
            "total_input_tokens": total_input_tokens,
            "target_output_tokens": target_output_tokens,
            "actual_output_tokens": actual_output_tokens,
        },
        "throughput": {
            "benchmark_duration_s": duration,
            "request_per_s": args.num_requests / duration,
            "input_token_per_s": total_input_tokens / duration,
            "output_token_per_s": actual_output_tokens / duration,
            "total_token_per_s": (total_input_tokens + actual_output_tokens) / duration,
            "accepted_token_per_s": speculative_stats["accepted_tokens"] / duration,
        },
        "latency_ms": {
            "ttft": distribution(ttfts, 1000),
            "tpot": distribution(tpots, 1000),
            "itl": distribution(itls, 1000),
            "e2e": distribution(e2es, 1000),
        },
        "phases": {
            "prefill": {
                "steps": prefill_steps,
                "seconds": prefill_seconds,
                "tokens": prefill_tokens,
                "token_per_s": prefill_tokens / prefill_seconds if prefill_seconds else 0.0,
            },
            "decode": {
                "steps": decode_steps,
                "seconds": decode_seconds,
                "tokens": decode_tokens,
                "token_per_s": decode_tokens / decode_seconds if decode_seconds else 0.0,
            },
            "mixed": {
                "steps": mixed_steps,
                "seconds": mixed_seconds,
                "tokens": mixed_tokens,
                "token_per_s": mixed_tokens / mixed_seconds if mixed_seconds else 0.0,
            },
        },
        "execution_modes": {
            mode: {
                **stats,
                "token_per_s": stats["tokens"] / stats["seconds"]
                if stats["seconds"]
                else 0.0,
            }
            for mode, stats in execution_mode_stats.items()
        },
        "speculative": {
            **speculative_stats,
            "acceptance_rate": (
                speculative_stats["accepted_tokens"]
                / speculative_stats["proposed_tokens"]
                if speculative_stats["proposed_tokens"]
                else 0.0
            ),
            "average_accepted_length": (
                speculative_stats["accepted_tokens"]
                / max(1, speculative_stats["verification_rounds"])
            ),
            "position_acceptance_rate": {
                "1": (
                    speculative_stats["accepted_position_1"]
                    / speculative_stats["verification_rounds"]
                    if speculative_stats["verification_rounds"]
                    else 0.0
                ),
                "2": (
                    speculative_stats["accepted_position_2"]
                    / speculative_stats["verification_rounds"]
                    if speculative_stats["verification_rounds"]
                    else 0.0
                ),
                "3": (
                    speculative_stats["accepted_position_3"]
                    / speculative_stats["verification_rounds"]
                    if speculative_stats["verification_rounds"]
                    else 0.0
                ),
            },
        },
        "scheduler": {
            "preemptions": llm.scheduler.num_preemptions,
            "prefix_cache_enabled": config.enable_prefix_cache,
            "max_batch_size": max_batch_size,
            "max_batched_tokens": max_batched_tokens,
            "average_running_requests": running_request_seconds / duration,
            "scheduled_actual_tokens": scheduled_actual_tokens,
            "scheduled_padded_tokens": scheduled_padded_tokens,
            "padding_ratio": (
                (scheduled_padded_tokens - scheduled_actual_tokens)
                / scheduled_padded_tokens
                if scheduled_padded_tokens
                else 0.0
            ),
            "cached_prompt_tokens": cached_prompt_tokens,
            "prefix_cache_hit_rate": cached_prompt_tokens / total_input_tokens if total_input_tokens else 0.0,
        },
        "memory": {
            "model_storage_mib": (parameter_bytes + buffer_bytes) / 2**20,
            "kv_cache_dtype": config.kv_cache_dtype,
            "kv_cache_payload_bytes_per_block": config.kvcache_payload_bytes,
            "kv_cache_storage_dtype": config.kvcache_storage_dtype,
            "kv_cache_scale_bytes_per_block": config.kvcache_scale_bytes,
            "mtp_kv_cache_bytes_per_block": config.mtp_kvcache_bytes,
            "kv_cache_bytes_per_block": config.kvcache_block_bytes,
            "kv_cache_blocks": config.num_kvcache_blocks,
            "kv_cache_token_capacity": config.num_kvcache_blocks * config.kvcache_block_size,
            "delta_state_per_request_mib": (
                model.delta_state_bytes(config.hf_config.dtype) / 2**20
                if hasattr(model, "delta_state_bytes") else 0.0
            ),
            "delta_state_capacity": llm.model_runner.delta_state_capacity,
            "max_active_delta_states": llm.model_runner.max_active_delta_states,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        },
        "startup": {
            "engine_init_s": init_seconds,
            "explicit_warmup_s": warmup_seconds,
            "warmup_num_requests": args.warmup_num_requests,
        },
        "slo": {
            "enabled": slo_enabled,
            "ttft_ms": args.ttft_slo_ms,
            "tpot_ms": args.tpot_slo_ms,
            "e2e_ms": args.e2e_slo_ms,
            "good_requests": good_requests if slo_enabled else None,
            "goodput_request_per_s": good_requests / duration if slo_enabled else None,
            "good_output_tokens": good_output_tokens if slo_enabled else None,
            "goodput_output_token_per_s": (
                good_output_tokens / duration if slo_enabled else None
            ),
            "attainment": good_requests / args.num_requests if slo_enabled else None,
        },
    }
    if args.request_details:
        result["requests"] = [
            {
                **asdict(record),
                "output_tokens": len(record.token_times_s),
                "ttft_ms": (record.first_token_s - record.arrival_s) * 1000,
                "e2e_ms": (record.finish_s - record.arrival_s) * 1000,
            }
            for record in finished_records
        ]

    print("\n============ nano-vLLM Benchmark ============")
    print(f"Model: {result['engine']['model']}")
    print(f"Dtype / quantization: {result['engine']['model_dtype']} / {result['engine']['quantization']}")
    print(
        f"Requests: {args.num_requests}  Input tokens: {total_input_tokens}  "
        f"Output tokens: {actual_output_tokens}  Duration: {duration:.3f}s"
    )
    print("\n--- Throughput ---")
    print(f"Request throughput:      {result['throughput']['request_per_s']:.2f} req/s")
    print(f"Input token throughput:  {result['throughput']['input_token_per_s']:.2f} tok/s")
    print(f"Output token throughput: {result['throughput']['output_token_per_s']:.2f} tok/s")
    print(f"Total token throughput:  {result['throughput']['total_token_per_s']:.2f} tok/s")
    print("\n--- Latency ---")
    format_distribution("TTFT", result["latency_ms"]["ttft"])
    format_distribution("TPOT", result["latency_ms"]["tpot"])
    format_distribution("ITL", result["latency_ms"]["itl"])
    format_distribution("E2E", result["latency_ms"]["e2e"])
    print("\n--- Phase / Scheduler ---")
    print(
        f"Prefill: {prefill_tokens} tok / {prefill_seconds:.3f}s = "
        f"{result['phases']['prefill']['token_per_s']:.2f} tok/s"
    )
    print(
        f"Decode:  {decode_tokens} tok / {decode_seconds:.3f}s = "
        f"{result['phases']['decode']['token_per_s']:.2f} tok/s"
    )
    print(
        f"Mixed:   {mixed_tokens} tok / {mixed_seconds:.3f}s = "
        f"{result['phases']['mixed']['token_per_s']:.2f} tok/s"
    )
    for mode, stats in result["execution_modes"].items():
        print(
            f"{mode:<9} {stats['tokens']} tok / {stats['seconds']:.3f}s = "
            f"{stats['token_per_s']:.2f} tok/s ({stats['steps']} steps)"
        )
    print(
        f"Max batch: {max_batch_size}  Max batched tokens: {max_batched_tokens}  "
        f"Preemptions: {llm.scheduler.num_preemptions}  Prefix cache hit: "
        f"{result['scheduler']['prefix_cache_hit_rate']:.2%}"
    )
    print("\n--- Memory / Startup ---")
    print(
        f"Model: {result['memory']['model_storage_mib']:.2f} MiB  "
        f"Peak allocated/reserved: {result['memory']['peak_allocated_mib']:.2f} / "
        f"{result['memory']['peak_reserved_mib']:.2f} MiB"
    )
    print(
        f"KV capacity: {result['memory']['kv_cache_token_capacity']} tokens  "
        f"Init: {init_seconds:.3f}s  Warmup: {warmup_seconds:.3f}s"
    )
    if slo_enabled:
        print(
            f"SLO goodput: {result['slo']['goodput_request_per_s']:.2f} req/s "
            f"({good_requests}/{args.num_requests} requests)"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON: {output_path}")

    return result


if __name__ == "__main__":
    main()
