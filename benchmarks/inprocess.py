from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from benchmarks.metrics import distribution, summarize
from benchmarks.models import RequestResult
from nanovllm import LLM, SamplingParams
from nanovllm.engine.cudagraph import ExecutionMode


@dataclass(slots=True)
class RequestRecord:
    seq_id: int
    input_tokens: int
    target_output_tokens: int
    arrival_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)

    def metric_result(self) -> RequestResult:
        if self.finish_s is None:
            raise RuntimeError(f"request {self.seq_id} has not finished")
        return RequestResult(
            request_id=str(self.seq_id),
            scheduled_s=self.arrival_s,
            started_s=self.arrival_s,
            finished_s=self.finish_s,
            first_content_s=self.first_token_s,
            chunk_times_s=self.token_times_s,
            status_code=200,
            prompt_tokens=self.input_tokens,
            completion_tokens=len(self.token_times_s),
            token_count_source="engine",
            saw_done=True,
        )


@dataclass(slots=True)
class PhaseStats:
    steps: int = 0
    seconds: float = 0.0
    tokens: int = 0

    def add(self, tokens: int, seconds: float) -> None:
        if tokens:
            self.steps += 1
            self.seconds += seconds
            self.tokens += tokens

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "seconds": self.seconds,
            "tokens": self.tokens,
            "token_per_s": self.tokens / self.seconds if self.seconds else 0.0,
        }


def _speculative_counters() -> dict[str, int]:
    return {
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


@dataclass(slots=True)
class RunStats:
    records: dict[int, RequestRecord] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)
    completed: int = 0
    prefill: PhaseStats = field(default_factory=PhaseStats)
    decode: PhaseStats = field(default_factory=PhaseStats)
    mixed: PhaseStats = field(default_factory=PhaseStats)
    speculative: dict[str, int] = field(default_factory=_speculative_counters)
    execution_modes: dict[str, dict] = field(default_factory=lambda: {
        mode.value: {"steps": 0, "seconds": 0.0, "tokens": 0, "padded_tokens": 0}
        for mode in ExecutionMode
    })
    scheduled_actual_tokens: int = 0
    scheduled_padded_tokens: int = 0
    running_request_seconds: float = 0.0
    max_batch_size: int = 0
    max_batched_tokens: int = 0

    def add_runner_metrics(self, metrics, step_seconds: float) -> None:
        self.scheduled_actual_tokens += metrics.real_tokens
        self.scheduled_padded_tokens += metrics.padded_tokens
        self.running_request_seconds += metrics.num_requests * step_seconds
        for key in self.speculative:
            self.speculative[key] += getattr(
                metrics.speculative, key.removesuffix("_tokens")
            )
        mode = self.execution_modes[metrics.execution_mode]
        mode["steps"] += 1
        mode["seconds"] += step_seconds
        mode["tokens"] += metrics.real_tokens
        mode["padded_tokens"] += metrics.padded_tokens

    def add_batch(self, batch, step_seconds: float) -> None:
        self.max_batch_size = max(self.max_batch_size, len(batch.sequences))
        self.max_batched_tokens = max(self.max_batched_tokens, batch.total_tokens)
        self.prefill.add(batch.prefill_tokens, step_seconds)
        self.decode.add(batch.decode_tokens, step_seconds)
        if batch.prefill_tokens and batch.decode_tokens:
            self.mixed.add(batch.total_tokens, step_seconds)

    def finish_sequences(self, seqs, previous_tokens: dict[int, int], timestamp: float) -> None:
        for seq in seqs:
            record = self.records[seq.seq_id]
            new_tokens = seq.num_completion_tokens - previous_tokens[seq.seq_id]
            if new_tokens:
                record.token_times_s.extend([timestamp] * new_tokens)
                record.first_token_s = record.first_token_s or timestamp
            if seq.is_finished:
                record.finish_s = timestamp
                self.active.remove(seq.seq_id)
                self.completed += 1


@dataclass(slots=True)
class BenchmarkRun:
    llm: LLM
    stats: RunStats
    args: object
    init_seconds: float
    warmup_seconds: float
    duration: float
    total_input_tokens: int
    target_output_tokens: int
    max_concurrency: int

    @property
    def records(self) -> list[RequestRecord]:
        records = list(self.stats.records.values())
        if len(records) != self.args.num_requests or any(item.finish_s is None for item in records):
            raise RuntimeError("benchmark ended before all requests completed")
        return records


def _sample_length(rng: random.Random, target: int, ratio: float) -> int:
    return rng.randint(max(1, int(target * (1 - ratio))), target)


def _make_workload(args, vocab_size: int):
    rng = random.Random(args.seed)
    prefix = [rng.randrange(vocab_size) for _ in range(args.shared_prefix_len)]
    workload = []
    for _ in range(args.num_requests):
        input_len = _sample_length(rng, args.input_len, args.range_ratio)
        output_len = _sample_length(rng, args.output_len, args.range_ratio)
        prompt = prefix + [rng.randrange(vocab_size) for _ in range(input_len - len(prefix))]
        params = SamplingParams(
            temperature=args.temperature, max_tokens=output_len,
            ignore_eos=args.ignore_eos,
        )
        workload.append((prompt, params))
    return workload


def _create_engine(args) -> tuple[LLM, float]:
    started = perf_counter()
    llm = LLM(
        args.model, quantization=args.quantization,
        gptq_kernel_backend=args.gptq_kernel_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        delta_state_dtype=args.delta_state_dtype,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        mtp_model=args.mtp_model, enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        master_port=args.master_port, cudagraph_mode=args.cudagraph_mode,
        piecewise_max_tokens=args.piecewise_max_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return llm, perf_counter() - started


def _warmup(llm: LLM, args) -> float:
    started = perf_counter()
    prompt = [1] * min(args.input_len, 32)
    llm.generate(
        [prompt] * args.warmup_num_requests,
        SamplingParams(
            temperature=args.temperature, max_tokens=args.warmup_output_len,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    llm.scheduler.num_preemptions = 0
    llm.model_runner.max_active_delta_states = 0
    return perf_counter() - started


def _admit_requests(llm, args, workload, stats, next_arrival, rng) -> float:
    now = perf_counter()
    while workload and len(stats.active) < (args.max_concurrency or args.num_requests):
        if now < next_arrival:
            break
        prompt, params = workload.popleft()
        arrival = perf_counter()
        seq_id = llm.add_request(prompt, params)
        stats.records[seq_id] = RequestRecord(
            seq_id, len(prompt), params.max_tokens, arrival
        )
        stats.active.add(seq_id)
        next_arrival = arrival if math.isinf(args.request_rate) else (
            max(next_arrival, arrival) + rng.expovariate(args.request_rate)
        )
        now = perf_counter()
    return next_arrival


def _execute_step(llm: LLM, stats: RunStats) -> None:
    started = perf_counter()
    batch = llm.scheduler.schedule()
    previous = {
        seq.seq_id: seq.num_completion_tokens for seq in batch.sequences
    }
    _, metrics = llm.execute_batch(batch)
    finished = perf_counter()
    seconds = finished - started
    stats.add_runner_metrics(metrics, seconds)
    stats.add_batch(batch, seconds)
    stats.finish_sequences(batch.sequences, previous, finished)


def run_engine(args) -> BenchmarkRun:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    llm, init_seconds = _create_engine(args)
    warmup_seconds = _warmup(llm, args)
    workload = deque(_make_workload(args, llm.model_runner.config.hf_config.vocab_size))
    total_input = sum(len(prompt) for prompt, _ in workload)
    target_output = sum(params.max_tokens for _, params in workload)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    stats = RunStats()
    rng = random.Random(args.seed + 1)
    started = next_arrival = perf_counter()
    while stats.completed < args.num_requests:
        next_arrival = _admit_requests(
            llm, args, workload, stats, next_arrival, rng
        )
        if llm.scheduler.is_finished():
            if workload:
                time.sleep(max(0.0, next_arrival - perf_counter()))
                continue
            break
        _execute_step(llm, stats)
    return BenchmarkRun(
        llm, stats, args, init_seconds, warmup_seconds,
        perf_counter() - started, total_input, target_output,
        args.max_concurrency or args.num_requests,
    )


def _engine_section(run: BenchmarkRun) -> dict:
    args, config = run.args, run.llm.model_runner.config
    return {
        "model": str(Path(args.model).resolve()),
        "model_dtype": str(config.hf_config.dtype).removeprefix("torch."),
        "model_family": config.model_family,
        "quantization": config.quantization or "none",
        "gptq_kernel_backend": config.gptq_kernel_backend,
        "kv_cache_dtype": config.kv_cache_dtype,
        "delta_state_dtype": config.delta_state_dtype,
        "speculative_method": config.speculative_method,
        "num_speculative_tokens": config.num_speculative_tokens,
        "mtp_model": config.mtp_model,
        "kv_cache_storage_dtype": config.kvcache_storage_dtype,
        "kv_cache_scale_mode": "per_token_per_kv_head"
        if config.kv_cache_dtype == "fp8_e4m3" else "none",
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
    }


def _speculative_section(run: BenchmarkRun) -> dict:
    stats = run.stats.speculative
    rounds = stats["verification_rounds"]
    result = {
        **stats,
        "rejected_prefix_target_replays": run.llm.model_runner.hybrid_state.rejected_prefix_target_replays,
        "state_branch_commits": run.llm.model_runner.hybrid_state.branch_commits,
        "state_branch_discards": run.llm.model_runner.hybrid_state.branch_discards,
        "acceptance_rate": stats["accepted_tokens"] / stats["proposed_tokens"]
        if stats["proposed_tokens"] else 0.0,
        "average_accepted_length": stats["accepted_tokens"] / max(1, rounds),
    }
    result["position_acceptance_rate"] = {
        str(index): stats[f"accepted_position_{index}"] / rounds if rounds else 0.0
        for index in (1, 2, 3)
    }
    return result


def _scheduler_section(run: BenchmarkRun) -> dict:
    stats, config = run.stats, run.llm.model_runner.config
    cached = max(0, run.total_input_tokens - stats.prefill.tokens)
    return {
        "preemptions": run.llm.scheduler.num_preemptions,
        "prefix_cache_enabled": config.enable_prefix_cache,
        "max_batch_size": stats.max_batch_size,
        "max_batched_tokens": stats.max_batched_tokens,
        "average_running_requests": stats.running_request_seconds / run.duration,
        "scheduled_actual_tokens": stats.scheduled_actual_tokens,
        "scheduled_padded_tokens": stats.scheduled_padded_tokens,
        "padding_ratio": (
            (stats.scheduled_padded_tokens - stats.scheduled_actual_tokens)
            / stats.scheduled_padded_tokens if stats.scheduled_padded_tokens else 0.0
        ),
        "cached_prompt_tokens": cached,
        "prefix_cache_hit_rate": cached / run.total_input_tokens
        if run.total_input_tokens else 0.0,
    }


def _memory_section(run: BenchmarkRun) -> dict:
    runner, config = run.llm.model_runner, run.llm.model_runner.config
    model = runner.model
    storage = sum(item.numel() * item.element_size() for item in model.parameters())
    storage += sum(item.numel() * item.element_size() for item in model.buffers())
    return {
        "model_storage_mib": storage / 2**20,
        "kv_cache_dtype": config.kv_cache_dtype,
        "kv_cache_payload_bytes_per_block": config.kvcache_payload_bytes,
        "kv_cache_storage_dtype": config.kvcache_storage_dtype,
        "kv_cache_scale_bytes_per_block": config.kvcache_scale_bytes,
        "mtp_kv_cache_bytes_per_block": config.mtp_kvcache_bytes,
        "kv_cache_bytes_per_block": config.kvcache_block_bytes,
        "kv_cache_blocks": config.num_kvcache_blocks,
        "kv_cache_token_capacity": config.num_kvcache_blocks * config.kvcache_block_size,
        "kv_cache_scale_overhead_ratio": config.kvcache_scale_bytes / config.kvcache_block_bytes
        if config.kvcache_block_bytes else 0.0,
        "delta_state_per_request_mib": model.delta_state_bytes(config.hf_config.dtype) / 2**20
        if hasattr(model, "delta_state_bytes") else 0.0,
        "delta_state_capacity": runner.delta_state_capacity,
        "max_active_delta_states": runner.max_active_delta_states,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def build_result(run: BenchmarkRun) -> dict:
    args, records = run.args, run.records
    request_metrics = summarize(
        [record.metric_result() for record in records],
        args.ttft_slo_ms, args.tpot_slo_ms, args.e2e_slo_ms,
    )
    actual_output = sum(len(record.token_times_s) for record in records)
    modes = {
        mode: {**stats, "token_per_s": stats["tokens"] / stats["seconds"]
               if stats["seconds"] else 0.0}
        for mode, stats in run.stats.execution_modes.items()
    }
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": args.tensor_parallel_size,
            "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        },
        "engine": _engine_section(run),
        "workload": _workload_section(run, records, actual_output),
        "throughput": _throughput_section(run, actual_output),
        "latency_ms": {
            "ttft": request_metrics["latency_ms"]["ttft"],
            "tpot": request_metrics["latency_ms"]["tpot"],
            "itl": request_metrics["latency_ms"]["inter_chunk"],
            "e2e": request_metrics["latency_ms"]["e2e"],
        },
        "phases": {name: getattr(run.stats, name).report()
                   for name in ("prefill", "decode", "mixed")},
        "execution_modes": modes,
        "speculative": _speculative_section(run),
        "scheduler": _scheduler_section(run),
        "memory": _memory_section(run),
        "startup": {
            "engine_init_s": run.init_seconds,
            "explicit_warmup_s": run.warmup_seconds,
            "warmup_num_requests": args.warmup_num_requests,
        },
        "slo": {**request_metrics["slo"], "delta_state_dtype": args.delta_state_dtype},
    }
    if args.request_details:
        result["requests"] = [_request_detail(item) for item in records]
    return result


def _workload_section(run, records, actual_output) -> dict:
    args = run.args
    return {
        "num_requests": args.num_requests,
        "request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
        "max_concurrency": run.max_concurrency,
        "seed": args.seed, "range_ratio": args.range_ratio,
        "shared_prefix_len": args.shared_prefix_len, "ignore_eos": args.ignore_eos,
        "input_tokens": distribution([item.input_tokens for item in records]),
        "output_tokens": distribution([len(item.token_times_s) for item in records]),
        "total_input_tokens": run.total_input_tokens,
        "target_output_tokens": run.target_output_tokens,
        "actual_output_tokens": actual_output,
    }


def _throughput_section(run, actual_output) -> dict:
    stats = run.stats.speculative
    return {
        "benchmark_duration_s": run.duration,
        "request_per_s": run.args.num_requests / run.duration,
        "input_token_per_s": run.total_input_tokens / run.duration,
        "output_token_per_s": actual_output / run.duration,
        "total_token_per_s": (run.total_input_tokens + actual_output) / run.duration,
        "accepted_token_per_s": stats["accepted_tokens"] / run.duration,
    }


def _request_detail(record: RequestRecord) -> dict:
    return {
        **asdict(record),
        "output_tokens": len(record.token_times_s),
        "ttft_ms": (record.first_token_s - record.arrival_s) * 1000,
        "e2e_ms": (record.finish_s - record.arrival_s) * 1000,
    }


def _format_distribution(name: str, values: dict[str, float]) -> None:
    print(
        f"{name:<8} mean={values['mean']:.2f}ms  p50={values['p50']:.2f}ms  "
        f"p90={values['p90']:.2f}ms  p95={values['p95']:.2f}ms  "
        f"p99={values['p99']:.2f}ms  max={values['max']:.2f}ms"
    )


def print_result(result: dict) -> None:
    print("\n============ nano-vLLM Benchmark ============")
    print(f"Model: {result['engine']['model']}")
    print(f"Dtype / quantization: {result['engine']['model_dtype']} / {result['engine']['quantization']}")
    workload, throughput = result["workload"], result["throughput"]
    print(
        f"Requests: {workload['num_requests']}  Input tokens: {workload['total_input_tokens']}  "
        f"Output tokens: {workload['actual_output_tokens']}  "
        f"Duration: {throughput['benchmark_duration_s']:.3f}s"
    )
    print("\n--- Throughput ---")
    for label, key, unit in (
        ("Request", "request_per_s", "req/s"),
        ("Input token", "input_token_per_s", "tok/s"),
        ("Output token", "output_token_per_s", "tok/s"),
        ("Total token", "total_token_per_s", "tok/s"),
    ):
        print(f"{label + ' throughput:':24} {throughput[key]:.2f} {unit}")
    print("\n--- Latency ---")
    for name in ("ttft", "tpot", "itl", "e2e"):
        _format_distribution(name.upper(), result["latency_ms"][name])
    _print_runtime_sections(result)


def _print_runtime_sections(result: dict) -> None:
    print("\n--- Phase / Scheduler ---")
    for name, stats in result["phases"].items():
        print(
            f"{name.title():<8}{stats['tokens']} tok / {stats['seconds']:.3f}s = "
            f"{stats['token_per_s']:.2f} tok/s"
        )
    for mode, stats in result["execution_modes"].items():
        print(
            f"{mode:<9} {stats['tokens']} tok / {stats['seconds']:.3f}s = "
            f"{stats['token_per_s']:.2f} tok/s ({stats['steps']} steps)"
        )
    scheduler, memory, startup = result["scheduler"], result["memory"], result["startup"]
    print(
        f"Max batch: {scheduler['max_batch_size']}  Max batched tokens: "
        f"{scheduler['max_batched_tokens']}  Preemptions: {scheduler['preemptions']}  "
        f"Prefix cache hit: {scheduler['prefix_cache_hit_rate']:.2%}"
    )
    print("\n--- Memory / Startup ---")
    print(
        f"Model: {memory['model_storage_mib']:.2f} MiB  Peak allocated/reserved: "
        f"{memory['peak_allocated_mib']:.2f} / {memory['peak_reserved_mib']:.2f} MiB"
    )
    print(
        f"KV capacity: {memory['kv_cache_token_capacity']} tokens  "
        f"Init: {startup['engine_init_s']:.3f}s  Warmup: {startup['explicit_warmup_s']:.3f}s"
    )
    if result["slo"]["enabled"]:
        print(
            f"SLO goodput: {result['slo']['goodput_request_per_s']:.2f} req/s "
            f"({result['slo']['good_requests']}/{result['workload']['num_requests']} requests)"
        )


def run_inprocess(args) -> dict:
    run = run_engine(args)
    result = build_result(run)
    print_result(result)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON: {path}")
    return result
