import argparse
import gc
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_PROMPTS = [
    "Explain why the sky is blue in simple terms.",
    "Write a concise explanation of how virtual memory works.",
    "Compare solar and wind power, including one limitation of each.",
    "Give practical advice to someone learning Python for the first time.",
    "Describe how rain forms in language suitable for a young student.",
    "Explain what a database index does and when it can hurt performance.",
    "Summarize the causes of the seasons without using an analogy.",
    "Discuss two benefits and two risks of using automation at work.",
]


@dataclass
class RequestRecord:
    index: int
    arrival_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)


def parse_csv_ints(value: str, *, name: str, allow_zero: bool = False) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{name} values must be {qualifier}")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop Qwen3.6 MTP draft-length/concurrency sweep."
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4",
    )
    parser.add_argument(
        "--mtp-model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27B-mtp",
    )
    parser.add_argument("--k", default="0,1,2,3", help="0 is the baseline")
    parser.add_argument("--concurrency", default="1,2,4,6,8")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", "--max-tokens", dest="output_len", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="0 means two times the current concurrency",
    )
    parser.add_argument("--warmup-output-len", type=int, default=16)
    parser.add_argument(
        "--warmup-all-batch-sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm every active batch size up to each measured concurrency",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-jsonl")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="0 means input-len + output-len",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--master-port-base", type=int, default=2460)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--trace-steps", action="store_true")
    parser.add_argument("--require-exact-output", action="store_true")
    parser.add_argument("--output-json", default="/tmp/nanovllm-mtp-sweep.json")
    parser.add_argument(
        "--print-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    args.k = parse_csv_ints(args.k, name="--k", allow_zero=True)
    args.concurrency = parse_csv_ints(args.concurrency, name="--concurrency")
    if any(k not in (0, 1, 2, 3) for k in args.k):
        parser.error("--k currently supports only 0, 1, 2, and 3")
    positive = (
        args.input_len,
        args.output_len,
        args.num_requests,
        args.warmup_output_len,
        args.repeats,
        args.max_num_batched_tokens,
        args.max_num_seqs,
    )
    if any(value <= 0 for value in positive) or args.warmup_requests < 0:
        parser.error("lengths, counts, and capacities must be positive")
    if args.max_model_len == 0:
        args.max_model_len = args.input_len + args.output_len
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input-len + output-len cannot exceed max-model-len")
    if args.num_requests < max(args.concurrency):
        parser.error("num-requests must be at least the largest concurrency")
    for path in (args.model, args.mtp_model):
        if not Path(path).is_dir():
            parser.error(f"model directory does not exist: {path}")
    if args.prompt_jsonl and not Path(args.prompt_jsonl).is_file():
        parser.error(f"prompt JSONL does not exist: {args.prompt_jsonl}")
    return args


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


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def load_prompt_texts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            prompts.append(value)
        elif isinstance(value, dict) and isinstance(value.get("prompt"), str):
            prompts.append(value["prompt"])
        else:
            raise ValueError(f"JSONL line {line_number} must be a string or contain prompt")
    if not prompts:
        raise ValueError("prompt JSONL is empty")
    return prompts


def make_prompt_tokens(tokenizer, texts: list[str], input_len: int, count: int):
    filler = tokenizer.encode(
        " Provide a precise answer, state important assumptions, and avoid repetition.",
        add_special_tokens=False,
    )
    prompts = []
    for index in range(count):
        tokens = tokenizer.encode(texts[index % len(texts)], add_special_tokens=True)
        while len(tokens) < input_len:
            tokens.extend(filler)
        prompts.append(tokens[:input_len])
    return prompts


def empty_speculative_stats():
    return {
        "drafted": 0,
        "proposed": 0,
        "accepted": 0,
        "rejected": 0,
        "bonus": 0,
        "verification_rounds": 0,
        "accepted_position_1": 0,
        "accepted_position_2": 0,
        "accepted_position_3": 0,
    }


def accumulate_stats(total, stats):
    total["drafted"] += stats.speculative_drafted_tokens
    total["proposed"] += stats.speculative_proposed_tokens
    total["accepted"] += stats.speculative_accepted_tokens
    total["rejected"] += stats.speculative_rejected_tokens
    total["bonus"] += stats.speculative_bonus_tokens
    total["verification_rounds"] += stats.speculative_verification_rounds
    total["accepted_position_1"] += stats.speculative_accepted_position_1
    total["accepted_position_2"] += stats.speculative_accepted_position_2
    total["accepted_position_3"] += stats.speculative_accepted_position_3


def output_hash(records: dict[int, RequestRecord]) -> str:
    ordered = [records[seq_id] for seq_id in sorted(records, key=lambda item: records[item].index)]
    payload = json.dumps([record.token_ids for record in ordered], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def run_closed_loop(
    llm,
    prompts,
    concurrency,
    output_len,
    ttft_slo_ms,
    tpot_slo_ms,
    trace_steps=False,
):
    if not llm.is_finished():
        raise RuntimeError("engine must be idle before a benchmark run")
    params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    records: dict[int, RequestRecord] = {}
    active_ids: set[int] = set()
    next_prompt = 0
    completed = 0
    steps = 0
    max_active = 0
    speculative = empty_speculative_stats()
    step_trace = []
    llm.scheduler.num_preemptions = 0

    def refill():
        nonlocal next_prompt, max_active
        while next_prompt < len(prompts) and len(active_ids) < concurrency:
            arrival = perf_counter()
            seq_id = llm.add_request(prompts[next_prompt], params)
            records[seq_id] = RequestRecord(next_prompt, arrival)
            active_ids.add(seq_id)
            next_prompt += 1
        max_active = max(max_active, len(active_ids))

    torch.cuda.synchronize()
    started = perf_counter()
    refill()
    while completed < len(prompts):
        active_before = len(active_ids)
        step_started = perf_counter()
        outputs, stats = llm.step()
        torch.cuda.synchronize()
        now = perf_counter()
        step_seconds = now - step_started
        steps += 1
        accumulate_stats(speculative, stats)
        trace = {
            "step": steps,
            "active_requests": active_before,
            "seconds": step_seconds,
            "outputs": len(outputs),
            "scheduled_tokens": stats.total_tokens,
            "proposed": stats.speculative_proposed_tokens,
            "accepted": stats.speculative_accepted_tokens,
            "rejected": stats.speculative_rejected_tokens,
        }
        if trace_steps:
            step_trace.append(trace)
            print("step-trace " + json.dumps(trace), flush=True)
        for output in outputs:
            record = records[output.seq_id]
            record.token_ids.append(output.token_id)
            record.token_times_s.append(now)
            if record.first_token_s is None:
                record.first_token_s = now
            if output.finished:
                record.finish_s = now
                active_ids.remove(output.seq_id)
                completed += 1
        refill()
    duration = perf_counter() - started

    finished = sorted(records.values(), key=lambda record: record.index)
    if any(record.finish_s is None or record.first_token_s is None for record in finished):
        raise RuntimeError("benchmark finished with incomplete request records")
    expected_tokens = len(prompts) * output_len
    actual_tokens = sum(len(record.token_ids) for record in finished)
    if actual_tokens != expected_tokens:
        raise RuntimeError(f"expected {expected_tokens} tokens, observed {actual_tokens}")
    ttft_ms = [(record.first_token_s - record.arrival_s) * 1000 for record in finished]
    e2e_ms = [(record.finish_s - record.arrival_s) * 1000 for record in finished]
    tpot_ms = [
        (record.finish_s - record.first_token_s) * 1000 / (len(record.token_ids) - 1)
        for record in finished
        if len(record.token_ids) > 1
    ]
    good_requests = 0
    for index, record in enumerate(finished):
        request_tpot = 0.0
        if len(record.token_ids) > 1:
            request_tpot = (
                (record.finish_s - record.first_token_s) * 1000
                / (len(record.token_ids) - 1)
            )
        good_requests += int(
            (ttft_slo_ms is None or ttft_ms[index] <= ttft_slo_ms)
            and (tpot_slo_ms is None or request_tpot <= tpot_slo_ms)
        )
    return {
        "duration_s": duration,
        "requests": len(prompts),
        "output_tokens": actual_tokens,
        "steps": steps,
        "max_active_requests": max_active,
        "request_per_s": len(prompts) / duration,
        "output_token_per_s": actual_tokens / duration,
        "preemptions": llm.scheduler.num_preemptions,
        "good_requests": good_requests,
        "goodput_request_per_s": good_requests / duration,
        "output_hash": output_hash(records),
        "_output_token_ids": [record.token_ids for record in finished],
        "step_trace": step_trace,
        "speculative": speculative,
        "_ttft_ms": ttft_ms,
        "_tpot_ms": tpot_ms,
        "_e2e_ms": e2e_ms,
    }


def summarize_runs(k: int, concurrency: int, runs: list[dict], slo_enabled: bool):
    speculative = empty_speculative_stats()
    for run in runs:
        for key in speculative:
            speculative[key] += run["speculative"][key]
    rounds = speculative["verification_rounds"]
    summary = {
        "k": k,
        "concurrency": concurrency,
        "repeats": len(runs),
        "requests": sum(run["requests"] for run in runs),
        "output_tokens": sum(run["output_tokens"] for run in runs),
        "duration_s": [run["duration_s"] for run in runs],
        "request_per_s": median(run["request_per_s"] for run in runs),
        "output_token_per_s": median(run["output_token_per_s"] for run in runs),
        "goodput_request_per_s": (
            median(run["goodput_request_per_s"] for run in runs)
            if slo_enabled
            else None
        ),
        "latency_ms": {
            "ttft": distribution([value for run in runs for value in run["_ttft_ms"]]),
            "tpot": distribution([value for run in runs for value in run["_tpot_ms"]]),
            "e2e": distribution([value for run in runs for value in run["_e2e_ms"]]),
        },
        "steps": sum(run["steps"] for run in runs),
        "max_active_requests": max(run["max_active_requests"] for run in runs),
        "preemptions": sum(run["preemptions"] for run in runs),
        "output_hashes": [run["output_hash"] for run in runs],
        "_output_token_ids": runs[0]["_output_token_ids"],
        "repeat_outputs_match": len({run["output_hash"] for run in runs}) == 1,
        "speculative": {
            **speculative,
            "acceptance_rate": (
                speculative["accepted"] / speculative["proposed"]
                if speculative["proposed"]
                else 0.0
            ),
            "average_accepted_length": speculative["accepted"] / rounds if rounds else 0.0,
            "position_acceptance_rate": {
                str(position): (
                    speculative[f"accepted_position_{position}"] / rounds
                    if rounds
                    else 0.0
                )
                for position in (1, 2, 3)
            },
        },
    }
    if any(run["step_trace"] for run in runs):
        summary["step_traces"] = [run["step_trace"] for run in runs]
    return summary


def run_k(args, k: int, prompt_texts: list[str]):
    use_mtp = k > 0
    init_started = perf_counter()
    llm = LLM(
        args.model,
        speculative_method="mtp" if use_mtp else "none",
        mtp_model=args.mtp_model if use_mtp else None,
        num_speculative_tokens=k if use_mtp else 1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        master_port=args.master_port_base + k,
    )
    init_seconds = perf_counter() - init_started
    try:
        capacity = min(llm.config.max_num_seqs, llm.model_runner.delta_state_capacity)
        all_prompts = make_prompt_tokens(
            llm.tokenizer, prompt_texts, args.input_len, args.num_requests
        )
        measurements = []
        skipped = []
        warmed_batch_sizes = set()
        for concurrency in args.concurrency:
            if concurrency > capacity:
                skipped.append({"concurrency": concurrency, "reason": f"capacity={capacity}"})
                continue
            warmup_sizes = (
                range(1, concurrency + 1)
                if args.warmup_all_batch_sizes
                else (concurrency,)
            )
            for batch_size in warmup_sizes:
                if batch_size in warmed_batch_sizes:
                    continue
                warmup_requests = max(
                    batch_size,
                    args.warmup_requests or 2 * batch_size,
                )
                warmup_prompts = make_prompt_tokens(
                    llm.tokenizer, prompt_texts, args.input_len, warmup_requests
                )
                run_closed_loop(
                    llm,
                    warmup_prompts,
                    batch_size,
                    min(args.warmup_output_len, args.output_len),
                    None,
                    None,
                    False,
                )
                warmed_batch_sizes.add(batch_size)
            runs = [
                run_closed_loop(
                    llm,
                    all_prompts,
                    concurrency,
                    args.output_len,
                    args.ttft_slo_ms,
                    args.tpot_slo_ms,
                    args.trace_steps,
                )
                for _ in range(args.repeats)
            ]
            summary = summarize_runs(
                k,
                concurrency,
                runs,
                args.ttft_slo_ms is not None or args.tpot_slo_ms is not None,
            )
            measurements.append(summary)
            print(
                f"k={k} concurrency={concurrency} "
                f"throughput={summary['output_token_per_s']:.3f} tok/s "
                f"TTFT-p95={summary['latency_ms']['ttft']['p95']:.1f} ms "
                f"TPOT-p95={summary['latency_ms']['tpot']['p95']:.1f} ms "
                f"acceptance={summary['speculative']['acceptance_rate']:.2%}",
                flush=True,
            )
        return {
            "k": k,
            "init_seconds": init_seconds,
            "requested_max_num_seqs": args.max_num_seqs,
            "active_request_capacity": capacity,
            "delta_state_capacity": llm.model_runner.delta_state_capacity,
            "delta_state_bytes_per_request": (
                llm.model_runner.model.delta_state_bytes(llm.config.hf_config.dtype)
                * (2 if use_mtp else 1)
            ),
            "kv_cache_blocks": llm.config.num_kvcache_blocks,
            "kv_cache_token_capacity": llm.config.num_kvcache_blocks * llm.config.kvcache_block_size,
            "measurements": measurements,
            "skipped": skipped,
        }
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    prompt_texts = load_prompt_texts(args.prompt_jsonl)
    cases = [run_k(args, k, prompt_texts) for k in args.k]
    by_concurrency = {}
    slo_enabled = args.ttft_slo_ms is not None or args.tpot_slo_ms is not None
    metric = "goodput_request_per_s" if slo_enabled else "output_token_per_s"
    for concurrency in args.concurrency:
        candidates = [
            measurement
            for case in cases
            for measurement in case["measurements"]
            if measurement["concurrency"] == concurrency
        ]
        if not candidates:
            continue
        baseline = next((item for item in candidates if item["k"] == 0), candidates[0])
        comparisons = {}
        for item in candidates:
            first_divergence = None
            matching_tokens = 0
            total_tokens = 0
            exact_requests = 0
            for request_index, (expected, actual) in enumerate(
                zip(baseline["_output_token_ids"], item["_output_token_ids"])
            ):
                request_matches = expected == actual
                exact_requests += int(request_matches)
                matching_tokens += sum(
                    expected_token == actual_token
                    for expected_token, actual_token in zip(expected, actual)
                )
                total_tokens += len(expected)
                for token_index, (expected_token, actual_token) in enumerate(
                    zip(expected, actual)
                ):
                    if expected_token != actual_token:
                        first_divergence = {
                            "request_index": request_index,
                            "token_index": token_index,
                            "expected_token": expected_token,
                            "actual_token": actual_token,
                            "expected_context": expected[
                                max(0, token_index - 3) : token_index + 4
                            ],
                            "actual_context": actual[
                                max(0, token_index - 3) : token_index + 4
                            ],
                        }
                        break
                if first_divergence is not None:
                    break
            comparisons[str(item["k"])] = {
                "exact_match": first_divergence is None,
                "exact_request_rate": exact_requests / len(baseline["_output_token_ids"]),
                "positionwise_token_agreement": (
                    matching_tokens / total_tokens if total_tokens else 0.0
                ),
                "first_divergence": first_divergence,
            }
        exact_candidates = [
            item
            for item in candidates
            if (
                not args.require_exact_output
                or comparisons[str(item["k"])]["exact_match"]
            )
        ]
        winner = max(exact_candidates, key=lambda item: item[metric])
        by_concurrency[str(concurrency)] = {
            "winner_k": winner["k"],
            "metric": metric,
            "metric_value": winner[metric],
            "mismatched_k_disqualified": [
                int(k) for k, comparison in comparisons.items()
                if args.require_exact_output and not comparison["exact_match"]
            ],
            "exact_output_match_across_k": all(
                comparison["exact_match"] for comparison in comparisons.values()
            ),
            "output_comparison_by_k": comparisons,
        }
    for case in cases:
        for measurement in case["measurements"]:
            measurement.pop("_output_token_ids", None)
    result = {
        "model": str(Path(args.model).resolve()),
        "mtp_model": str(Path(args.mtp_model).resolve()),
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "closed_loop": True,
            "greedy": True,
            "ignore_eos": True,
            "prompt_source": args.prompt_jsonl or "built-in natural prompts",
        },
        "methodology": {
            "warmup_requests": args.warmup_requests or "2 * concurrency",
            "warmup_output_len": args.warmup_output_len,
            "timed_repeats": args.repeats,
            "winner_metric": metric,
            "tokenization_in_timing": False,
            "engine_reused_across_concurrency": True,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "require_exact_output": args.require_exact_output,
        },
        "cases": cases,
        "winner_by_concurrency": by_concurrency,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.print_json:
        print(rendered)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
