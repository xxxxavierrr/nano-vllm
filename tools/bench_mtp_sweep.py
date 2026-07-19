import argparse
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_PROMPTS = [
    "Explain why the sky is blue in simple terms.",
    "Write one sentence about a quiet winter morning.",
    "What is the difference between RAM and storage?",
    "Give two practical tips for learning Python.",
    "Why do leaves change color in autumn?",
    "Describe how rain forms for a child.",
    "Name a benefit and a drawback of solar power.",
    "Summarize what a database index does.",
]


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3.6 MTP draft length and burst concurrency."
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4",
    )
    parser.add_argument(
        "--mtp-model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27B-mtp",
    )
    parser.add_argument("--k", default="1,2,3", help="comma-separated draft lengths")
    parser.add_argument(
        "--concurrency", default="1,2,4,8", help="comma-separated burst sizes"
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--master-port-base", type=int, default=2460)
    parser.add_argument("--output-json", default="/tmp/nanovllm-mtp-sweep.json")
    args = parser.parse_args()
    args.k = parse_csv_ints(args.k, name="--k")
    args.concurrency = parse_csv_ints(args.concurrency, name="--concurrency")
    if any(k not in (1, 2, 3) for k in args.k):
        parser.error("--k currently supports only 1, 2, and 3")
    if args.max_tokens <= 0 or args.warmup_tokens <= 0 or args.repeats <= 0:
        parser.error("token counts and repeats must be positive")
    if args.max_model_len <= args.max_tokens:
        parser.error("--max-model-len must leave room for the prompt")
    for path in (args.model, args.mtp_model):
        if not Path(path).is_dir():
            parser.error(f"model directory does not exist: {path}")
    return args


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


def prompts_for(concurrency: int) -> list[str]:
    return [DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)] for index in range(concurrency)]


def warmup(llm: LLM, concurrency: int, warmup_tokens: int):
    llm.generate(
        prompts_for(concurrency),
        SamplingParams(temperature=0.0, max_tokens=warmup_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    torch.cuda.synchronize()


def run_wave(llm: LLM, concurrency: int, max_tokens: int):
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for prompt in prompts_for(concurrency):
        llm.add_request(prompt, params)
    torch.cuda.synchronize()
    started = perf_counter()
    output_tokens = 0
    steps = 0
    speculative = empty_speculative_stats()
    while not llm.is_finished():
        outputs, stats = llm.step()
        output_tokens += len(outputs)
        steps += 1
        accumulate_stats(speculative, stats)
    torch.cuda.synchronize()
    duration = perf_counter() - started
    expected_tokens = concurrency * max_tokens
    if output_tokens != expected_tokens:
        raise RuntimeError(
            f"expected {expected_tokens} output tokens, observed {output_tokens}"
        )
    return {
        "duration_s": duration,
        "output_tokens": output_tokens,
        "steps": steps,
        "speculative": speculative,
    }


def summarize_waves(k: int, concurrency: int, waves: list[dict]):
    duration = sum(wave["duration_s"] for wave in waves)
    output_tokens = sum(wave["output_tokens"] for wave in waves)
    steps = sum(wave["steps"] for wave in waves)
    wave_durations = [wave["duration_s"] for wave in waves]
    wave_throughputs = [
        wave["output_tokens"] / wave["duration_s"] for wave in waves
    ]
    speculative = empty_speculative_stats()
    for wave in waves:
        for key in speculative:
            speculative[key] += wave["speculative"][key]
    rounds = speculative["verification_rounds"]
    position_rates = {
        str(position): (
            speculative[f"accepted_position_{position}"] / rounds if rounds else 0.0
        )
        for position in (1, 2, 3)
    }
    return {
        "k": k,
        "concurrency": concurrency,
        "repeats": len(waves),
        "duration_s_total": duration,
        "duration_s_median": median(wave_durations),
        "wave_durations_s": wave_durations,
        "steps_total": steps,
        "output_tokens": output_tokens,
        "request_per_s": concurrency * len(waves) / duration,
        "output_token_per_s": median(wave_throughputs),
        "output_token_per_s_aggregate": output_tokens / duration,
        "wave_output_token_per_s": wave_throughputs,
        "speculative": {
            **speculative,
            "acceptance_rate": (
                speculative["accepted"] / speculative["proposed"]
                if speculative["proposed"]
                else 0.0
            ),
            "average_accepted_length": speculative["accepted"] / rounds if rounds else 0.0,
            "position_acceptance_rate": position_rates,
        },
    }


def run_k(args, k: int):
    init_started = perf_counter()
    llm = LLM(
        args.model,
        speculative_method="mtp",
        mtp_model=args.mtp_model,
        num_speculative_tokens=k,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        master_port=args.master_port_base + k,
    )
    init_seconds = perf_counter() - init_started
    try:
        delta_capacity = llm.model_runner.delta_state_capacity
        capacity = min(llm.config.max_num_seqs, delta_capacity)
        if capacity < 1:
            raise RuntimeError("engine reported zero active-request capacity")
        concurrencies = sorted(
            {min(value, capacity) for value in args.concurrency} | {capacity}
        )
        measurements = []
        for concurrency in concurrencies:
            warmup(llm, concurrency, args.warmup_tokens)
            waves = [run_wave(llm, concurrency, args.max_tokens) for _ in range(args.repeats)]
            summary = summarize_waves(k, concurrency, waves)
            measurements.append(summary)
            print(
                f"k={k} concurrency={concurrency} "
                f"throughput={summary['output_token_per_s']:.3f} tok/s "
                f"acceptance={summary['speculative']['acceptance_rate']:.2%}",
                flush=True,
            )
        return {
            "k": k,
            "init_seconds": init_seconds,
            "requested_max_num_seqs": args.max_num_seqs,
            "engine_max_num_seqs": llm.config.max_num_seqs,
            "delta_state_capacity": delta_capacity,
            "active_request_capacity": capacity,
            "kv_cache_blocks": llm.config.num_kvcache_blocks,
            "kv_cache_token_capacity": (
                llm.config.num_kvcache_blocks * llm.config.kvcache_block_size
            ),
            "measurements": measurements,
        }
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    cases = [run_k(args, k) for k in args.k]
    measurements = [item for case in cases for item in case["measurements"]]
    winner = max(measurements, key=lambda item: item["output_token_per_s"])
    result = {
        "model": str(Path(args.model).resolve()),
        "mtp_model": str(Path(args.mtp_model).resolve()),
        "methodology": {
            "greedy": True,
            "ignore_eos": True,
            "max_tokens_per_request": args.max_tokens,
            "warmup_tokens_per_request": args.warmup_tokens,
            "timed_repeats": args.repeats,
            "winner_metric": "median steady-state output_token_per_s",
            "burst_arrival": True,
            "tokenization_in_timing": False,
            "engine_reused_across_concurrency": True,
        },
        "cases": cases,
        "winner": {
            "k": winner["k"],
            "concurrency": winner["concurrency"],
            "output_token_per_s": winner["output_token_per_s"],
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
