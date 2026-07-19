import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Qwen3.6 greedy baseline with native MTP."
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4",
    )
    parser.add_argument(
        "--mtp-model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27B-mtp",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why the sky is blue in simple terms.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument(
        "--num-speculative-tokens", type=int, choices=[1, 2], default=2
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--baseline-master-port", type=int, default=2451)
    parser.add_argument("--mtp-master-port", type=int, default=2452)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.warmup_tokens < 1:
        parser.error("--max-tokens and --warmup-tokens must be positive")
    if not Path(args.model).is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not Path(args.mtp_model).is_dir():
        parser.error(f"MTP model directory does not exist: {args.mtp_model}")
    return args


def run_case(args, *, use_mtp: bool):
    init_started = perf_counter()
    llm = LLM(
        args.model,
        speculative_method="mtp" if use_mtp else "none",
        mtp_model=args.mtp_model if use_mtp else None,
        num_speculative_tokens=args.num_speculative_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        master_port=(
            args.mtp_master_port
            if use_mtp
            else args.baseline_master_port
        ),
    )
    init_seconds = perf_counter() - init_started
    token_ids = []
    speculative = {
        "drafted": 0,
        "proposed": 0,
        "accepted": 0,
        "rejected": 0,
        "bonus": 0,
        "verification_rounds": 0,
        "accepted_position_1": 0,
        "accepted_position_2": 0,
    }
    try:
        llm.generate(
            [args.prompt],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.warmup_tokens,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        llm.add_request(
            args.prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                ignore_eos=True,
            ),
        )
        torch.cuda.synchronize()
        started = perf_counter()
        first_token_seconds = None
        steps = 0
        while not llm.is_finished():
            outputs, stats = llm.step()
            torch.cuda.synchronize()
            steps += 1
            for output in outputs:
                token_ids.append(output.token_id)
                if first_token_seconds is None:
                    first_token_seconds = perf_counter() - started
            speculative["drafted"] += stats.speculative_drafted_tokens
            speculative["proposed"] += stats.speculative_proposed_tokens
            speculative["accepted"] += stats.speculative_accepted_tokens
            speculative["rejected"] += stats.speculative_rejected_tokens
            speculative["bonus"] += stats.speculative_bonus_tokens
            speculative["verification_rounds"] += (
                stats.speculative_verification_rounds
            )
            speculative["accepted_position_1"] += (
                stats.speculative_accepted_position_1
            )
            speculative["accepted_position_2"] += (
                stats.speculative_accepted_position_2
            )
        generate_seconds = perf_counter() - started
        return {
            "init_seconds": init_seconds,
            "generate_seconds": generate_seconds,
            "ttft_seconds": first_token_seconds,
            "steps": steps,
            "output_tokens": len(token_ids),
            "output_token_per_s": len(token_ids) / generate_seconds,
            "token_ids": token_ids,
            "text": llm.tokenizer.decode(token_ids),
            "speculative": {
                **speculative,
                "acceptance_rate": (
                    speculative["accepted"] / speculative["proposed"]
                    if speculative["proposed"]
                    else 0.0
                ),
                "average_accepted_length": (
                    speculative["accepted"]
                    / speculative["verification_rounds"]
                    if speculative["verification_rounds"]
                    else 0.0
                ),
                "position_acceptance_rate": {
                    "1": (
                        speculative["accepted_position_1"]
                        / speculative["verification_rounds"]
                        if speculative["verification_rounds"]
                        else 0.0
                    ),
                    "2": (
                        speculative["accepted_position_2"]
                        / speculative["verification_rounds"]
                        if speculative["verification_rounds"]
                        else 0.0
                    ),
                },
            },
            "kv_cache": {
                "bytes_per_block": llm.config.kvcache_block_bytes,
                "mtp_bytes_per_block": llm.config.mtp_kvcache_bytes,
                "blocks": llm.config.num_kvcache_blocks,
                "token_capacity": (
                    llm.config.num_kvcache_blocks
                    * llm.config.kvcache_block_size
                ),
            },
        }
    finally:
        llm.exit()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    baseline = run_case(args, use_mtp=False)
    gc.collect()
    torch.cuda.empty_cache()
    mtp = run_case(args, use_mtp=True)
    exact_match = baseline["token_ids"] == mtp["token_ids"]
    result = {
        "model": str(Path(args.model).resolve()),
        "mtp_model": str(Path(args.mtp_model).resolve()),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "num_speculative_tokens": args.num_speculative_tokens,
        "exact_token_match": exact_match,
        "baseline": baseline,
        "mtp": mtp,
        "speedup": (
            baseline["generate_seconds"] / mtp["generate_seconds"]
            if mtp["generate_seconds"]
            else 0.0
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if not exact_match:
        raise AssertionError("MTP greedy output differs from baseline")


if __name__ == "__main__":
    main()
