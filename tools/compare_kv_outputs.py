import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that returns the Fibonacci sequence.",
    "用三句话解释什么是 KV Cache。",
    "List five practical ways to reduce inference latency.",
]


def _run(args, kv_cache_dtype):
    started = perf_counter()
    llm = LLM(
        args.model,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=len(args.prompts),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    init_seconds = perf_counter() - started
    try:
        warmup_started = perf_counter()
        llm.generate(
            [args.prompts[0]],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        warmup_seconds = perf_counter() - warmup_started
        params = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
        )
        started = perf_counter()
        outputs = llm.generate(args.prompts, params, use_tqdm=False)
        generation_seconds = perf_counter() - started
        config = llm.model_runner.config
        capacity = config.num_kvcache_blocks * config.kvcache_block_size
        return {
            "kv_cache_dtype": kv_cache_dtype,
            "init_seconds": init_seconds,
            "warmup_seconds": warmup_seconds,
            "generation_seconds": generation_seconds,
            "kv_cache_bytes_per_block": config.kvcache_block_bytes,
            "kv_cache_blocks": config.num_kvcache_blocks,
            "kv_cache_token_capacity": capacity,
            "outputs": [
                {
                    "text": output["text"],
                    "token_ids": list(output["token_ids"]),
                }
                for output in outputs
            ],
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _compare(baseline, candidate):
    details = []
    matching_positions = 0
    total_positions = 0
    for prompt, base, cand in zip(
        baseline["prompts"], baseline["outputs"], candidate["outputs"]
    ):
        base_tokens = base["token_ids"]
        cand_tokens = cand["token_ids"]
        shared = min(len(base_tokens), len(cand_tokens))
        first_divergence = next(
            (
                index
                for index in range(shared)
                if base_tokens[index] != cand_tokens[index]
            ),
            None,
        )
        if first_divergence is None and len(base_tokens) != len(cand_tokens):
            first_divergence = shared
        matches = sum(
            left == right for left, right in zip(base_tokens, cand_tokens)
        )
        positions = max(len(base_tokens), len(cand_tokens))
        matching_positions += matches
        total_positions += positions
        details.append(
            {
                "prompt": prompt,
                "baseline_tokens": len(base_tokens),
                "candidate_tokens": len(cand_tokens),
                "token_matches": matches,
                "token_positions": positions,
                "token_agreement": matches / positions if positions else 1.0,
                "exact": base_tokens == cand_tokens,
                "text_exact": base["text"] == cand["text"],
                "first_divergence": first_divergence,
                "baseline_text": base["text"],
                "candidate_text": cand["text"],
            }
        )
    return {
        "token_agreement": (
            matching_positions / total_positions if total_positions else 1.0
        ),
        "exact_requests": sum(item["exact"] for item in details),
        "total_requests": len(details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare deterministic BF16 and FP8 KV-cache outputs"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.prompts = args.prompts or DEFAULT_PROMPTS
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    baseline = _run(args, "auto")
    baseline["prompts"] = args.prompts
    candidate = _run(args, "fp8_e4m3")
    result = {
        "model": str(Path(args.model).resolve()),
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _compare(baseline, candidate),
    }
    result["capacity_ratio"] = (
        candidate["kv_cache_token_capacity"]
        / baseline["kv_cache_token_capacity"]
    )
    result["theoretical_block_compression_ratio"] = (
        baseline["kv_cache_bytes_per_block"]
        / candidate["kv_cache_bytes_per_block"]
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()