from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nanovllm.engine.model_runner as model_runner_module
import nanovllm.models.qwen3_5 as qwen35_module
from nanovllm import LLM, SamplingParams


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(","))
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError(
            "prompt lengths must be positive integers"
        )
    return lengths


def _edit_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1]
                + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def _compare_tokens(
    baseline: list[int],
    candidate: list[int],
) -> dict[str, int | float | None | bool]:
    limit = max(len(baseline), len(candidate))
    matches = sum(
        left == right for left, right in zip(baseline, candidate)
    )
    prefix = 0
    for left, right in zip(baseline, candidate):
        if left != right:
            break
        prefix += 1
    exact = baseline == candidate
    distance = _edit_distance(baseline, candidate)
    return {
        "exact": exact,
        "baseline_tokens": len(baseline),
        "candidate_tokens": len(candidate),
        "matching_prefix_tokens": prefix,
        "first_divergence": None if exact else prefix,
        "position_agreement": 1.0 if limit == 0 else matches / limit,
        "edit_similarity": 1.0 if limit == 0 else 1.0 - distance / limit,
    }


def _make_prompts(llm: LLM, lengths: tuple[int, ...]) -> list[list[int]]:
    seed_tokens = llm.tokenizer.encode(
        "Explain this carefully and preserve every relevant detail. "
    )
    if not seed_tokens:
        raise RuntimeError("tokenizer produced an empty seed prompt")
    return [
        (seed_tokens * ((length + len(seed_tokens) - 1) // len(seed_tokens)))[:length]
        for length in lengths
    ]


def _set_chunk_threshold(threshold: int) -> None:
    qwen35_module.DELTA_CHUNK_MIN_TOKENS = threshold
    model_runner_module.DELTA_CHUNK_MIN_TOKENS = threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy Qwen3.6 outputs with recurrent and hybrid "
            "DeltaNet execution using one loaded checkpoint."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-lengths",
        type=_parse_lengths,
        default=_parse_lengths("512,1024"),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--candidate-threshold", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.candidate_threshold <= 0:
        raise ValueError("candidate-threshold must be positive")

    max_model_len = max(args.prompt_lengths) + args.max_tokens
    max_batched_tokens = sum(args.prompt_lengths)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_batched_tokens=max_batched_tokens,
        max_num_seqs=len(args.prompt_lengths),
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        prompts = _make_prompts(llm, args.prompt_lengths)
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

        _set_chunk_threshold(10**9)
        started = perf_counter()
        baseline = llm.generate(prompts, params, use_tqdm=False)
        baseline_seconds = perf_counter() - started

        _set_chunk_threshold(args.candidate_threshold)
        started = perf_counter()
        candidate = llm.generate(prompts, params, use_tqdm=False)
        candidate_seconds = perf_counter() - started
    finally:
        llm.exit()

    details = []
    for prompt_length, baseline_item, candidate_item in zip(
        args.prompt_lengths, baseline, candidate
    ):
        baseline_tokens = list(baseline_item["token_ids"])
        candidate_tokens = list(candidate_item["token_ids"])
        details.append({
            "prompt_tokens": prompt_length,
            **_compare_tokens(baseline_tokens, candidate_tokens),
            "text_exact": baseline_item["text"] == candidate_item["text"],
            "baseline_text": baseline_item["text"],
            "candidate_text": candidate_item["text"],
            "baseline_token_ids": baseline_tokens,
            "candidate_token_ids": candidate_tokens,
        })

    exact_requests = sum(item["exact"] for item in details)
    total_positions = sum(
        max(item["baseline_tokens"], item["candidate_tokens"])
        for item in details
    )
    matching_positions = sum(
        round(item["position_agreement"] * max(
            item["baseline_tokens"], item["candidate_tokens"]
        ))
        for item in details
    )
    report = {
        "model": args.model,
        "sampling": "greedy",
        "max_tokens": args.max_tokens,
        "candidate_threshold": args.candidate_threshold,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "request_exact_match_rate": exact_requests / len(details),
        "token_position_agreement": (
            1.0 if total_positions == 0
            else matching_positions / total_positions
        ),
        "mean_edit_similarity": (
            sum(item["edit_similarity"] for item in details) / len(details)
        ),
        "details": details,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(serialized + chr(10), encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
