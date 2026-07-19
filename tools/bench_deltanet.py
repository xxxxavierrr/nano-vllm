from __future__ import annotations

import argparse
import json

import torch
import triton

from nanovllm.layers.deltanet import gated_delta_recurrent
from nanovllm.layers.deltanet_chunk import gated_delta_chunk


def _parse_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item) for item in value.split(","))
    if not tokens or any(item <= 0 for item in tokens):
        raise argparse.ArgumentTypeError("tokens must be positive integers")
    return tokens


def _make_inputs(
    tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(tokens)
    query = torch.randn(tokens, heads, key_dim, device="cuda") * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(tokens, heads, value_dim, device="cuda")
    beta = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    decay = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    state = torch.zeros(heads, key_dim, value_dim, device="cuda")
    return query, key, value, beta, decay, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare nano-vLLM DeltaNet recurrent and chunk kernels."
    )
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=_parse_tokens("1,8,32,64,128,512,1024,2048,4096"),
    )
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=150)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    results = []
    for tokens in args.tokens:
        query, key, value, beta, decay, state = _make_inputs(
            tokens,
            args.heads,
            args.key_dim,
            args.value_dim,
        )
        recurrent_state = state.clone()
        chunk_state = state.clone()
        expected = gated_delta_recurrent(
            query, key, value, beta, decay, recurrent_state
        )
        actual = gated_delta_chunk(
            query, key, value, beta, decay, chunk_state
        )
        torch.cuda.synchronize()

        recurrent_ms = triton.testing.do_bench(
            lambda: gated_delta_recurrent(
                query, key, value, beta, decay, recurrent_state
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
        chunk_ms = triton.testing.do_bench(
            lambda: gated_delta_chunk(
                query, key, value, beta, decay, chunk_state
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
        results.append(
            {
                "tokens": tokens,
                "recurrent_ms": recurrent_ms,
                "chunk_ms": chunk_ms,
                "chunk_speedup": recurrent_ms / chunk_ms,
                "max_output_error": (
                    actual.float() - expected.float()
                ).abs().max().item(),
                "max_state_error": (
                    chunk_state.float() - recurrent_state.float()
                ).abs().max().item(),
            }
        )

    print(json.dumps(
        {
            "device": torch.cuda.get_device_name(),
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "results": results,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
