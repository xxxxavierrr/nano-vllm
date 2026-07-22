from __future__ import annotations

import argparse
import json

import torch
import triton

from nanovllm.layers.deltanet_chunk import (
    DELTA_CHUNK_MIN_TOKENS,
    DELTA_CHUNK_SIZE,
    gated_delta_packed,
)


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


def _make_mixed_metadata(
    prefill_tokens: int,
    decode_sequences: int,
) -> tuple[torch.Tensor, ...]:
    lengths = (1,) * decode_sequences + (prefill_tokens,)
    cu_seqlens = [0]
    chunk_indices = []
    cu_chunks = [0]
    chunk_sequences = []
    recurrent_sequences = []
    for sequence, length in enumerate(lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        if length >= DELTA_CHUNK_MIN_TOKENS:
            chunk_sequences.append(sequence)
            chunks = (length + DELTA_CHUNK_SIZE - 1) // DELTA_CHUNK_SIZE
            chunk_indices.extend(
                (sequence, chunk) for chunk in range(chunks)
            )
            cu_chunks.append(cu_chunks[-1] + chunks)
        else:
            recurrent_sequences.append(sequence)

    def cuda_int32(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    return (
        cuda_int32(cu_seqlens),
        cuda_int32(chunk_indices).reshape(-1, 2),
        cuda_int32(cu_chunks),
        cuda_int32(chunk_sequences),
        cuda_int32(recurrent_sequences),
    )


def _make_all_recurrent_metadata(
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    return (
        empty.reshape(0, 2),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        empty,
        torch.arange(
            cu_seqlens.numel() - 1,
            device="cuda",
            dtype=torch.int32,
        ),
    )


def _make_all_chunk_metadata(tokens: int) -> tuple[torch.Tensor, ...]:
    chunks = (tokens + DELTA_CHUNK_SIZE - 1) // DELTA_CHUNK_SIZE
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    return (
        torch.tensor(
            tuple((0, chunk) for chunk in range(chunks)),
            device="cuda",
            dtype=torch.int32,
        ),
        torch.tensor((0, chunks), device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        empty,
    )


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
    parser.add_argument(
        "--mixed-decode-seqs",
        type=int,
        default=0,
        help="Also benchmark this many single-token decode sequences mixed with each prefill.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=150)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mixed_decode_seqs < 0:
        raise ValueError("mixed-decode-seqs must be non-negative")

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
        cu_seqlens = torch.tensor(
            (0, tokens), device="cuda", dtype=torch.int32
        )
        slots = torch.zeros(1, device="cuda", dtype=torch.int32)
        recurrent_metadata = _make_all_recurrent_metadata(cu_seqlens)
        chunk_metadata = _make_all_chunk_metadata(tokens)
        expected = gated_delta_packed(
            query,
            key,
            value,
            beta,
            decay,
            cu_seqlens,
            *recurrent_metadata,
            slots,
            recurrent_state.unsqueeze(0),
        )
        actual = gated_delta_packed(
            query,
            key,
            value,
            beta,
            decay,
            cu_seqlens,
            *chunk_metadata,
            slots,
            chunk_state.unsqueeze(0),
        )
        torch.cuda.synchronize()
        output_error = (
            actual.float() - expected.float()
        ).abs().max().item()
        state_error = (
            chunk_state.float() - recurrent_state.float()
        ).abs().max().item()

        recurrent_ms = triton.testing.do_bench(
            lambda: gated_delta_packed(
                query,
                key,
                value,
                beta,
                decay,
                cu_seqlens,
                *recurrent_metadata,
                slots,
                recurrent_state.unsqueeze(0),
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
        chunk_ms = triton.testing.do_bench(
            lambda: gated_delta_packed(
                query,
                key,
                value,
                beta,
                decay,
                cu_seqlens,
                *chunk_metadata,
                slots,
                chunk_state.unsqueeze(0),
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
        result = {
            "tokens": tokens,
            "recurrent_ms": recurrent_ms,
            "chunk_ms": chunk_ms,
            "chunk_speedup": recurrent_ms / chunk_ms,
            "max_output_error": output_error,
            "max_state_error": state_error,
        }

        if args.mixed_decode_seqs:
            total_tokens = tokens + args.mixed_decode_seqs
            mixed_inputs = _make_inputs(
                total_tokens,
                args.heads,
                args.key_dim,
                args.value_dim,
            )
            mixed_query, mixed_key, mixed_value, mixed_beta, mixed_decay, _ = (
                mixed_inputs
            )
            metadata = _make_mixed_metadata(
                tokens, args.mixed_decode_seqs
            )
            cu_seqlens, chunk_indices, cu_chunks = metadata[:3]
            chunk_sequences, recurrent_sequences = metadata[3:]
            num_sequences = args.mixed_decode_seqs + 1
            slots = torch.arange(
                num_sequences, device="cuda", dtype=torch.int32
            )
            state_slab = torch.zeros(
                num_sequences,
                args.heads,
                args.key_dim,
                args.value_dim,
                device="cuda",
            )
            recurrent_slab = state_slab.clone()
            hybrid_slab = state_slab.clone()
            recurrent_metadata = _make_all_recurrent_metadata(cu_seqlens)
            mixed_expected = gated_delta_packed(
                mixed_query,
                mixed_key,
                mixed_value,
                mixed_beta,
                mixed_decay,
                cu_seqlens,
                *recurrent_metadata,
                slots,
                recurrent_slab,
            )
            mixed_actual = gated_delta_packed(
                mixed_query,
                mixed_key,
                mixed_value,
                mixed_beta,
                mixed_decay,
                cu_seqlens,
                chunk_indices,
                cu_chunks,
                chunk_sequences,
                recurrent_sequences,
                slots,
                hybrid_slab,
            )
            torch.cuda.synchronize()
            mixed_output_error = (
                mixed_actual.float() - mixed_expected.float()
            ).abs().max().item()
            mixed_state_error = (
                hybrid_slab.float() - recurrent_slab.float()
            ).abs().max().item()
            recurrent_partition_ms = triton.testing.do_bench(
                lambda: gated_delta_packed(
                    mixed_query,
                    mixed_key,
                    mixed_value,
                    mixed_beta,
                    mixed_decay,
                    cu_seqlens,
                    *recurrent_metadata,
                    slots,
                    recurrent_slab,
                ),
                warmup=args.warmup,
                rep=args.rep,
            )
            mixed_partition_ms = triton.testing.do_bench(
                lambda: gated_delta_packed(
                    mixed_query,
                    mixed_key,
                    mixed_value,
                    mixed_beta,
                    mixed_decay,
                    cu_seqlens,
                    chunk_indices,
                    cu_chunks,
                    chunk_sequences,
                    recurrent_sequences,
                    slots,
                    hybrid_slab,
                ),
                warmup=args.warmup,
                rep=args.rep,
            )
            result["mixed"] = {
                "decode_sequences": args.mixed_decode_seqs,
                "total_tokens": total_tokens,
                "recurrent_partition_ms": recurrent_partition_ms,
                "mixed_partition_ms": mixed_partition_ms,
                "mixed_speedup": (
                    recurrent_partition_ms / mixed_partition_ms
                ),
                "max_output_error": mixed_output_error,
                "max_state_error": mixed_state_error,
            }
        results.append(result)

    print(json.dumps(
        {
            "device": torch.cuda.get_device_name(),
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "chunk_min_tokens": DELTA_CHUNK_MIN_TOKENS,
            "results": results,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
