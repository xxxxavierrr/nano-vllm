from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    dtype: str
    scale_mode: str
    block_size: int
    target_layers: int
    mtp_layers: int
    num_kv_heads: int
    head_dim: int
    target_payload_bytes_per_block: int
    mtp_payload_bytes_per_block: int
    scale_bytes_per_block: int
    total_bytes_per_block: int

    @property
    def scale_overhead_ratio(self) -> float:
        return (
            self.scale_bytes_per_block / self.total_bytes_per_block
            if self.total_bytes_per_block
            else 0.0
        )

    def report(self, *, num_blocks: int | None = None) -> dict:
        result = {
            **asdict(self),
            "scale_overhead_ratio": self.scale_overhead_ratio,
        }
        if num_blocks is not None:
            if num_blocks < 0:
                raise ValueError("num_blocks cannot be negative")
            result.update(
                num_blocks=num_blocks,
                token_capacity=num_blocks * self.block_size,
                allocated_bytes=num_blocks * self.total_bytes_per_block,
            )
        return result


def make_kv_cache_layout(
    *,
    kv_cache_dtype: str,
    target_layers: int,
    mtp_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    activation_bytes: int = 2,
    native_dtype: str = "bf16",
) -> KVCacheLayout:
    values = {
        "target_layers": target_layers,
        "mtp_layers": mtp_layers,
        "block_size": block_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "activation_bytes": activation_bytes,
    }
    if any(value < 0 for value in (target_layers, mtp_layers)):
        raise ValueError("KV layer counts cannot be negative")
    if any(values[name] <= 0 for name in (
        "block_size", "num_kv_heads", "head_dim", "activation_bytes"
    )):
        raise ValueError("KV dimensions and activation bytes must be positive")
    if kv_cache_dtype not in {"auto", "fp8_e4m3"}:
        raise ValueError("unsupported KV cache dtype")
    fp8 = kv_cache_dtype == "fp8_e4m3"
    target_element_bytes = 1 if fp8 else activation_bytes
    target_payload = (
        2 * target_layers * block_size * num_kv_heads * head_dim
        * target_element_bytes
    )
    mtp_payload = (
        2 * mtp_layers * block_size * num_kv_heads * head_dim * activation_bytes
    )
    scale_bytes = (
        2 * target_layers * block_size * num_kv_heads * 2 if fp8 else 0
    )
    return KVCacheLayout(
        dtype="fp8_e4m3" if fp8 else native_dtype,
        scale_mode="per_token_per_kv_head" if fp8 else "none",
        block_size=block_size,
        target_layers=target_layers,
        mtp_layers=mtp_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        target_payload_bytes_per_block=target_payload,
        mtp_payload_bytes_per_block=mtp_payload,
        scale_bytes_per_block=scale_bytes,
        total_bytes_per_block=target_payload + mtp_payload + scale_bytes,
    )


def compare_kv_cache_layouts(
    baseline: KVCacheLayout,
    candidate: KVCacheLayout,
    *,
    available_bytes: int,
) -> dict:
    if available_bytes <= 0:
        raise ValueError("available_bytes must be positive")
    baseline_blocks = available_bytes // baseline.total_bytes_per_block
    candidate_blocks = available_bytes // candidate.total_bytes_per_block
    return {
        "available_bytes": available_bytes,
        "baseline": baseline.report(num_blocks=baseline_blocks),
        "candidate": candidate.report(num_blocks=candidate_blocks),
        "block_compression_ratio": (
            baseline.total_bytes_per_block / candidate.total_bytes_per_block
        ),
        "token_capacity_ratio": (
            candidate_blocks / baseline_blocks if baseline_blocks else None
        ),
    }
