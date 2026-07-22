import pytest

from nanovllm.engine.kv_capacity import (
    compare_kv_cache_layouts,
    make_kv_cache_layout,
)


def test_fp8_kv_capacity_report_includes_scale_overhead():
    bf16 = make_kv_cache_layout(
        kv_cache_dtype="auto",
        target_layers=16,
        mtp_layers=0,
        block_size=256,
        num_kv_heads=2,
        head_dim=256,
    )
    fp8 = make_kv_cache_layout(
        kv_cache_dtype="fp8_e4m3",
        target_layers=16,
        mtp_layers=0,
        block_size=256,
        num_kv_heads=2,
        head_dim=256,
    )
    comparison = compare_kv_cache_layouts(
        bf16, fp8, available_bytes=4 * 1024**3
    )

    assert fp8.scale_bytes_per_block > 0
    assert fp8.scale_overhead_ratio == pytest.approx(2 / 258)
    assert comparison["block_compression_ratio"] == pytest.approx(512 / 258)
    assert comparison["token_capacity_ratio"] == pytest.approx(
        comparison["candidate"]["num_blocks"]
        / comparison["baseline"]["num_blocks"]
    )


def test_native_mtp_cache_reduces_combined_fp8_capacity_gain():
    bf16 = make_kv_cache_layout(
        kv_cache_dtype="auto",
        target_layers=16,
        mtp_layers=1,
        block_size=256,
        num_kv_heads=2,
        head_dim=256,
    )
    fp8 = make_kv_cache_layout(
        kv_cache_dtype="fp8_e4m3",
        target_layers=16,
        mtp_layers=1,
        block_size=256,
        num_kv_heads=2,
        head_dim=256,
    )
    assert 1 < bf16.total_bytes_per_block / fp8.total_bytes_per_block < 512 / 258


def test_kv_capacity_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        make_kv_cache_layout(
            kv_cache_dtype="auto",
            target_layers=1,
            mtp_layers=0,
            block_size=0,
            num_kv_heads=2,
            head_dim=128,
        )
