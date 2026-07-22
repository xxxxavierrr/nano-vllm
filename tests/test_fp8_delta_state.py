from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanovllm.config import Config
from nanovllm.engine.hybrid_state import HybridStateManager
from nanovllm.layers.fp8_delta_state import (
    DeltaStateShape,
    FP8DeltaStatePool,
    dequantize_delta_state_reference,
    experimental_quantize_state_rows,
    make_delta_state_layout,
    quantize_conv_state_reference,
    quantize_recurrent_state_reference,
)
from tools.estimate_delta_state import main as estimate_delta_state_main


SHAPE = DeltaStateShape(
    layers=2,
    conv_channels=6,
    conv_kernel_size=4,
    recurrent_heads=2,
    recurrent_key_dim=4,
    recurrent_value_dim=8,
)


def test_fp8_delta_state_reference_scale_granularity_and_error():
    torch.manual_seed(0)
    conv = torch.randn(2, 3, 6, 4, dtype=torch.bfloat16)
    recurrent = torch.randn(2, 3, 2, 4, 8, dtype=torch.float32)
    conv_payload, conv_scale = quantize_conv_state_reference(conv)
    recurrent_payload, recurrent_scale = quantize_recurrent_state_reference(
        recurrent
    )
    restored_conv = dequantize_delta_state_reference(
        conv_payload, conv_scale, dtype=torch.bfloat16
    )
    restored_recurrent = dequantize_delta_state_reference(
        recurrent_payload, recurrent_scale
    )

    assert conv_scale.shape == (2, 3, 6)
    assert recurrent_scale.shape == (2, 3, 2, 4)
    assert conv_scale.dtype is torch.float16
    assert recurrent_scale.dtype is torch.float16
    torch.testing.assert_close(restored_conv, conv, rtol=0.13, atol=0.03)
    torch.testing.assert_close(
        restored_recurrent, recurrent, rtol=0.13, atol=0.03
    )


def test_fp8_delta_state_zero_and_nonfinite_values_are_safe():
    values = torch.zeros(1, 2, 4)
    values[0, 0] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0])
    payload, scale = quantize_conv_state_reference(values)
    restored = dequantize_delta_state_reference(payload, scale)
    assert torch.isfinite(restored).all()
    assert bool((scale > 0).all())


def test_fp8_delta_state_capacity_counts_scales_and_branches():
    native = make_delta_state_layout(SHAPE, dtype="auto")
    fp8 = make_delta_state_layout(SHAPE, dtype="fp8_e4m3")
    report = fp8.report(request_capacity=8, branch_slots_per_request=4)

    assert fp8.conv_scale_bytes == SHAPE.layers * SHAPE.conv_channels * 2
    assert fp8.recurrent_scale_bytes == (
        SHAPE.layers * SHAPE.recurrent_heads * SHAPE.recurrent_key_dim * 2
    )
    assert fp8.bytes_per_slot < native.bytes_per_slot
    assert report["total_slots"] == 40
    assert report["total_bytes"] == 40 * fp8.bytes_per_slot


class FakeHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @staticmethod
    def create_delta_state(device, dtype):
        return torch.zeros(1, device=device), torch.zeros(1, device=device)

    @staticmethod
    def create_delta_state_slab(capacity, device, dtype):
        return (
            torch.zeros(1, capacity, 1, device=device, dtype=dtype),
            torch.zeros(1, capacity, 1, device=device),
        )


def test_fp8_pool_uses_same_committed_and_branch_slot_lifecycle():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(1, device="cpu", branch_slots_per_sequence=3)
    manager.get(9)
    manager.reserve_branches({9: 3})
    pool = FP8DeltaStatePool(SHAPE, manager.total_slot_capacity)
    candidates = manager.branches[9]
    for prefix, slot in enumerate(candidates, start=1):
        pool.store(
            slot,
            torch.full(
                (SHAPE.layers, SHAPE.conv_channels, SHAPE.conv_kernel_size),
                prefix,
                dtype=torch.float32,
            ),
            torch.full(
                (
                    SHAPE.layers,
                    SHAPE.recurrent_heads,
                    SHAPE.recurrent_key_dim,
                    SHAPE.recurrent_value_dim,
                ),
                prefix,
                dtype=torch.float32,
            ),
        )
    manager.commit_branches({9: 2})
    conv, recurrent = pool.load(manager.slots[9])
    torch.testing.assert_close(conv.float(), torch.full_like(conv.float(), 2))
    torch.testing.assert_close(
        recurrent, torch.full_like(recurrent, 2), rtol=1e-3, atol=1e-3
    )
    assert manager.rejected_prefix_target_replays == 0


def test_experimental_triton_state_path_is_fail_closed(monkeypatch):
    monkeypatch.delenv(
        "NANOVLLM_ENABLE_EXPERIMENTAL_FP8_DELTA_STATE_KERNELS", raising=False
    )
    with pytest.raises(RuntimeError, match="disabled"):
        experimental_quantize_state_rows(torch.randn(2, 8))


def test_config_rejects_fp8_delta_state_for_nonhybrid_model(
    tmp_path, monkeypatch
):
    outer = SimpleNamespace(
        model_type="qwen3",
        quantization_config=None,
        max_position_embeddings=1024,
    )
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained", lambda _: outer
    )
    with pytest.raises(ValueError, match="only defined"):
        Config(str(tmp_path), delta_state_dtype="fp8_e4m3")


def test_config_accepts_explicit_fp8_delta_state_for_hybrid_model(
    tmp_path, monkeypatch
):
    text = SimpleNamespace(
        quantization_config=None,
        max_position_embeddings=1024,
    )
    outer = SimpleNamespace(
        model_type="qwen3_5",
        quantization_config=None,
        text_config=text,
    )
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained", lambda _: outer
    )
    config = Config(str(tmp_path), delta_state_dtype="fp8_e4m3")
    assert config.delta_state_dtype == "fp8_e4m3"


def test_delta_state_capacity_cli_is_gpu_independent(capsys):
    result = estimate_delta_state_main([
        "--layers", "2",
        "--conv-channels", "6",
        "--conv-kernel-size", "4",
        "--recurrent-heads", "2",
        "--recurrent-key-dim", "4",
        "--recurrent-value-dim", "8",
        "--request-capacity", "8",
        "--branch-slots-per-request", "4",
    ])
    assert result["compression_ratio"] > 1
    assert result["runtime_enabled"] is False
    assert '"total_slots": 40' in capsys.readouterr().out
