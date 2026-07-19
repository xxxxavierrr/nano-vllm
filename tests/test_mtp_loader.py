from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.utils.mtp_loader import (
    expected_mtp_shapes,
    inspect_mtp_checkpoint,
    load_mtp_model,
)


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)


@pytest.fixture
def config():
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_bias=False,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.5,
        max_position_embeddings=64,
        rope_parameters={"rope_theta": 10_000_000},
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        layer_types=["linear_attention"],
    )


def checkpoint_tensors(config, *, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return {
        name: torch.randn(shape, dtype=dtype)
        for name, shape in expected_mtp_shapes(config).items()
    }


def test_mtp_checkpoint_validation_and_fused_loading(tmp_path, config):
    tensors = checkpoint_tensors(config)
    save_file(tensors, str(tmp_path / "mtp.safetensors"))
    assert inspect_mtp_checkpoint(str(tmp_path), config) == expected_mtp_shapes(config)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3_5MTP(config)
        load_mtp_model(model, str(tmp_path), config)
    finally:
        torch.set_default_dtype(old_dtype)

    qkv = model.layers[0].self_attn.qkv_proj.weight.detach()
    assert torch.equal(qkv[:32], tensors["mtp.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv[32:40], tensors["mtp.layers.0.self_attn.k_proj.weight"])
    assert torch.equal(qkv[40:], tensors["mtp.layers.0.self_attn.v_proj.weight"])
    gate_up = model.layers[0].mlp.gate_up_proj.weight.detach()
    assert torch.equal(gate_up[:32], tensors["mtp.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(gate_up[32:], tensors["mtp.layers.0.mlp.up_proj.weight"])


def test_mtp_checkpoint_rejects_missing_tensor(tmp_path, config):
    tensors = checkpoint_tensors(config)
    del tensors["mtp.norm.weight"]
    save_file(tensors, str(tmp_path / "mtp.safetensors"))
    with pytest.raises(ValueError, match="missing: mtp.norm.weight"):
        inspect_mtp_checkpoint(str(tmp_path), config)


def test_mtp_checkpoint_rejects_non_bf16(tmp_path, config):
    save_file(
        checkpoint_tensors(config, dtype=torch.float16),
        str(tmp_path / "mtp.safetensors"),
    )
    with pytest.raises(TypeError, match="must use BF16"):
        inspect_mtp_checkpoint(str(tmp_path), config)


def test_mtp_checkpoint_rejects_wrong_shape(tmp_path, config):
    tensors = checkpoint_tensors(config)
    tensors["mtp.fc.weight"] = torch.randn(15, 32, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "mtp.safetensors"))
    with pytest.raises(ValueError, match="mtp.fc.weight shape"):
        inspect_mtp_checkpoint(str(tmp_path), config)
