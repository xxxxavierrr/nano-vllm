import pytest
import torch
import torch.distributed as dist
from torch import nn
from safetensors.torch import save_file

from nanovllm.layers.gptq import GPTQConfig, unpack_gptq_qweight
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from nanovllm.utils.loader import load_model


class _PackedQKVModel(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=256,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=1,
            quant_config=GPTQConfig(desc_act=True),
        )


def _checkpoint_tensors(mismatched_g_idx=False, omit=None):
    torch.manual_seed(0)
    tensors = {}
    base_g_idx = torch.arange(256, dtype=torch.int32) // 128
    for name, width in (("q_proj", 16), ("k_proj", 8), ("v_proj", 8)):
        tensors[f"{name}.qweight"] = torch.randint(
            -(2**31), 2**31 - 1, (32, width), dtype=torch.int32
        )
        tensors[f"{name}.scales"] = torch.randn(
            2, width, dtype=torch.bfloat16
        ).abs_()
        tensors[f"{name}.qzeros"] = torch.randint(
            -(2**31), 2**31 - 1, (2, width // 8), dtype=torch.int32
        )
        tensors[f"{name}.g_idx"] = base_g_idx.clone()
    if mismatched_g_idx:
        tensors["v_proj.g_idx"][0] = 1
    if omit is not None:
        del tensors[omit]
    return tensors


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)


def _load(tmp_path, tensors):
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, str(checkpoint))
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = _PackedQKVModel()
        load_model(model, str(tmp_path))
    finally:
        torch.set_default_dtype(old_dtype)
    return model


def test_gptq_loader_fuses_equal_g_idx_without_bf16_weight(tmp_path):
    tensors = _checkpoint_tensors()
    model = _load(tmp_path, tensors)
    linear = model.qkv_proj

    assert linear.weight is None
    assert linear.qweight.shape == (32, 32)
    assert linear.scales.shape == (2, 32)
    assert linear.qzeros.shape == (2, 4)
    assert torch.equal(linear.g_idx.cpu(), tensors["q_proj.g_idx"])
    assert torch.equal(linear.qweight[:, :16].cpu(), tensors["q_proj.qweight"])
    assert torch.equal(linear.qweight[:, 16:24].cpu(), tensors["k_proj.qweight"])
    assert torch.equal(linear.qweight[:, 24:].cpu(), tensors["v_proj.qweight"])


def test_gptq_loader_detects_packed_symmetric_zero(tmp_path):
    tensors = _checkpoint_tensors()
    for name in ("q_proj", "k_proj", "v_proj"):
        tensors[f"{name}.qzeros"].fill_(0x77777777)
    model = _load(tmp_path, tensors)
    assert model.qkv_proj._gptq_symmetric_zero


def test_gptq_loader_repacks_nonmonotonic_desc_act_once(tmp_path):
    tensors = _checkpoint_tensors()
    permutation = torch.randperm(256)
    shuffled_g_idx = (
        torch.arange(256, dtype=torch.int32) // 128
    ).index_select(0, permutation)
    for name in ("q_proj", "k_proj", "v_proj"):
        tensors[f"{name}.g_idx"] = shuffled_g_idx.clone()
    checkpoint_qweight = torch.cat(
        [
            tensors["q_proj.qweight"],
            tensors["k_proj.qweight"],
            tensors["v_proj.qweight"],
        ],
        dim=1,
    )
    runtime_perm = torch.argsort(shuffled_g_idx, stable=True).to(torch.int32)

    model = _load(tmp_path, tensors)
    linear = model.qkv_proj

    assert torch.equal(linear.gptq_input_perm.cpu(), runtime_perm)
    assert torch.equal(
        unpack_gptq_qweight(linear.qweight.cpu()),
        unpack_gptq_qweight(checkpoint_qweight).index_select(
            0, runtime_perm.long()
        ),
    )


def test_gptq_loader_rejects_fused_g_idx_mismatch(tmp_path):
    with pytest.raises(ValueError, match="fused GPTQ g_idx mismatch"):
        _load(tmp_path, _checkpoint_tensors(mismatched_g_idx=True))


def test_gptq_loader_rejects_missing_packed_tensor(tmp_path):
    with pytest.raises(ValueError, match=r"qzeros\['v'\]"):
        _load(tmp_path, _checkpoint_tensors(omit="v_proj.qzeros"))

class _PackedGateUpModel(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            256,
            [16, 16],
            quant_config=GPTQConfig(desc_act=True),
        )


def _gate_up_tensors(mismatched_g_idx=False):
    tensors = {}
    g_idx = torch.arange(256, dtype=torch.int32) // 128
    for name in ("gate_proj", "up_proj"):
        tensors[f"{name}.qweight"] = torch.zeros(32, 16, dtype=torch.int32)
        tensors[f"{name}.scales"] = torch.ones(2, 16, dtype=torch.bfloat16)
        tensors[f"{name}.qzeros"] = torch.zeros(2, 2, dtype=torch.int32)
        tensors[f"{name}.g_idx"] = g_idx.clone()
    if mismatched_g_idx:
        tensors["up_proj.g_idx"][3] = 1
    return tensors


def _load_gate_up(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "model.safetensors"))
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = _PackedGateUpModel()
        load_model(model, str(tmp_path))
    finally:
        torch.set_default_dtype(old_dtype)
    return model


def test_gptq_loader_fuses_gate_up_with_equal_g_idx(tmp_path):
    model = _load_gate_up(tmp_path, _gate_up_tensors())
    assert model.gate_up_proj.qweight.shape == (32, 32)


def test_gptq_loader_rejects_gate_up_g_idx_mismatch(tmp_path):
    with pytest.raises(ValueError, match="fused GPTQ g_idx mismatch"):
        _load_gate_up(tmp_path, _gate_up_tensors(mismatched_g_idx=True))
