import json

import pytest
import torch
import torch.distributed as dist
from torch import nn
from safetensors.torch import save_file

from nanovllm.calibration.cache import (
    CalibrationBatch,
    CalibrationCacheReader,
    CalibrationCacheWriter,
)
from nanovllm.calibration.checkpoint import (
    load_gptq_checkpoint_tensors,
    projected_dspark_bytes,
    quantize_dspark_model,
    save_gptq_checkpoint,
)
from nanovllm.calibration.dspark import (
    DSparkCalibrationModel,
    DSparkConfig,
    map_dspark_state_dict,
)
from nanovllm.calibration.gptq_quantizer import (
    GPTQQuantizerConfig,
    HessianAccumulator,
    quantize_linear_gptq,
)
from nanovllm.layers.gptq import (
    GPTQConfig,
    dequantize_gptq_weight,
    gptq_linear_reference,
)
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.utils.loader import load_model
from tools.quantize_dspark import main as quantize_dspark_main


def make_batch(tokens=8, hidden=128):
    return CalibrationBatch(
        token_ids=torch.arange(tokens, dtype=torch.int64) % 16,
        positions=torch.arange(tokens, dtype=torch.int64),
        cu_seqlens=torch.tensor([0, tokens // 2, tokens], dtype=torch.int32),
        target_hidden_states=torch.randn(tokens, hidden, dtype=torch.float32),
    )


def test_calibration_cache_is_sharded_resumable_and_verified(tmp_path):
    provenance = {"target": "synthetic", "revision": "test"}
    writer = CalibrationCacheWriter(
        tmp_path, provenance=provenance, hidden_size=128
    )
    writer.append(make_batch())
    resumed = CalibrationCacheWriter(
        tmp_path, provenance=provenance, hidden_size=128, resume=True
    )
    resumed.append(make_batch(tokens=4))

    batches = list(CalibrationCacheReader(tmp_path))
    manifest = json.loads((tmp_path / "calibration_manifest.json").read_text())
    assert [batch.token_ids.numel() for batch in batches] == [8, 4]
    assert manifest["total_tokens"] == 12
    assert manifest["total_sequences"] == 4

    manifest["shards"][0]["sha256"] = "0" * 64
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        list(CalibrationCacheReader(tmp_path))


def test_dspark_model_forward_and_strict_weight_mapping():
    config = DSparkConfig(32, 128, 128, num_hidden_layers=1, markov_order=2)
    model = DSparkCalibrationModel(config)
    batch = make_batch(hidden=128)
    logits, confidence = model(
        batch.target_hidden_states, batch.token_ids, batch.positions
    )
    assert logits.shape == (8, 2, 32)
    assert confidence.shape == (8, 2)

    checkpoint = {
        f"draft_model.{name}": tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    mapped = map_dspark_state_dict(model, checkpoint)
    assert set(mapped) == set(model.state_dict())
    checkpoint.pop(next(iter(checkpoint)))
    with pytest.raises(ValueError, match="incomplete"):
        map_dspark_state_dict(model, checkpoint)


def test_symmetric_gptq_quantizer_emits_runtime_loader_layout():
    torch.manual_seed(0)
    weight = torch.randn(24, 128) * 0.1
    activations = torch.randn(256, 128)
    accumulator = HessianAccumulator(128)
    accumulator.add(activations[:128])
    accumulator.add(activations[128:])
    config = GPTQQuantizerConfig()
    packed = quantize_linear_gptq(
        weight, accumulator.finalize(config.damping_percent), config
    )

    assert packed["qweight"].shape == (16, 24)
    assert packed["scales"].shape == (1, 24)
    assert packed["qzeros"].shape == (1, 3)
    assert packed["g_idx"].shape == (128,)
    GPTQConfig.from_dict(config.as_checkpoint_dict())
    restored = dequantize_gptq_weight(**packed)
    assert torch.isfinite(restored).all()
    assert (weight - restored).pow(2).mean().sqrt() < 0.04
    actual = gptq_linear_reference(activations[:4], **packed)
    assert actual.shape == (4, 24)


def test_synthetic_dspark_checkpoint_round_trip(tmp_path):
    torch.manual_seed(1)
    config = DSparkConfig(16, 128, 128, num_hidden_layers=1, markov_order=2)
    model = DSparkCalibrationModel(config).eval()
    quantizer = GPTQQuantizerConfig()
    tensors = quantize_dspark_model(model, [make_batch(hidden=128)], quantizer)
    assert "input_projection.weight" not in tensors
    assert "input_projection.qweight" in tensors
    assert "markov_head.weight" in tensors

    save_gptq_checkpoint(
        tensors,
        tmp_path,
        model_config={
            "model_type": "dspark",
            "vocab_size": 16,
            "hidden_size": 128,
            "intermediate_size": 128,
        },
        quantizer=quantizer,
        max_shard_bytes=64 * 1024,
    )
    loaded = load_gptq_checkpoint_tensors(tmp_path)
    checkpoint_config = json.loads((tmp_path / "config.json").read_text())
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    GPTQConfig.from_dict(checkpoint_config["quantization_config"])
    assert set(loaded) == set(tensors)
    assert set(index["weight_map"]) == set(tensors)


def test_generated_linear_tensors_are_accepted_by_production_loader(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    weight = torch.randn(128, 128)
    activations = torch.randn(128, 128)
    accumulator = HessianAccumulator(128)
    accumulator.add(activations)
    config = GPTQQuantizerConfig()
    packed = quantize_linear_gptq(
        weight, accumulator.finalize(config.damping_percent), config
    )
    save_file(
        {f"proj.{name}": tensor for name, tensor in packed.items()},
        str(tmp_path / "model.safetensors"),
    )

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ReplicatedLinear(128, 128, quant_config=GPTQConfig())

    model = ToyModel()
    load_model(model, str(tmp_path))
    assert model.proj.weight is None
    assert model.proj._gptq_symmetric_zero
    assert model.proj._gptq_runtime_processed


def test_dspark_dry_run_does_not_allocate_or_require_checkpoints(tmp_path):
    config = {
        "vocab_size": 32,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "markov_order": 2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = quantize_dspark_main(["--config", str(path), "--dry-run"])
    expected = projected_dspark_bytes(DSparkConfig.from_dict(config))
    assert result == expected
    assert result["projected_total_bytes"] > 0
