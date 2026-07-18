import json

import pytest
from transformers import Qwen3Config

from nanovllm.config import Config


def _write_config(path, **overrides):
    quantization_config = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": True,
        "pack_dtype": "int32",
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
    }
    quantization_config.update(overrides)
    config = Qwen3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
    ).to_dict()
    config["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_config_auto_detects_gptq(tmp_path):
    _write_config(tmp_path)
    config = Config(str(tmp_path), enforce_eager=True)
    assert config.quantization == "gptq"
    assert config.gptq_config.desc_act
    assert config.gptq_config.group_size == 128


def test_config_rejects_gptq_tensor_parallel(tmp_path):
    _write_config(tmp_path)
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        Config(str(tmp_path), tensor_parallel_size=2, enforce_eager=True)


def test_config_rejects_unsupported_gptq_format(tmp_path):
    _write_config(tmp_path, bits=3)
    with pytest.raises(ValueError, match="requires 4 bits"):
        Config(str(tmp_path), enforce_eager=True)


def test_config_rejects_fp8_override_for_gptq_checkpoint(tmp_path):
    _write_config(tmp_path)
    with pytest.raises(ValueError, match="cannot use BF16/FP8"):
        Config(str(tmp_path), quantization="fp8", enforce_eager=True)
