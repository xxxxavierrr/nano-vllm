import os
from glob import glob

import torch
from safetensors import safe_open

from nanovllm.utils.loader import default_weight_loader


def expected_mtp_shapes(config) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = config.head_dim
    q_size = config.num_attention_heads * head_dim * 2
    kv_size = config.num_key_value_heads * head_dim
    return {
        "mtp.fc.weight": (hidden, hidden * 2),
        "mtp.layers.0.input_layernorm.weight": (hidden,),
        "mtp.layers.0.mlp.down_proj.weight": (hidden, intermediate),
        "mtp.layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "mtp.layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
        "mtp.layers.0.self_attn.k_norm.weight": (head_dim,),
        "mtp.layers.0.self_attn.k_proj.weight": (kv_size, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (
            hidden,
            config.num_attention_heads * head_dim,
        ),
        "mtp.layers.0.self_attn.q_norm.weight": (head_dim,),
        "mtp.layers.0.self_attn.q_proj.weight": (q_size, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv_size, hidden),
        "mtp.norm.weight": (hidden,),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
    }


def inspect_mtp_checkpoint(path: str, config) -> dict[str, tuple[int, ...]]:
    if not os.path.isdir(path):
        raise ValueError(f"MTP model path is not a directory: {path}")
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"MTP model path contains no safetensors files: {path}")

    expected = expected_mtp_shapes(config)
    found: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, object] = {}
    for file in files:
        with safe_open(file, "pt", "cpu") as checkpoint:
            for name in checkpoint.keys():
                if not name.startswith("mtp."):
                    continue
                if name in found:
                    raise ValueError(f"duplicate MTP tensor {name!r}")
                tensor_slice = checkpoint.get_slice(name)
                found[name] = tuple(tensor_slice.get_shape())
                dtypes[name] = tensor_slice.get_dtype()

    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing:
        raise ValueError(f"incomplete MTP checkpoint; missing: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unsupported MTP tensors: {', '.join(extra)}")
    for name, shape in expected.items():
        if found[name] != shape:
            raise ValueError(
                f"MTP tensor {name} shape {found[name]} does not match {shape}"
            )
        if str(dtypes[name]) != "BF16":
            raise TypeError(
                f"MTP tensor {name} must use BF16, got {dtypes[name]}"
            )
    return found


def load_mtp_model(model, path: str, config) -> None:
    expected = inspect_mtp_checkpoint(path, config)
    packed_mapping = getattr(model, "packed_modules_mapping", {})
    loaded = set()
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as checkpoint:
            for checkpoint_name in checkpoint.keys():
                if checkpoint_name not in expected:
                    continue
                parameter_name = checkpoint_name.removeprefix("mtp.")
                tensor = checkpoint.get_tensor(checkpoint_name)
                for source_name, (packed_name, shard_id) in packed_mapping.items():
                    if source_name in parameter_name:
                        parameter_name = parameter_name.replace(
                            source_name, packed_name
                        )
                        parameter = model.get_parameter(parameter_name)
                        loader = getattr(parameter, "weight_loader")
                        loader(parameter, tensor, shard_id)
                        break
                else:
                    parameter = model.get_parameter(parameter_name)
                    loader = getattr(
                        parameter, "weight_loader", default_weight_loader
                    )
                    loader(parameter, tensor)
                loaded.add(checkpoint_name)
    missing = sorted(set(expected) - loaded)
    if missing:
        raise ValueError(f"failed to load MTP tensors: {', '.join(missing)}")
