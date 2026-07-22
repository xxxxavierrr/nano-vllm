import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open

from nanovllm.layers.linear import LinearBase


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    skipped_weight_prefixes = getattr(model, "skipped_weight_prefixes", ())
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name.startswith(skipped_weight_prefixes):
                    continue
                for key in packed_modules_mapping:
                    if key in weight_name:
                        packed_name, shard_id = packed_modules_mapping[key]
                        param_name = weight_name.replace(key, packed_name)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))

    for module in model.modules():
        if isinstance(module, LinearBase):
            module.validate_gptq_loaded()
            module.process_gptq_weights_after_loading()
