from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from nanovllm.calibration.cache import CalibrationBatch
from nanovllm.calibration.dspark import (
    DEFAULT_DSPARK_PREFIX_MAP,
    DSparkCalibrationModel,
    DSparkConfig,
)
from nanovllm.calibration.gptq_quantizer import (
    GPTQQuantizerConfig,
    HessianAccumulator,
    quantize_linear_gptq,
)
from nanovllm.layers.gptq import dequantize_gptq_weight


def _map_name(name: str, prefix_map: dict[str, str]) -> str:
    for source_prefix, target_prefix in prefix_map.items():
        if name.startswith(source_prefix):
            return target_prefix + name[len(source_prefix):]
    return name


def load_dspark_safetensors(
    model: DSparkCalibrationModel,
    checkpoint_dir: str | Path,
    *,
    prefix_map: dict[str, str] | None = None,
) -> None:
    """Strict streaming BF16 draft loader that never constructs a target."""
    prefix_map = prefix_map or DEFAULT_DSPARK_PREFIX_MAP
    expected = dict(model.named_parameters())
    seen: set[str] = set()
    files = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not files:
        raise ValueError("DSpark checkpoint contains no safetensors files")
    unknown: list[str] = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                target_name = _map_name(source_name, prefix_map)
                parameter = expected.get(target_name)
                if parameter is None:
                    unknown.append(source_name)
                    continue
                if target_name in seen:
                    raise ValueError(f"duplicate DSpark tensor {target_name}")
                tensor = handle.get_tensor(source_name)
                if tensor.shape != parameter.shape:
                    raise ValueError(
                        f"DSpark shape mismatch for {source_name}: "
                        f"{tuple(tensor.shape)} != {tuple(parameter.shape)}"
                    )
                parameter.data.copy_(tensor.to(parameter.dtype))
                seen.add(target_name)
    missing = sorted(set(expected) - seen)
    if missing or unknown:
        raise ValueError(
            f"DSpark checkpoint incomplete; missing={missing}, unknown={sorted(unknown)}"
        )


@torch.inference_mode()
def _capture_hessian(
    model: DSparkCalibrationModel,
    module: torch.nn.Linear,
    batches: list[CalibrationBatch],
    damping_percent: float,
) -> torch.Tensor:
    accumulator = HessianAccumulator(module.in_features)

    def hook(_module, inputs):
        accumulator.add(inputs[0])

    handle = module.register_forward_pre_hook(hook)
    try:
        for batch in batches:
            model(
                batch.target_hidden_states.to(next(model.parameters()).dtype),
                batch.token_ids,
                batch.positions,
            )
    finally:
        handle.remove()
    return accumulator.finalize(damping_percent)


@torch.inference_mode()
def quantize_dspark_model(
    model: DSparkCalibrationModel,
    calibration_batches: list[CalibrationBatch],
    config: GPTQQuantizerConfig = GPTQQuantizerConfig(),
) -> dict[str, torch.Tensor]:
    """Layer-at-a-time CPU GPTQ with prior-layer error propagated forward."""
    config.validate()
    if not calibration_batches:
        raise ValueError("DSpark quantization requires calibration batches")
    for batch in calibration_batches:
        batch.validate()
        if batch.target_hidden_states.shape[1] != model.config.hidden_size:
            raise ValueError("calibration hidden size differs from DSpark config")

    output = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    for name, module in model.quantizable_linears():
        if module.in_features % config.group_size:
            raise ValueError(f"{name} input width is not divisible by group_size")
        hessian = _capture_hessian(
            model, module, calibration_batches, config.damping_percent
        )
        packed = quantize_linear_gptq(module.weight, hessian, config)
        output.pop(f"{name}.weight")
        for suffix, tensor in packed.items():
            output[f"{name}.{suffix}"] = tensor.contiguous()
        # CPU-only sequential calibration uses the quantized reconstruction so
        # later modules observe upstream quantization error.
        reconstructed = dequantize_gptq_weight(**packed).to(module.weight.dtype)
        module.weight.data.copy_(reconstructed)
    return output


def projected_dspark_bytes(
    config: DSparkConfig,
    quantizer: GPTQQuantizerConfig = GPTQQuantizerConfig(),
) -> dict[str, int]:
    """Conservative architecture-level size estimate without model allocation."""
    quantizer.validate()
    h = config.hidden_size
    i = config.intermediate_size
    quantized_shapes = [(h, 2 * h)]
    for _ in range(config.num_hidden_layers):
        quantized_shapes.extend(((3 * h, h), (h, h), (2 * i, h), (h, i)))
    packed = 0
    original = 0
    for out_features, in_features in quantized_shapes:
        original += out_features * in_features * 2
        packed += out_features * in_features // 2
        packed += (in_features // quantizer.group_size) * out_features * 2
        packed += (in_features // quantizer.group_size) * ((out_features + 7) // 8) * 4
        packed += in_features * 4
    unquantized = (
        config.vocab_size * h * 2
        + h * config.vocab_size * config.markov_order * 2
        + h * config.markov_order * 2
        + config.markov_order * 2
        + (2 * config.num_hidden_layers + 1) * h * 2
    )
    return {
        "quantized_linear_bf16_bytes": original,
        "quantized_linear_gptq_bytes": packed,
        "unquantized_bytes": unquantized,
        "projected_total_bytes": packed + unquantized,
    }


def save_gptq_checkpoint(
    tensors: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    model_config: dict,
    quantizer: GPTQQuantizerConfig = GPTQQuantizerConfig(),
    max_shard_bytes: int = 2 * 1024**3,
) -> None:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        (output_dir / "model.safetensors.index.json").exists()
        or (output_dir / "config.json").exists()
        or any(output_dir.glob("model-*-of-*.safetensors"))
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing GPTQ checkpoint in {output_dir}"
        )
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    weight_map = {}
    total = len(shards)
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        temporary = output_dir / f"{filename}.partial"
        save_file(shard, str(temporary))
        temporary.replace(output_dir / filename)
        weight_map.update({name: filename for name in shard})
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config_payload = {
        **model_config,
        "quantization_config": quantizer.as_checkpoint_dict(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_gptq_checkpoint_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(Path(path).glob("model-*-of-*.safetensors")):
        tensors.update(load_file(str(shard), device="cpu"))
    return tensors
