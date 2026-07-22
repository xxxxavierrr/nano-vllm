from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanovllm.calibration.cache import CalibrationCacheReader
from nanovllm.calibration.checkpoint import (
    load_dspark_safetensors,
    projected_dspark_bytes,
    quantize_dspark_model,
    save_gptq_checkpoint,
)
from nanovllm.calibration.dspark import DSparkCalibrationModel, DSparkConfig
from nanovllm.calibration.gptq_quantizer import GPTQQuantizerConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sequentially calibrate a BF16 DSpark draft into GPTQ INT4"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--bf16-model")
    parser.add_argument("--calibration-cache")
    parser.add_argument("--output")
    parser.add_argument("--max-shard-mib", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not all(
        (args.bf16_model, args.calibration_cache, args.output)
    ):
        parser.error("non-dry-run requires --bf16-model, --calibration-cache, and --output")
    return args


def main(argv=None):
    args = parse_args(argv)
    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = DSparkConfig.from_dict(raw_config)
    quantizer = GPTQQuantizerConfig()
    projection = projected_dspark_bytes(config, quantizer)
    print(json.dumps({"mode": "dry-run" if args.dry_run else "quantize", **projection}, indent=2))
    if args.dry_run:
        return projection

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = DSparkCalibrationModel(config).cpu().eval()
    finally:
        torch.set_default_dtype(default_dtype)
    load_dspark_safetensors(model, args.bf16_model)
    batches = list(CalibrationCacheReader(args.calibration_cache))
    tensors = quantize_dspark_model(model, batches, quantizer)
    save_gptq_checkpoint(
        tensors,
        args.output,
        model_config=raw_config,
        quantizer=quantizer,
        max_shard_bytes=args.max_shard_mib * 2**20,
    )
    return projection


if __name__ == "__main__":
    main()
