from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file


MANIFEST_NAME = "calibration_manifest.json"
CACHE_VERSION = 1


@dataclass(slots=True)
class CalibrationBatch:
    token_ids: torch.Tensor
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    target_hidden_states: torch.Tensor

    def validate(self) -> None:
        if self.token_ids.dtype is not torch.int64 or self.token_ids.ndim != 1:
            raise TypeError("token_ids must be rank-1 int64")
        if self.positions.dtype is not torch.int64 or self.positions.shape != self.token_ids.shape:
            raise TypeError("positions must be int64 with token_ids shape")
        if self.cu_seqlens.dtype is not torch.int32 or self.cu_seqlens.ndim != 1:
            raise TypeError("cu_seqlens must be rank-1 int32")
        if self.cu_seqlens.numel() < 2 or self.cu_seqlens[0].item() != 0:
            raise ValueError("cu_seqlens must begin at zero and contain a sequence")
        if self.cu_seqlens[-1].item() != self.token_ids.numel():
            raise ValueError("cu_seqlens must end at the flattened token count")
        if bool((self.cu_seqlens[1:] < self.cu_seqlens[:-1]).any()):
            raise ValueError("cu_seqlens must be monotonic")
        if self.target_hidden_states.ndim != 2:
            raise ValueError("target_hidden_states must be [tokens, hidden]")
        if self.target_hidden_states.shape[0] != self.token_ids.numel():
            raise ValueError("hidden-state token count mismatch")
        if not self.target_hidden_states.is_floating_point():
            raise TypeError("target_hidden_states must be floating point")

    def cpu_contiguous(self) -> "CalibrationBatch":
        return CalibrationBatch(
            token_ids=self.token_ids.detach().to("cpu").contiguous(),
            positions=self.positions.detach().to("cpu").contiguous(),
            cu_seqlens=self.cu_seqlens.detach().to("cpu").contiguous(),
            target_hidden_states=self.target_hidden_states.detach().to("cpu").contiguous(),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibrationCacheWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        provenance: dict[str, str],
        hidden_size: int,
        resume: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / MANIFEST_NAME
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(f"calibration manifest exists: {self.manifest_path}")
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["hidden_size"] != hidden_size:
                raise ValueError("resume hidden_size differs from manifest")
            if self.manifest["provenance"] != provenance:
                raise ValueError("resume provenance differs from manifest")
        else:
            self.manifest = {
                "version": CACHE_VERSION,
                "hidden_size": hidden_size,
                "provenance": provenance,
                "total_tokens": 0,
                "total_sequences": 0,
                "shards": [],
            }
            self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def append(self, batch: CalibrationBatch) -> Path:
        batch = batch.cpu_contiguous()
        batch.validate()
        if batch.target_hidden_states.shape[1] != self.manifest["hidden_size"]:
            raise ValueError("batch hidden_size differs from manifest")
        index = len(self.manifest["shards"])
        filename = f"calibration-{index:05d}.safetensors"
        final_path = self.output_dir / filename
        temporary = final_path.with_suffix(".safetensors.partial")
        save_file(
            {
                "token_ids": batch.token_ids,
                "positions": batch.positions,
                "cu_seqlens": batch.cu_seqlens,
                "target_hidden_states": batch.target_hidden_states,
            },
            str(temporary),
        )
        temporary.replace(final_path)
        record = {
            "filename": filename,
            "sha256": _sha256(final_path),
            "tokens": batch.token_ids.numel(),
            "sequences": batch.cu_seqlens.numel() - 1,
            "hidden_dtype": str(batch.target_hidden_states.dtype).removeprefix("torch."),
        }
        self.manifest["shards"].append(record)
        self.manifest["total_tokens"] += record["tokens"]
        self.manifest["total_sequences"] += record["sequences"]
        self._write_manifest()
        return final_path


class CalibrationCacheReader:
    def __init__(self, cache_dir: str | Path, *, verify_hashes: bool = True):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads(
            (self.cache_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if self.manifest.get("version") != CACHE_VERSION:
            raise ValueError("unsupported calibration cache version")
        self.verify_hashes = verify_hashes

    def __iter__(self) -> Iterator[CalibrationBatch]:
        for record in self.manifest["shards"]:
            path = self.cache_dir / record["filename"]
            if self.verify_hashes and _sha256(path) != record["sha256"]:
                raise ValueError(f"calibration shard hash mismatch: {path.name}")
            tensors = load_file(str(path), device="cpu")
            batch = CalibrationBatch(**tensors)
            batch.validate()
            yield batch
