from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class DSparkConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int = 5
    markov_order: int = 3
    rms_norm_eps: float = 1.0e-6

    @classmethod
    def from_dict(cls, value: dict) -> "DSparkConfig":
        config = cls(
            vocab_size=int(value["vocab_size"]),
            hidden_size=int(value["hidden_size"]),
            intermediate_size=int(value.get("intermediate_size", value["hidden_size"] * 4)),
            num_hidden_layers=int(value.get("num_hidden_layers", 5)),
            markov_order=int(value.get("markov_order", value.get("num_speculative_tokens", 3))),
            rms_norm_eps=float(value.get("rms_norm_eps", 1.0e-6)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if min(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.markov_order,
        ) <= 0:
            raise ValueError("DSpark dimensions must be positive")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class DFlashCalibrationLayer(nn.Module):
    """Draft-only calibration shell; real-checkpoint semantics require GPU validation."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        hidden = config.hidden_size
        self.input_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.post_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.gate_up_proj = nn.Linear(hidden, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        query, key, value = self.qkv_proj(self.input_norm(hidden_states)).chunk(3, dim=-1)
        gate = torch.sigmoid((query.float() * key.float()) / query.shape[-1] ** 0.5).to(value.dtype)
        hidden_states = residual + self.out_proj(gate * value)
        residual = hidden_states
        gate, up = self.gate_up_proj(self.post_norm(hidden_states)).chunk(2, dim=-1)
        return residual + self.down_proj(F.silu(gate) * up)


class DSparkCalibrationModel(nn.Module):
    """CPU/offline DFlash+Markov calibration model, never the online target."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.input_projection = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            DFlashCalibrationLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.markov_head = nn.Linear(
            config.hidden_size,
            config.vocab_size * config.markov_order,
            bias=False,
        )
        self.confidence_head = nn.Linear(
            config.hidden_size,
            config.markov_order,
            bias=True,
        )

    def forward(
        self,
        target_hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        if target_hidden_states.ndim != 2 or token_ids.ndim != 1:
            raise ValueError("DSpark calibration inputs must be flattened token tensors")
        if target_hidden_states.shape != (token_ids.numel(), self.config.hidden_size):
            raise ValueError("DSpark target hidden-state shape mismatch")
        embeddings = self.embed_tokens(token_ids)
        hidden_states = self.input_projection(
            torch.cat((target_hidden_states, embeddings), dim=-1)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.markov_head(hidden_states).reshape(
            token_ids.numel(), self.config.markov_order, self.config.vocab_size
        )
        confidence = self.confidence_head(hidden_states)
        return logits, confidence

    def quantizable_linears(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not name.endswith(
                ("markov_head", "confidence_head")
            ):
                yield name, module


DEFAULT_DSPARK_PREFIX_MAP = {
    "model.draft_model.": "",
    "draft_model.": "",
    "dspark.": "",
}


def map_dspark_state_dict(
    model: DSparkCalibrationModel,
    checkpoint: dict[str, torch.Tensor],
    prefix_map: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Map an author checkpoint strictly; real Avesed mapping is a GPU gate."""
    prefix_map = prefix_map or DEFAULT_DSPARK_PREFIX_MAP
    expected = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    unknown = []
    for source_name, tensor in checkpoint.items():
        target_name = source_name
        for source_prefix, target_prefix in prefix_map.items():
            if source_name.startswith(source_prefix):
                target_name = target_prefix + source_name[len(source_prefix):]
                break
        if target_name not in expected:
            unknown.append(source_name)
            continue
        if tensor.shape != expected[target_name].shape:
            raise ValueError(
                f"DSpark weight shape mismatch for {source_name}: "
                f"{tuple(tensor.shape)} != {tuple(expected[target_name].shape)}"
            )
        if target_name in mapped:
            raise ValueError(f"duplicate DSpark mapping for {target_name}")
        mapped[target_name] = tensor
    missing = sorted(set(expected) - set(mapped))
    if missing or unknown:
        raise ValueError(
            f"DSpark checkpoint mapping incomplete; missing={missing}, unknown={sorted(unknown)}"
        )
    return mapped
