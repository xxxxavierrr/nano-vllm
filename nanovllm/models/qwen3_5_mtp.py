import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm


class Qwen3_5MTP(nn.Module):
    """One-layer BF16 multi-token predictor for Qwen3.5/3.6."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("Qwen3.5/3.6 MTP v1 only supports TP=1")
        if getattr(config, "mtp_num_hidden_layers", 0) != 1:
            raise ValueError("Qwen3.5/3.6 MTP v1 requires exactly one MTP layer")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Qwen3.5/3.6 MTP v1 requires shared token embeddings")
        self.config = config
        self.fc = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(
                    config,
                    layer_idx=0,
                    state_idx=None,
                    quant_config=None,
                    block_type="full_attention",
                )
            ]
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if target_hidden_states.shape != next_token_embeddings.shape:
            raise ValueError(
                "MTP target hidden states and token embeddings must have equal shapes"
            )
        hidden_states = torch.cat(
            (
                self.pre_fc_norm_embedding(next_token_embeddings),
                self.pre_fc_norm_hidden(target_hidden_states),
            ),
            dim=-1,
        )
        hidden_states = self.fc(hidden_states)
        hidden_states, residual = self.layers[0](
            positions, hidden_states, residual=None
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
