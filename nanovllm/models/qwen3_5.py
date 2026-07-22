import math
import weakref

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.deltanet import packed_causal_conv1d
from nanovllm.layers.deltanet_chunk import gated_delta_packed
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gptq import GPTQConfig
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


_GDN_LAYERS = weakref.WeakValueDictionary()


@torch.library.custom_op(
    "nanovllm::qwen_gdn_core",
    mutates_args={"core_output"},
)
def _qwen_gdn_core(
    projected_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_output: torch.Tensor,
    layer_id: int,
) -> None:
    """Opaque stateful GDN core; projections and output remain compilable."""
    layer = _GDN_LAYERS[layer_id]
    layer._forward_core(projected_qkv, b, a, core_output)


@_qwen_gdn_core.register_fake
def _qwen_gdn_core_fake(
    projected_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_output: torch.Tensor,
    layer_id: int,
) -> None:
    return None


class Qwen3_5RMSNorm(nn.Module):
    """One-centered RMSNorm used by Qwen3.5/3.6 checkpoints."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer(
            "eps",
            torch.tensor(eps, dtype=torch.float32),
            persistent=False,
        )
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(original_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(x)
        x = x.float().add(residual.float()).to(x.dtype)
        return self._norm(x), x


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer(
            "eps",
            torch.tensor(eps, dtype=torch.float32),
            persistent=False,
        )
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        x = x * F.silu(gate.float())
        return x.to(original_dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Correctness-first packed DeltaNet implementation.

    The recurrent and causal-convolution states live in ModelRunner and are
    supplied through the per-step context. This keeps state lifetime aligned
    with request abort, finish, and scheduler preemption.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        state_idx: int,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.layer_idx = layer_idx
        self.state_idx = state_idx
        self._core_op_id = id(self)
        _GDN_LAYERS[self._core_op_id] = self

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, eps=config.rms_norm_eps
        )
        self.out_proj = ReplicatedLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.in_proj_qkvz = MergedColumnParallelLinear(
            self.hidden_size,
            [self.conv_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            self.hidden_size,
            [self.num_v_heads, self.num_v_heads],
            bias=False,
            quant_config=None,
        )

    def _forward_packed_core(
        self,
        projected_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_state_slab: torch.Tensor,
        cu_seqlens: torch.Tensor,
        chunk_indices: torch.Tensor,
        cu_chunks: torch.Tensor,
        chunk_sequences: torch.Tensor,
        recurrent_sequences: torch.Tensor,
        state_slots: torch.Tensor,
        branch_state_slots: torch.Tensor,
        recurrent_state_slab: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        num_tokens = projected_qkv.size(0)
        mixed_qkv = packed_causal_conv1d(
            projected_qkv.contiguous(),
            self.conv1d.weight,
            cu_seqlens,
            state_slots,
            branch_state_slots,
            conv_state_slab,
            max_seqlen_q,
        )

        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(num_tokens, self.num_k_heads, self.head_k_dim)
        key = key.view(num_tokens, self.num_k_heads, self.head_k_dim)
        value = value.view(num_tokens, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads != self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=1)
            key = key.repeat_interleave(repeats, dim=1)
        query = _l2norm(query) / math.sqrt(self.head_k_dim)
        key = _l2norm(key)
        value = value.float()
        decay = (
            -self.A_log.float().exp()
            * F.softplus(a.float() + self.dt_bias.float())
        ).exp()
        beta = b.float().sigmoid()

        core = gated_delta_packed(
            query,
            key,
            value,
            beta.float(),
            decay,
            cu_seqlens,
            chunk_indices,
            cu_chunks,
            chunk_sequences,
            recurrent_sequences,
            state_slots,
            branch_state_slots,
            recurrent_state_slab,
        )
        return core.to(projected_qkv.dtype)

    def _forward_core(
        self,
        projected_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_output: torch.Tensor,
    ) -> None:
        context = get_context()
        metadata = context.gdn
        if metadata is None:
            raise ValueError("Qwen GDN execution requires prepared GDN metadata")
        num_actual_tokens = context.signature.num_tokens
        projected_qkv = projected_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]
        output = self._forward_packed_core(
            projected_qkv,
            b,
            a,
            metadata.conv_slab[self.state_idx],
            metadata.cu_seqlens,
            metadata.chunk_indices,
            metadata.cu_chunks,
            metadata.chunk_sequences,
            metadata.recurrent_sequences,
            metadata.state_slots,
            metadata.branch_state_slots,
            metadata.recurrent_slab[self.state_idx],
            context.attention.max_seqlen_q,
        )
        core_output[:num_actual_tokens].copy_(output)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projection -> opaque stateful GDN core -> output projection."""
        num_tokens = hidden_states.size(0)
        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        projected_qkv, z = mixed_qkvz.split(
            [self.conv_dim, self.value_dim],
            dim=-1,
        )
        z = z.view(
            num_tokens,
            self.num_v_heads,
            self.head_v_dim,
        )
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()
        core = torch.zeros(
            num_tokens,
            self.num_v_heads,
            self.head_v_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.nanovllm.qwen_gdn_core(
            projected_qkv,
            b,
            a,
            core,
            self._core_op_id,
        )
        core = self.norm(core, z)
        return self.out_proj(core.reshape(num_tokens, self.value_dim))


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("Qwen3.5/3.6 attention v1 only supports TP=1")
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.q_proj_size = self.q_size * 2
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_proj_size, self.kv_size, self.kv_size],
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            rope_parameters = getattr(config, "rope_scaling", None)
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        rope_theta = rope_parameters.get(
            "rope_theta", getattr(config, "rope_theta", 10_000_000)
        )
        rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        query_and_gate, key, value = qkv.split(
            [self.q_proj_size, self.kv_size, self.kv_size], dim=-1
        )
        query, gate = query_and_gate.view(
            -1, self.num_heads, self.head_dim * 2
        ).chunk(2, dim=-1)
        gate = gate.reshape(-1, self.q_size)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value).flatten(1, -1)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output)


class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        state_idx: int | None,
        quant_config: GPTQConfig | None = None,
        block_type: str | None = None,
    ):
        super().__init__()
        self.block_type = block_type or config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            if state_idx is None:
                raise ValueError("linear attention layer requires a state index")
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config, layer_idx, state_idx, quant_config
            )
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config, layer_idx, quant_config
            )
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type {self.block_type!r}")
        self.mlp = Qwen3_5MLP(config, quant_config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = (
                self.input_layernorm(hidden_states),
                hidden_states,
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5TextModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        state_idx = 0
        layers = []
        for layer_idx, layer_type in enumerate(config.layer_types):
            current_state_idx = (
                state_idx if layer_type == "linear_attention" else None
            )
            layers.append(
                Qwen3_5DecoderLayer(
                    config, layer_idx, current_state_idx, quant_config
                )
            )
            if current_state_idx is not None:
                state_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5Model(nn.Module):
    """Text-only outer container matching checkpoint parameter prefixes."""

    def __init__(self, config, quant_config: GPTQConfig | None = None):
        super().__init__()
        self.language_model = Qwen3_5TextModel(config, quant_config)


class Qwen3_5ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "in_proj_qkv": ("in_proj_qkvz", 0),
        "in_proj_z": ("in_proj_qkvz", 1),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
    }
    skipped_weight_prefixes = ("model.visual.", "mtp.")

    def __init__(
        self,
        config,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("Qwen3.5/3.6 text inference v1 only supports TP=1")
        self.config = config
        self.model = Qwen3_5Model(config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = (
                self.model.language_model.embed_tokens.weight.data
            )

        self.num_linear_attention_layers = sum(
            layer_type == "linear_attention"
            for layer_type in config.layer_types
        )
        self.delta_conv_dim = (
            config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.delta_conv_kernel_size = config.linear_conv_kernel_dim
        self.delta_num_value_heads = config.linear_num_value_heads
        self.delta_key_head_dim = config.linear_key_head_dim
        self.delta_value_head_dim = config.linear_value_head_dim

    def create_delta_state(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state = torch.zeros(
            self.num_linear_attention_layers,
            self.delta_conv_dim,
            self.delta_conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        recurrent_state = torch.zeros(
            self.num_linear_attention_layers,
            self.delta_num_value_heads,
            self.delta_key_head_dim,
            self.delta_value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        return conv_state, recurrent_state

    def create_delta_state_slab(
        self,
        capacity: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state = torch.zeros(
            self.num_linear_attention_layers,
            capacity,
            self.delta_conv_dim,
            self.delta_conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        recurrent_state = torch.zeros(
            self.num_linear_attention_layers,
            capacity,
            self.delta_num_value_heads,
            self.delta_key_head_dim,
            self.delta_value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        return conv_state, recurrent_state

    def delta_state_bytes(self, dtype: torch.dtype) -> int:
        conv_values = (
            self.num_linear_attention_layers
            * self.delta_conv_dim
            * self.delta_conv_kernel_size
        )
        recurrent_values = (
            self.num_linear_attention_layers
            * self.delta_num_value_heads
            * self.delta_key_head_dim
            * self.delta_value_head_dim
        )
        return (
            conv_values * torch.empty((), dtype=dtype).element_size()
            + recurrent_values * torch.empty(
                (), dtype=torch.float32
            ).element_size()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
