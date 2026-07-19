import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.deltanet import (
    gated_delta_recurrent,
    gated_delta_recurrent_packed,
)
from nanovllm.layers.deltanet_chunk import (
    DELTA_CHUNK_MIN_TOKENS,
    gated_delta_hybrid_packed,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gptq import GPTQConfig
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


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
        self.in_proj_qkv = ReplicatedLinear(
            self.hidden_size,
            self.conv_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.in_proj_z = ReplicatedLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.in_proj_b = ReplicatedLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=None,
        )
        self.in_proj_a = ReplicatedLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=None,
        )

    def _transient_state(self, device, dtype):
        conv = torch.zeros(
            self.state_idx + 1,
            self.conv_dim,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        recurrent = torch.zeros(
            self.state_idx + 1,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )
        return conv, recurrent

    def _forward_sequence(
        self,
        hidden_states: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.size(0)
        conv_states, recurrent_states = state
        conv_state = conv_states[self.state_idx]
        recurrent_state = recurrent_states[self.state_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(0, 1)
        z = self.in_proj_z(hidden_states).view(
            seq_len, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)

        conv_input = torch.cat((conv_state, mixed_qkv), dim=-1)
        conv_state.copy_(conv_input[:, -self.conv_kernel_size :])
        mixed_qkv = F.conv1d(
            conv_input.unsqueeze(0),
            self.conv1d.weight,
            bias=None,
            groups=self.conv_dim,
        )[0, :, -seq_len:]
        mixed_qkv = F.silu(mixed_qkv.transpose(0, 1))

        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(seq_len, self.num_v_heads, self.head_v_dim)
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

        core = gated_delta_recurrent(
            query,
            key,
            value,
            beta.float(),
            decay,
            recurrent_state,
        ).to(hidden_states.dtype)
        core = self.norm(core, z)
        return self.out_proj(core.reshape(seq_len, self.value_dim))

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        states: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        conv_state_slab: torch.Tensor | None,
        slices: tuple[tuple[int, int], ...],
        cu_seqlens: torch.Tensor,
        chunk_indices: torch.Tensor,
        cu_chunks: torch.Tensor,
        chunk_sequences: torch.Tensor,
        recurrent_sequences: torch.Tensor,
        state_slots: torch.Tensor,
        recurrent_state_slab: torch.Tensor,
        max_seqlen_q: int,
        uniform_decode: bool,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        projected_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).view(
            num_tokens, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)

        if uniform_decode:
            if conv_state_slab is None:
                raise ValueError("uniform DeltaNet requires a convolution state slab")
            num_sequences = state_slots.numel()
            if num_tokens != num_sequences * max_seqlen_q:
                raise ValueError("uniform DeltaNet token shape is inconsistent")
            state_indices = state_slots.long()
            conv_state = conv_state_slab[self.state_idx].index_select(
                0, state_indices
            )
            mixed_qkv = projected_qkv.view(
                num_sequences, max_seqlen_q, self.conv_dim
            ).transpose(1, 2)
            conv_input = torch.cat((conv_state, mixed_qkv), dim=-1)
            conv_state_slab[self.state_idx].index_copy_(
                0,
                state_indices,
                conv_input[:, :, -self.conv_kernel_size :],
            )
            convolved = F.conv1d(
                conv_input,
                self.conv1d.weight,
                bias=None,
                groups=self.conv_dim,
            )[:, :, -max_seqlen_q:]
            mixed_qkv = F.silu(convolved.transpose(1, 2)).reshape(
                num_tokens, self.conv_dim
            )
        else:
            conv_outputs = []
            for (start, end), state in zip(slices, states):
                conv_state = state[0][self.state_idx]
                mixed_qkv = projected_qkv[start:end].transpose(0, 1)
                conv_input = torch.cat((conv_state, mixed_qkv), dim=-1)
                conv_state.copy_(conv_input[:, -self.conv_kernel_size :])
                convolved = F.conv1d(
                    conv_input.unsqueeze(0),
                    self.conv1d.weight,
                    bias=None,
                    groups=self.conv_dim,
                )[0, :, -(end - start) :]
                conv_outputs.append(F.silu(convolved.transpose(0, 1)))
            mixed_qkv = torch.cat(conv_outputs, dim=0)

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

        if max_seqlen_q >= DELTA_CHUNK_MIN_TOKENS:
            core = gated_delta_hybrid_packed(
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
                recurrent_state_slab,
            )
        else:
            core = gated_delta_recurrent_packed(
                query,
                key,
                value,
                beta.float(),
                decay,
                cu_seqlens,
                state_slots,
                recurrent_state_slab,
            )
        core = core.to(hidden_states.dtype)
        core = self.norm(core, z)
        return self.out_proj(core.reshape(num_tokens, self.value_dim))

    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        num_actual_tokens = context.num_actual_tokens or hidden_states.size(0)
        real_hidden_states = hidden_states[:num_actual_tokens]
        slices = context.sequence_slices or ((0, num_actual_tokens),)
        states = context.delta_states
        graph_safe_uniform = (
            context.is_uniform_decode and context.delta_conv_slab is not None
        )
        if not states and not graph_safe_uniform:
            states = tuple(
                self._transient_state(hidden_states.device, hidden_states.dtype)
                for _ in slices
            )
        if not graph_safe_uniform and len(states) != len(slices):
            raise ValueError("DeltaNet state count does not match packed sequences")

        if context.delta_recurrent_slab is not None:
            if (
                context.delta_state_slots is None
                or context.cu_seqlens_q is None
                or context.delta_chunk_indices is None
                or context.delta_cu_chunks is None
                or context.delta_chunk_sequences is None
                or context.delta_recurrent_sequences is None
            ):
                raise ValueError("packed DeltaNet metadata is incomplete")
            output = self._forward_packed(
                real_hidden_states,
                states,
                context.delta_conv_slab,
                slices,
                context.cu_seqlens_q,
                context.delta_chunk_indices,
                context.delta_cu_chunks,
                context.delta_chunk_sequences,
                context.delta_recurrent_sequences,
                context.delta_state_slots,
                context.delta_recurrent_slab[self.state_idx],
                context.max_seqlen_q,
                context.is_uniform_decode,
            )
        else:
            outputs = [
                self._forward_sequence(real_hidden_states[start:end], state)
                for (start, end), state in zip(slices, states)
            ]
            output = torch.cat(outputs, dim=0)
        if hidden_states.size(0) != num_actual_tokens:
            output = F.pad(
                output, (0, 0, 0, hidden_states.size(0) - num_actual_tokens)
            )
        return output


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
