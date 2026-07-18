import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.gptq import GPTQConfig, default_g_idx
from nanovllm.layers.gptq_kernel import gptq_w4a16_linear


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_config = quant_config
        self._gptq_loaded: set[tuple[str, object]] = set()
        self._g_idx_reference_shard: object | None = None
        self._gptq_symmetric_zero = False

        if quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
            self.register_parameter("qweight", None)
            self.register_parameter("scales", None)
            self.register_parameter("qzeros", None)
            self.register_parameter("g_idx", None)
        else:
            quant_config.validate()
            if self.tp_size != 1:
                raise ValueError("GPTQ W4A16 v1 only supports tensor parallel size 1")
            if input_size % quant_config.values_per_int32:
                raise ValueError("GPTQ input size must be divisible by 8")
            if input_size % quant_config.group_size:
                raise ValueError("GPTQ input size must be divisible by group_size=128")
            if output_size % quant_config.values_per_int32:
                raise ValueError("GPTQ output size must be divisible by 8")
            self.register_parameter("weight", None)
            groups = input_size // quant_config.group_size
            self.qweight = nn.Parameter(
                torch.empty(
                    input_size // quant_config.values_per_int32,
                    output_size,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.scales = nn.Parameter(
                torch.empty(groups, output_size), requires_grad=False
            )
            self.qzeros = nn.Parameter(
                torch.empty(
                    groups,
                    output_size // quant_config.values_per_int32,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.g_idx = nn.Parameter(
                torch.empty(input_size, dtype=torch.int32), requires_grad=False
            )
            if not quant_config.desc_act:
                self.g_idx.data.copy_(
                    default_g_idx(input_size, groups, device=self.g_idx.device)
                )
            for name in ("qweight", "scales", "qzeros", "g_idx"):
                param = getattr(self, name)
                param.gptq_name = name
                param.weight_loader = self.gptq_weight_loader

        self.register_buffer("weight_scale", None, persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    @property
    def is_gptq(self) -> bool:
        return self.quant_config is not None

    def _gptq_shard_ids(self) -> tuple[object, ...]:
        return (None,)

    def _gptq_output_shard(self, shard_id: object) -> tuple[int, int]:
        if shard_id is not None:
            raise ValueError(
                f"{type(self).__name__} does not accept GPTQ shard {shard_id!r}"
            )
        return 0, self.output_size

    def gptq_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: object = None,
    ):
        if not self.is_gptq:
            raise RuntimeError("received a GPTQ tensor for a non-GPTQ Linear")
        name = param.gptq_name
        shard_ids = self._gptq_shard_ids()
        if loaded_shard_id not in shard_ids:
            raise ValueError(
                f"invalid GPTQ shard {loaded_shard_id!r} for {type(self).__name__}"
            )
        if name in ("qweight", "qzeros", "g_idx"):
            if loaded_weight.dtype is not torch.int32:
                raise TypeError(
                    f"{name} must use int32 checkpoint packing, got {loaded_weight.dtype}"
                )
        elif loaded_weight.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                f"scales must use float16/bfloat16, got {loaded_weight.dtype}"
            )

        if name == "g_idx":
            if loaded_weight.shape != param.shape:
                raise ValueError(
                    f"g_idx shape {tuple(loaded_weight.shape)} does not match "
                    f"{tuple(param.shape)}"
                )
            groups = self.input_size // self.quant_config.group_size
            if loaded_weight.numel() and (
                loaded_weight.min().item() < 0
                or loaded_weight.max().item() >= groups
            ):
                raise ValueError(f"g_idx values must be in [0, {groups})")
            loaded_cpu = loaded_weight.to(device="cpu", dtype=torch.int32)
            if self._g_idx_reference_shard is None:
                param.data.copy_(loaded_cpu.to(param.device))
                self._g_idx_reference_shard = loaded_shard_id
            else:
                reference = param.detach().cpu()
                if not torch.equal(reference, loaded_cpu):
                    raise ValueError(
                        f"fused GPTQ g_idx mismatch in {type(self).__name__}: "
                        f"shard {loaded_shard_id!r} differs from "
                        f"{self._g_idx_reference_shard!r}"
                    )
        else:
            offset, size = self._gptq_output_shard(loaded_shard_id)
            if name == "qzeros":
                if offset % 8 or size % 8:
                    raise ValueError("fused GPTQ qzeros require 8-column alignment")
                target = param.data.narrow(1, offset // 8, size // 8)
            else:
                target = param.data.narrow(1, offset, size)
            if loaded_weight.shape != target.shape:
                raise ValueError(
                    f"{name} shard {loaded_shard_id!r} shape "
                    f"{tuple(loaded_weight.shape)} does not match "
                    f"{tuple(target.shape)}"
                )
            target.copy_(
                loaded_weight.to(device=target.device, dtype=target.dtype)
            )
        self._gptq_loaded.add((name, loaded_shard_id))

    def validate_gptq_loaded(self):
        if not self.is_gptq:
            return
        required_names = ["qweight", "scales", "qzeros"]
        if self.quant_config.desc_act:
            required_names.append("g_idx")
        required = {
            (name, shard_id)
            for name in required_names
            for shard_id in self._gptq_shard_ids()
        }
        missing = sorted(
            required - self._gptq_loaded,
            key=lambda item: (item[0], repr(item[1])),
        )
        if missing:
            details = ", ".join(f"{name}[{shard!r}]" for name, shard in missing)
            raise ValueError(
                f"incomplete GPTQ checkpoint for {type(self).__name__}: {details}"
            )
        if self.quant_config.sym:
            stored_symmetric_zero = 0x77777777
            self._gptq_symmetric_zero = bool(
                torch.all(self.qzeros == stored_symmetric_zero).item()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def quantize_fp8(self):
        if self.is_gptq:
            raise RuntimeError("cannot apply FP8 quantization to a GPTQ Linear")
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max
        scale = self.weight.detach().abs().amax(dim=1, keepdim=True).float()
        scale = scale.clamp_min(1e-12).div_(fp8_max)
        weight = self.weight.detach().div(scale).to(fp8_dtype)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = scale

    def linear(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        bias = self.bias if use_bias else None
        if self.is_gptq:
            return gptq_w4a16_linear(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
                self.g_idx,
                bias,
                symmetric_zero=self._gptq_symmetric_zero,
            )
        if self.weight_scale is None:
            return F.linear(x, self.weight, bias)

        input_shape = x.shape
        output_dtype = x.dtype
        x = x.reshape(-1, input_shape[-1])
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max
        input_scale = x.abs().amax(dim=1, keepdim=True).float()
        input_scale = input_scale.clamp_min(1e-12).div_(fp8_max)
        x = x.div(input_scale).to(fp8_dtype)
        output = torch._scaled_mm(
            x,
            self.weight.t(),
            scale_a=input_scale,
            scale_b=self.weight_scale.t(),
            out_dtype=output_dtype,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*input_shape[:-1], self.weight.shape[0])


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: GPTQConfig | None = None,
    ):
        super().__init__(
            input_size, output_size, bias, quant_config=quant_config
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: GPTQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: GPTQConfig | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size, sum(output_sizes), bias, quant_config
        )

    def _gptq_shard_ids(self) -> tuple[object, ...]:
        return tuple(range(len(self.output_sizes)))

    def _gptq_output_shard(self, shard_id: object) -> tuple[int, int]:
        if not isinstance(shard_id, int) or not 0 <= shard_id < len(self.output_sizes):
            return super()._gptq_output_shard(shard_id)
        return sum(self.output_sizes[:shard_id]), self.output_sizes[shard_id]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: GPTQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config)

    def _gptq_shard_ids(self) -> tuple[object, ...]:
        return ("q", "k", "v")

    def _gptq_output_shard(self, shard_id: object) -> tuple[int, int]:
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            return 0, q_size
        if shard_id == "k":
            return q_size, kv_size
        if shard_id == "v":
            return q_size + kv_size, kv_size
        return super()._gptq_output_shard(shard_id)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: GPTQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x, use_bias=self.tp_rank == 0)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


def quantize_fp8(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.quantize_fp8()
