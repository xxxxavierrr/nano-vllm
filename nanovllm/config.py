import os
from dataclasses import dataclass, field

from transformers import AutoConfig

from nanovllm.engine.cudagraph import CUDAGraphMode
from nanovllm.layers.gptq import GPTQConfig


@dataclass(slots=True)
class Config:
    model: str
    quantization: str | None = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    master_host: str = "127.0.0.1"
    master_port: int = 2333
    shm_name: str = "nanovllm"
    device_ids: list[int] | None = None
    enforce_eager: bool = False
    cudagraph_mode: CUDAGraphMode | str = CUDAGraphMode.FULL_AND_PIECEWISE
    piecewise_max_tokens: int = 512
    hf_config: AutoConfig | None = None
    gptq_config: GPTQConfig | None = field(init=False, default=None)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        checkpoint_config = getattr(self.hf_config, "quantization_config", None)
        if checkpoint_config is None:
            quantization_config = None
            checkpoint_method = None
        else:
            quantization_config = (
                checkpoint_config
                if isinstance(checkpoint_config, dict)
                else checkpoint_config.to_dict()
            )
            checkpoint_method = str(
                quantization_config.get(
                    "quant_method", quantization_config.get("format", "")
                )
            ).lower()

        if self.quantization is None and checkpoint_method == "gptq":
            self.quantization = "gptq"
        if self.quantization not in (None, "fp8", "gptq"):
            raise ValueError("quantization must be None, 'fp8', or 'gptq'")
        if quantization_config is not None and checkpoint_method != "gptq":
            raise ValueError(
                f"unsupported checkpoint quantization method {checkpoint_method!r}"
            )
        if checkpoint_method == "gptq" and self.quantization != "gptq":
            raise ValueError(
                "a GPTQ checkpoint cannot use BF16/FP8 weight loading"
            )
        if self.quantization == "gptq":
            if quantization_config is None:
                raise ValueError("GPTQ requires checkpoint quantization_config metadata")
            if self.tensor_parallel_size != 1:
                raise ValueError("GPTQ W4A16 v1 only supports tensor_parallel_size=1")
            self.gptq_config = GPTQConfig.from_dict(quantization_config)

        self.cudagraph_mode = CUDAGraphMode.parse(self.cudagraph_mode)
        if self.enforce_eager:
            self.cudagraph_mode = CUDAGraphMode.NONE
        assert self.piecewise_max_tokens > 0
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.master_port <= 65535
        if self.device_ids is None:
            self.device_ids = list(range(self.tensor_parallel_size))
        assert len(self.device_ids) == self.tensor_parallel_size
        assert len(set(self.device_ids)) == len(self.device_ids)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
