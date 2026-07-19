import os
from dataclasses import dataclass, field

from transformers import AutoConfig

from nanovllm.engine.cudagraph import CUDAGraphMode
from nanovllm.layers.gptq import GPTQConfig


@dataclass(slots=True)
class Config:
    model: str
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    speculative_method: str = "none"
    num_speculative_tokens: int = 2
    mtp_model: str | None = None
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
    model_family: str = field(init=False, default="")
    enable_prefix_cache: bool = field(init=False, default=True)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kvcache_storage_dtype: str = field(init=False, default="")
    kvcache_block_bytes: int = field(init=False, default=0)
    kvcache_payload_bytes: int = field(init=False, default=0)
    kvcache_scale_bytes: int = field(init=False, default=0)
    mtp_kvcache_bytes: int = field(init=False, default=0)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        outer_config = AutoConfig.from_pretrained(self.model)
        self.model_family = str(getattr(outer_config, "model_type", ""))
        text_config = getattr(outer_config, "text_config", None)

        checkpoint_config = getattr(outer_config, "quantization_config", None)
        if checkpoint_config is None and text_config is not None:
            checkpoint_config = getattr(text_config, "quantization_config", None)
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

        if self.model_family == "qwen3_5":
            if text_config is None:
                raise ValueError("Qwen3.5/3.6 config is missing text_config")
            if self.tensor_parallel_size != 1:
                raise ValueError("Qwen3.5/3.6 text inference v1 only supports TP=1")
            self.hf_config = text_config
            self.enable_prefix_cache = False
        else:
            self.hf_config = outer_config

        self.speculative_method = str(self.speculative_method).lower()
        if self.speculative_method not in ("none", "mtp"):
            raise ValueError("speculative_method must be 'none' or 'mtp'")
        if self.speculative_method == "mtp":
            if self.model_family != "qwen3_5":
                raise ValueError("MTP v1 only supports Qwen3.5/3.6")
            if self.tensor_parallel_size != 1:
                raise ValueError("Qwen3.5/3.6 MTP v1 only supports TP=1")
            if self.num_speculative_tokens not in (1, 2, 3):
                raise ValueError(
                    "MTP v1 supports one, two, or three speculative tokens"
                )
            minimum_batch_tokens = 1 + self.num_speculative_tokens
            if self.max_num_batched_tokens < minimum_batch_tokens:
                raise ValueError(
                    "MTP requires max_num_batched_tokens at least "
                    f"{minimum_batch_tokens}"
                )
            if getattr(self.hf_config, "mtp_num_hidden_layers", 0) != 1:
                raise ValueError("MTP v1 requires exactly one checkpoint MTP layer")
            if getattr(self.hf_config, "mtp_use_dedicated_embeddings", False):
                raise ValueError("MTP v1 requires shared token embeddings")
            self.mtp_model = self.mtp_model or self.model
            if not os.path.isdir(self.mtp_model):
                raise ValueError(
                    f"MTP model path is not a directory: {self.mtp_model}"
                )

        self.kv_cache_dtype = str(self.kv_cache_dtype).lower()
        if self.kv_cache_dtype not in ("auto", "fp8_e4m3"):
            raise ValueError("kv_cache_dtype must be 'auto' or 'fp8_e4m3'")

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
