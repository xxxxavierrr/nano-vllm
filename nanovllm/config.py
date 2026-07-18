import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.engine.cudagraph import CUDAGraphMode


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
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.quantization in (None, "fp8"), "quantization must be None or 'fp8'"
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
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
