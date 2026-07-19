import pickle
from dataclasses import dataclass

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.capacity import plan_delta_state_capacity
from nanovllm.config import Config
from nanovllm.engine.cudagraph import (
    BatchDescriptor,
    CUDAGraphDispatcher,
    CUDAGraphMode,
    ExecutionMode,
    make_full_capture_sizes,
    make_piecewise_capture_sizes,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import greedy_accept
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.deltanet_chunk import (
    DELTA_CHUNK_MIN_TOKENS,
    DELTA_CHUNK_SIZE,
)
from nanovllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.layers.linear import quantize_fp8
from nanovllm.layers.fp8_attention import FP8_QUERY_TILE_SIZE
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.mtp_loader import load_mtp_model


@dataclass(slots=True)
class SpeculativeBatchOutput:
    token_ids: list[list[int]]
    accepted_counts: list[int]
    next_draft_token_ids: list[list[int] | None]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.cudagraph_mode = CUDAGraphMode.parse(config.cudagraph_mode)
        self.enforce_eager = self.cudagraph_mode is CUDAGraphMode.NONE
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        init_method = f"tcp://{config.master_host}:{config.master_port}"
        dist.init_process_group("nccl", init_method, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(config.device_ids[rank])
        if config.kv_cache_dtype == "fp8_e4m3":
            capability = torch.cuda.get_device_capability()
            head_dim = getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
            if capability < (8, 9):
                raise RuntimeError(
                    "FP8 KV cache v1 requires an SM89 or newer CUDA GPU; "
                    f"current capability is SM{capability[0]}{capability[1]}"
                )
            if hf_config.dtype != torch.bfloat16:
                raise RuntimeError("FP8 KV cache v1 requires BF16 activations")
            if head_dim not in (64, 128, 256):
                raise RuntimeError(
                    "FP8 KV cache v1 supports head_dim 64, 128, or 256; "
                    f"got {head_dim}"
                )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        if config.model_family == "qwen3_5":
            self.model = Qwen3_5ForConditionalGeneration(
                hf_config, config.gptq_config
            )
        else:
            self.model = Qwen3ForCausalLM(
                hf_config, config.gptq_config
            )
        self.mtp_model = None
        if config.speculative_method == "mtp":
            self.mtp_model = Qwen3_5MTP(hf_config)
        self.has_delta_state = hasattr(self.model, "create_delta_state")
        self.delta_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.delta_state_slab: tuple[torch.Tensor, torch.Tensor] | None = None
        self.delta_working_slab: tuple[torch.Tensor, torch.Tensor] | None = None
        self.delta_state_slots: dict[int, int] = {}
        self.free_delta_state_slots: list[int] = []
        self.delta_state_capacity: int | None = None
        load_model(self.model, config.model)
        if self.mtp_model is not None:
            load_mtp_model(self.mtp_model, config.mtp_model, hf_config)
        self.max_active_delta_states = 0
        if config.quantization == "fp8":
            quantize_fp8(self.model)
            torch.cuda.empty_cache()
        # Compile and replay with the same ambient default device used by
        # steady-state engine steps. All persistent GPU tensors below specify
        # their device explicitly.
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self.sampler = Sampler()
        max_full_batch_size = min(self.config.max_num_seqs, 512)
        max_piecewise_tokens = min(
            self.config.piecewise_max_tokens,
            self.config.max_num_batched_tokens,
        )
        self.full_capture_sizes = make_full_capture_sizes(max_full_batch_size)
        self.piecewise_capture_sizes = make_piecewise_capture_sizes(max_piecewise_tokens)
        self.cudagraph_dispatcher = CUDAGraphDispatcher(
            self.cudagraph_mode,
            self.full_capture_sizes,
            self.piecewise_capture_sizes,
        )
        self.last_execution_mode = ExecutionMode.EAGER.value
        self.last_speculative_stats = {
            "drafted": 0,
            "proposed": 0,
            "accepted": 0,
            "rejected": 0,
            "bonus": 0,
            "verification_rounds": 0,
            "accepted_position_1": 0,
            "accepted_position_2": 0,
            "accepted_position_3": 0,
        }
        self.piecewise_model = None
        if self.cudagraph_mode.uses_piecewise:
            try:
                self.piecewise_model = torch.compile(
                    self.model,
                    dynamic=True,
                    fullgraph=False,
                    mode="reduce-overhead",
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed to initialize Piecewise CUDA Graph compilation; "
                    "use cudagraph_mode=NONE or enforce_eager=True to disable it"
                ) from exc
        try:
            self.warmup_model()
            if self.cudagraph_mode.uses_piecewise:
                self.capture_piecewise_cudagraphs()
        except Exception as exc:
            if self.cudagraph_mode.uses_piecewise:
                raise RuntimeError(
                    "failed to compile or capture Piecewise CUDA Graphs; use "
                    "cudagraph_mode=NONE or enforce_eager=True to disable them"
                ) from exc
            raise
        self.allocate_kv_cache()
        self.warmup_fp8_kv_cache()
        if self.cudagraph_mode.uses_full:
            try:
                self.capture_cudagraph()
            except Exception as exc:
                raise RuntimeError(
                    "failed to capture Full CUDA Graphs; use "
                    "cudagraph_mode=NONE or enforce_eager=True to disable them"
                ) from exc
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        if self.piecewise_model is not None:
            del self.piecewise_model
        if self.mtp_model is not None:
            del self.mtp_model
            self.mtp_model = None
        self.delta_states.clear()
        self.delta_state_slots.clear()
        self.free_delta_state_slots.clear()
        self.delta_state_slab = None
        self.delta_working_slab = None
        for attribute in (
            "mtp_kv_cache",
            "kv_scale",
            "kv_cache",
            "sampler",
            "model",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def get_delta_state(
        self,
        seq_id: int,
        *,
        working: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.has_delta_state:
            return None

        if self.delta_state_slab is None:
            state = self.delta_states.get(seq_id)
            if state is None:
                state = self.model.create_delta_state(
                    device="cuda", dtype=self.config.hf_config.dtype
                )
                self.delta_states[seq_id] = state
                self.max_active_delta_states = max(
                    self.max_active_delta_states, len(self.delta_states)
                )
            return state

        slot = self.delta_state_slots.get(seq_id)
        if slot is None:
            if not self.free_delta_state_slots:
                raise RuntimeError(
                    "Qwen3.5/3.6 DeltaNet state capacity exhausted: "
                    f"{self.delta_state_capacity} active sequences"
                )
            slot = self.free_delta_state_slots.pop()
            conv_slab, recurrent_slab = self.delta_state_slab
            conv_slab[:, slot].zero_()
            recurrent_slab[:, slot].zero_()
            self.delta_state_slots[seq_id] = slot
            self.max_active_delta_states = max(
                self.max_active_delta_states, len(self.delta_state_slots)
            )

        if self.delta_state_capacity is None or slot >= self.delta_state_capacity:
            raise RuntimeError(
                f"invalid DeltaNet state slot {slot} for capacity "
                f"{self.delta_state_capacity}"
            )
        slab = self.delta_working_slab if working else self.delta_state_slab
        if slab is None:
            raise RuntimeError("DeltaNet state slab is not initialized")
        conv_slab, recurrent_slab = slab
        return conv_slab[:, slot], recurrent_slab[:, slot]

    def release_sequences(self, seq_ids):
        for seq_id in seq_ids:
            if self.delta_state_slab is None:
                self.delta_states.pop(seq_id, None)
                continue
            slot = self.delta_state_slots.pop(seq_id, None)
            if slot is not None:
                self.free_delta_state_slots.append(slot)


    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        if self.has_delta_state:
            seq_len = min(seq_len, 8)
            num_seqs = 1
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.temperature = 0.0
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        self.release_sequences(tuple(seq.seq_id for seq in seqs))
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        mtp_attention_modules = (
            [
                module
                for module in self.mtp_model.modules()
                if hasattr(module, "k_cache") and hasattr(module, "v_cache")
            ]
            if self.mtp_model is not None
            else []
        )
        num_attention_layers = len(attention_modules)
        use_fp8_kv = config.kv_cache_dtype == "fp8_e4m3"
        cache_dtype = torch.float8_e4m3fn if use_fp8_kv else hf_config.dtype
        config.kvcache_storage_dtype = (
            "fp8_e4m3"
            if use_fp8_kv
            else str(cache_dtype).removeprefix("torch.")
        )
        target_payload_bytes = (
            2
            * num_attention_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * cache_dtype.itemsize
        )
        scale_bytes = (
            2
            * num_attention_layers
            * self.block_size
            * num_kv_heads
            * torch.tensor([], dtype=torch.float16).element_size()
            if use_fp8_kv
            else 0
        )
        mtp_payload_bytes = (
            2
            * len(mtp_attention_modules)
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=hf_config.dtype).element_size()
        )
        payload_bytes = target_payload_bytes + mtp_payload_bytes
        block_bytes = payload_bytes + scale_bytes
        config.kvcache_payload_bytes = payload_bytes
        config.kvcache_scale_bytes = scale_bytes
        config.kvcache_block_bytes = block_bytes
        config.mtp_kvcache_bytes = mtp_payload_bytes
        available_bytes = int(
            total * config.gpu_memory_utilization - used - peak + current
        )
        reserved_delta_bytes = 0
        delta_state_bytes_per_sequence = 0
        guaranteed_kv_blocks_per_sequence = 0
        minimum_delta_kv_blocks = 0
        if self.has_delta_state:
            state_bytes = self.model.delta_state_bytes(hf_config.dtype)
            state_copies = 2 if self.mtp_model is not None else 1
            capacity_plan = plan_delta_state_capacity(
                available_bytes=available_bytes,
                state_bytes=state_bytes,
                state_copies=state_copies,
                block_bytes=block_bytes,
                block_size=self.block_size,
                max_model_len=config.max_model_len,
                max_num_seqs=config.max_num_seqs,
                speculative_tokens=(
                    config.num_speculative_tokens
                    if self.mtp_model is not None
                    else 0
                ),
            )
            if capacity_plan.capacity < 1:
                raise RuntimeError(
                    "insufficient GPU memory for DeltaNet state and the "
                    "configured per-sequence KV capacity"
                )
            self.delta_state_capacity = capacity_plan.capacity
            config.max_num_seqs = self.delta_state_capacity
            delta_state_bytes_per_sequence = (
                capacity_plan.state_bytes_per_sequence
            )
            guaranteed_kv_blocks_per_sequence = (
                capacity_plan.kv_blocks_per_sequence
            )
            minimum_delta_kv_blocks = capacity_plan.minimum_kv_blocks
            reserved_delta_bytes = self.delta_state_capacity * (
                delta_state_bytes_per_sequence
            )
            self.delta_states.clear()
            self.delta_state_slab = self.model.create_delta_state_slab(
                self.delta_state_capacity,
                device="cuda",
                dtype=hf_config.dtype,
            )
            self.delta_working_slab = (
                self.model.create_delta_state_slab(
                    self.delta_state_capacity,
                    device="cuda",
                    dtype=hf_config.dtype,
                )
                if self.mtp_model is not None
                else None
            )
            self.delta_state_slots.clear()
            self.free_delta_state_slots = list(
                range(self.delta_state_capacity - 1, -1, -1)
            )
        cache_bytes = available_bytes - reserved_delta_bytes
        config.num_kvcache_blocks = cache_bytes // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError("insufficient GPU memory for one KV cache block")
        if config.num_kvcache_blocks < minimum_delta_kv_blocks:
            raise RuntimeError(
                "KV cache allocation fell below the per-sequence capacity "
                f"guarantee: {config.num_kvcache_blocks} blocks available, "
                f"{minimum_delta_kv_blocks} required"
            )
        self.kv_cache = torch.empty(
            2,
            num_attention_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=cache_dtype,
            device="cuda",
        )
        self.kv_scale = (
            torch.empty(
                2,
                num_attention_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                dtype=torch.float16,
                device="cuda",
            )
            if use_fp8_kv
            else None
        )
        for layer_id, module in enumerate(attention_modules):
            module.k_cache.tensor = self.kv_cache[0, layer_id]
            module.v_cache.tensor = self.kv_cache[1, layer_id]
            if self.kv_scale is not None:
                module.k_scale.tensor = self.kv_scale[0, layer_id]
                module.v_scale.tensor = self.kv_scale[1, layer_id]

        self.mtp_kv_cache = None
        if mtp_attention_modules:
            self.mtp_kv_cache = torch.empty(
                2,
                len(mtp_attention_modules),
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=hf_config.dtype,
                device="cuda",
            )
            for layer_id, module in enumerate(mtp_attention_modules):
                module.k_cache.tensor = self.mtp_kv_cache[0, layer_id]
                module.v_cache.tensor = self.mtp_kv_cache[1, layer_id]

        if self.rank == 0:
            scale_mode = (
                "per_token_per_kv_head" if use_fp8_kv else "none"
            )
            print(
                "KV cache: "
                f"requested_dtype={config.kv_cache_dtype}, "
                f"dtype={config.kvcache_storage_dtype}, "
                f"scale_mode={scale_mode}, "
                f"payload_bytes_per_block={payload_bytes}, "
                f"scale_bytes_per_block={scale_bytes}, "
                f"mtp_bytes_per_block={mtp_payload_bytes}, "
                f"blocks={config.num_kvcache_blocks}, "
                f"delta_state_slots={self.delta_state_capacity or 0}, "
                f"delta_state_bytes_per_sequence="
                f"{delta_state_bytes_per_sequence}, "
                f"guaranteed_kv_blocks_per_sequence="
                f"{guaranteed_kv_blocks_per_sequence}, "
                f"token_capacity={config.num_kvcache_blocks * self.block_size}"
            )

    def warmup_fp8_kv_cache(self):
        if self.config.kv_cache_dtype != "fp8_e4m3":
            return
        num_tokens = min(FP8_QUERY_TILE_SIZE, self.config.max_model_len - 1)
        if num_tokens <= 0:
            raise RuntimeError("FP8 KV cache requires max_model_len at least 2")
        seq = Sequence([0] * num_tokens)
        seq.block_table = [0]
        seq.num_scheduled_tokens = num_tokens
        prefill = BatchDescriptor(
            num_tokens=num_tokens,
            num_padded_tokens=num_tokens,
            num_seqs=1,
            uniform_decode=False,
            execution_mode=ExecutionMode.EAGER,
        )
        input_ids, positions = self.prepare_inputs([seq], prefill)
        try:
            self.model(input_ids, positions)
        finally:
            reset_context()

        seq.num_cached_tokens = num_tokens
        seq.append_token(0)
        seq.num_scheduled_tokens = 1
        decode = BatchDescriptor(
            num_tokens=1,
            num_padded_tokens=1,
            num_seqs=1,
            uniform_decode=True,
            execution_mode=ExecutionMode.EAGER,
        )
        input_ids, positions = self.prepare_inputs([seq], decode)
        try:
            self.model(input_ids, positions)
        finally:
            reset_context()
            self.release_sequences((seq.seq_id,))
        torch.cuda.synchronize()

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_inputs(
        self,
        seqs: list[Sequence],
        descriptor: BatchDescriptor,
        *,
        use_working_delta: bool = False,
    ):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        sequence_slices = []
        delta_chunk_indices_host = []
        delta_cu_chunks_host = [0]
        delta_chunk_sequences_host = []
        delta_recurrent_sequences_host = []
        slot_mapping = []
        logits_indices = []
        context_lens = []
        query_tile_seq_ids = []
        use_fp8_kv = self.config.kv_cache_dtype == "fp8_e4m3"
        query_tile_starts = []
        query_tile_lens = []
        query_tile_positions = []
        has_block_tables = [bool(seq.block_table) for seq in seqs]
        if any(has_block_tables) and not all(has_block_tables):
            raise ValueError("all sequences in a model batch must either use KV cache or be warmup sequences")
        use_kv_cache = all(has_block_tables)
        for seq in seqs:
            seq_id = getattr(seq, "seq_id", "unknown")
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            if seqlen_q <= 0:
                raise ValueError(f"sequence {seq_id} has no scheduled tokens")
            end = start + seqlen_q
            seqlen_k = end
            if seqlen_q == 1 and not seq.draft_token_ids:
                # Decode sequences only serialize last_token to TP workers.
                input_ids.append(seq.last_token)
            else:
                input_ids.extend(seq.scheduled_token_ids())
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            sequence_slices.append((cu_seqlens_q[-2], cu_seqlens_q[-1]))
            sequence_index = len(sequence_slices) - 1
            if use_fp8_kv:
                for tile_offset in range(0, seqlen_q, FP8_QUERY_TILE_SIZE):
                    query_tile_seq_ids.append(sequence_index)
                    query_tile_starts.append(cu_seqlens_q[-2] + tile_offset)
                    query_tile_lens.append(
                        min(FP8_QUERY_TILE_SIZE, seqlen_q - tile_offset)
                    )
                    query_tile_positions.append(start + tile_offset)
            if seqlen_q >= DELTA_CHUNK_MIN_TOKENS:
                delta_chunk_sequences_host.append(sequence_index)
                num_delta_chunks = (
                    seqlen_q + DELTA_CHUNK_SIZE - 1
                ) // DELTA_CHUNK_SIZE
                delta_chunk_indices_host.extend(
                    (sequence_index, chunk) for chunk in range(num_delta_chunks)
                )
                delta_cu_chunks_host.append(
                    delta_cu_chunks_host[-1] + num_delta_chunks
                )
            else:
                delta_recurrent_sequences_host.append(sequence_index)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(seqlen_k)
            if seq.will_sample:
                if seq.draft_token_ids:
                    logits_indices.extend(
                        range(cu_seqlens_q[-2], cu_seqlens_q[-1])
                    )
                else:
                    logits_indices.append(cu_seqlens_q[-1] - 1)
            if not use_kv_cache:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        num_actual_tokens = len(input_ids)
        if num_actual_tokens != descriptor.num_tokens:
            raise ValueError(
                f"dispatcher expected {descriptor.num_tokens} tokens, prepared {num_actual_tokens}"
            )
        model_num_tokens = (
            descriptor.num_padded_tokens
            if descriptor.execution_mode is ExecutionMode.PIECEWISE
            else num_actual_tokens
        )
        num_padding_tokens = model_num_tokens - num_actual_tokens
        if num_padding_tokens < 0:
            raise ValueError("CUDA Graph bucket is smaller than the real token batch")
        input_ids.extend([0] * num_padding_tokens)
        positions.extend([0] * num_padding_tokens)
        block_tables = self.prepare_block_tables(seqs) if use_kv_cache else None
        is_uniform_decode = use_kv_cache and all(
            seq.num_scheduled_tokens == 1 and not seq.is_prefill for seq in seqs
        )
        if is_uniform_decode != descriptor.uniform_decode:
            raise ValueError("dispatcher and prepared attention metadata disagree on batch shape")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        delta_states = ()
        delta_recurrent_slab = None
        delta_state_slots = None
        delta_chunk_indices = None
        delta_cu_chunks = None
        delta_chunk_sequences = None
        delta_recurrent_sequences = None
        if self.has_delta_state:
            delta_states = tuple(
                self.get_delta_state(
                    seq.seq_id, working=use_working_delta
                )
                for seq in seqs
            )
            selected_slab = (
                self.delta_working_slab
                if use_working_delta
                else self.delta_state_slab
            )
            if selected_slab is not None:
                delta_recurrent_slab = selected_slab[1]
                delta_state_slots = torch.tensor(
                    [self.delta_state_slots[seq.seq_id] for seq in seqs],
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                delta_chunk_indices = torch.tensor(
                    delta_chunk_indices_host,
                    dtype=torch.int32,
                    pin_memory=True,
                ).reshape(-1, 2).cuda(non_blocking=True)
                delta_cu_chunks = torch.tensor(
                    delta_cu_chunks_host,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                delta_chunk_sequences = torch.tensor(
                    delta_chunk_sequences_host,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                delta_recurrent_sequences = torch.tensor(
                    delta_recurrent_sequences_host,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if use_fp8_kv:
            query_tile_seq_ids = torch.tensor(
                query_tile_seq_ids, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            query_tile_starts = torch.tensor(
                query_tile_starts, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            query_tile_lens = torch.tensor(
                query_tile_lens, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            query_tile_positions = torch.tensor(
                query_tile_positions, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        else:
            query_tile_seq_ids = query_tile_starts = None
            query_tile_lens = query_tile_positions = None
        logits_indices = torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_context(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            logits_indices=logits_indices,
            context_lens=context_lens,
            query_tile_seq_ids=query_tile_seq_ids,
            query_tile_starts=query_tile_starts,
            query_tile_lens=query_tile_lens,
            query_tile_positions=query_tile_positions,
            use_kv_cache=use_kv_cache,
            sequence_slices=tuple(sequence_slices),
            delta_states=delta_states,
            delta_recurrent_slab=delta_recurrent_slab,
            delta_state_slots=delta_state_slots,
            delta_chunk_indices=delta_chunk_indices,
            delta_cu_chunks=delta_cu_chunks,
            delta_chunk_sequences=delta_chunk_sequences,
            delta_recurrent_sequences=delta_recurrent_sequences,
            is_uniform_decode=is_uniform_decode,
            num_actual_tokens=num_actual_tokens,
            num_padded_tokens=model_num_tokens,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _delta_slot_indices(self, seqs: list[Sequence]) -> torch.Tensor:
        if self.delta_state_slab is None:
            return torch.empty(0, dtype=torch.long, device="cuda")
        for seq in seqs:
            self.get_delta_state(seq.seq_id)
        return torch.tensor(
            [self.delta_state_slots[seq.seq_id] for seq in seqs],
            dtype=torch.long,
            device="cuda",
        )

    def _copy_delta_bank(
        self,
        seqs: list[Sequence],
        source: tuple[torch.Tensor, torch.Tensor],
        destination: tuple[torch.Tensor, torch.Tensor],
    ):
        slots = self._delta_slot_indices(seqs)
        if not slots.numel():
            return
        for source_tensor, destination_tensor in zip(source, destination):
            destination_tensor.index_copy_(
                1, slots, source_tensor.index_select(1, slots)
            )

    def _prepare_working_delta(self, seqs: list[Sequence]):
        if self.delta_state_slab is None or self.delta_working_slab is None:
            return
        self._copy_delta_bank(
            seqs, self.delta_state_slab, self.delta_working_slab
        )

    def _commit_working_delta(self, seqs: list[Sequence]):
        if (
            not seqs
            or self.delta_state_slab is None
            or self.delta_working_slab is None
        ):
            return
        self._copy_delta_bank(
            seqs, self.delta_working_slab, self.delta_state_slab
        )

    @torch.inference_mode()
    def _replay_rejected_prefixes(
        self,
        rejected: list[tuple[Sequence, int]],
    ):
        if not rejected:
            return
        saved = [
            (seq, seq.num_scheduled_tokens)
            for seq, _ in rejected
        ]
        replay_lengths = [1 + accepted for _, accepted in rejected]
        try:
            for (seq, accepted), replay_length in zip(
                rejected, replay_lengths
            ):
                if not 0 <= accepted < len(seq.draft_token_ids):
                    raise ValueError("invalid partial draft acceptance for replay")
                seq.num_scheduled_tokens = replay_length
            uniform_decode = all(length == 1 for length in replay_lengths)
            descriptor = BatchDescriptor(
                num_tokens=sum(replay_lengths),
                num_padded_tokens=sum(replay_lengths),
                num_seqs=len(rejected),
                uniform_decode=uniform_decode,
                execution_mode=ExecutionMode.EAGER,
            )
            seqs = [seq for seq, _ in rejected]
            input_ids, positions = self.prepare_inputs(seqs, descriptor)
            try:
                self.run_model(input_ids, positions, descriptor)
            finally:
                reset_context()
        finally:
            for seq, scheduled in saved:
                seq.num_scheduled_tokens = scheduled

    def _mtp_slot_mapping(
        self,
        seq: Sequence,
        start: int,
        end: int,
    ) -> list[int]:
        slots = []
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for block_index in range(start_block, end_block):
            slot_start = seq.block_table[block_index] * self.block_size
            if block_index == start_block:
                slot_start += start % self.block_size
            slot_end = (
                seq.block_table[block_index] * self.block_size
                + self.block_size
            )
            if block_index == end_block - 1:
                slot_end = (
                    seq.block_table[block_index] * self.block_size
                    + end
                    - block_index * self.block_size
                )
            slots.extend(range(slot_start, slot_end))
        return slots

    @torch.inference_mode()
    def _run_mtp_proposal(
        self,
        seqs: list[Sequence],
        target_hidden_states: torch.Tensor,
        token_groups: list[list[int]],
        accepted_counts: list[int],
    ) -> list[list[int] | None]:
        sampled_seqs = [seq for seq in seqs if seq.will_sample]
        if self.mtp_model is None:
            return [None] * len(sampled_seqs)

        hidden_parts = []
        next_token_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        logits_indices = []
        sampled_next_positions = []
        offset = 0
        group_index = 0
        use_kv_cache = all(bool(seq.block_table) for seq in seqs)
        if any(bool(seq.block_table) for seq in seqs) != use_kv_cache:
            raise ValueError("MTP batch mixes cached and uncached sequences")

        for seq in seqs:
            scheduled = seq.num_scheduled_tokens
            sequence_hidden = target_hidden_states[offset : offset + scheduled]
            offset += scheduled
            if seq.will_sample:
                outputs = token_groups[group_index]
                accepted = accepted_counts[group_index]
                valid_inputs = (
                    1 + accepted if seq.draft_token_ids else scheduled
                )
                target_inputs = seq.scheduled_token_ids()[:valid_inputs]
                shifted_ids = target_inputs[1:] + [outputs[-1]]
                group_index += 1
            else:
                valid_inputs = scheduled
                start = seq.num_cached_tokens
                shifted_ids = seq.token_ids[
                    start + 1 : start + valid_inputs + 1
                ]
            if len(shifted_ids) != valid_inputs:
                raise ValueError(
                    f"cannot build shifted MTP inputs for sequence {seq.seq_id}"
                )

            start = seq.num_cached_tokens
            end = start + valid_inputs
            hidden_parts.append(sequence_hidden[:valid_inputs])
            next_token_ids.extend(shifted_ids)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + valid_inputs)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            context_lens.append(end)
            if seq.will_sample:
                logits_indices.append(cu_seqlens_q[-1] - 1)
                sampled_next_positions.append(end)
            if use_kv_cache:
                slot_mapping.extend(self._mtp_slot_mapping(seq, start, end))

        num_tokens = len(next_token_ids)
        if num_tokens == 0:
            return []
        hidden_states = torch.cat(hidden_parts, dim=0)
        input_ids = torch.tensor(
            next_token_ids, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        positions_tensor = torch.tensor(
            positions, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        cu_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slots = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        logits_indices_tensor = torch.tensor(
            logits_indices, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = (
            self.prepare_block_tables(seqs) if use_kv_cache else None
        )
        uniform_decode = use_kv_cache and all(
            query_end - query_start == 1
            for query_start, query_end in zip(
                cu_seqlens_q[:-1], cu_seqlens_q[1:]
            )
        )
        set_context(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(
                query_end - query_start
                for query_start, query_end in zip(
                    cu_seqlens_q[:-1], cu_seqlens_q[1:]
                )
            ),
            max_seqlen_k=max(context_lens),
            slot_mapping=slots,
            block_tables=block_tables,
            logits_indices=logits_indices_tensor,
            context_lens=context_lens_tensor,
            use_kv_cache=use_kv_cache,
            is_uniform_decode=uniform_decode,
            num_actual_tokens=num_tokens,
            num_padded_tokens=num_tokens,
        )
        try:
            embeddings = (
                self.model.model.language_model.embed_tokens(input_ids)
            )
            mtp_hidden = self.mtp_model(
                positions_tensor, hidden_states, embeddings
            )
            if not logits_indices:
                return []
            sampled_hidden = mtp_hidden.index_select(
                0, logits_indices_tensor
            )
            logits = self.model.compute_logits(mtp_hidden)
            proposal_ids = logits.argmax(dim=-1)
        finally:
            reset_context()

        draft_chains = [
            [token_id] for token_id in proposal_ids.tolist()
        ]
        for _ in range(1, self.config.num_speculative_tokens):
            batch_size = len(sampled_seqs)
            recursive_positions = torch.tensor(
                sampled_next_positions,
                dtype=torch.long,
                pin_memory=True,
            ).cuda(non_blocking=True)
            recursive_input_ids = proposal_ids.to(dtype=torch.long)
            recursive_cu_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device="cuda"
            )
            if use_kv_cache:
                recursive_cu_k_host = [0]
                recursive_slots = []
                recursive_context_lens = []
                for seq, position in zip(
                    sampled_seqs, sampled_next_positions
                ):
                    recursive_cu_k_host.append(
                        recursive_cu_k_host[-1] + position + 1
                    )
                    recursive_context_lens.append(position + 1)
                    recursive_slots.extend(
                        self._mtp_slot_mapping(
                            seq, position, position + 1
                        )
                    )
                recursive_cu_k = torch.tensor(
                    recursive_cu_k_host,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                recursive_slot_mapping = torch.tensor(
                    recursive_slots,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                recursive_context = torch.tensor(
                    recursive_context_lens,
                    dtype=torch.int32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                recursive_blocks = self.prepare_block_tables(sampled_seqs)
            else:
                recursive_cu_k = recursive_cu_q
                recursive_slot_mapping = torch.empty(
                    0, dtype=torch.int32, device="cuda"
                )
                recursive_context = torch.ones(
                    batch_size, dtype=torch.int32, device="cuda"
                )
                recursive_blocks = None
            recursive_logits_indices = torch.arange(
                batch_size, dtype=torch.long, device="cuda"
            )
            set_context(
                cu_seqlens_q=recursive_cu_q,
                cu_seqlens_k=recursive_cu_k,
                max_seqlen_q=1,
                max_seqlen_k=(
                    max(sampled_next_positions) + 1
                    if use_kv_cache
                    else 1
                ),
                slot_mapping=recursive_slot_mapping,
                block_tables=recursive_blocks,
                logits_indices=recursive_logits_indices,
                context_lens=recursive_context,
                use_kv_cache=use_kv_cache,
                is_uniform_decode=use_kv_cache,
                num_actual_tokens=batch_size,
                num_padded_tokens=batch_size,
            )
            try:
                recursive_embeddings = (
                    self.model.model.language_model.embed_tokens(
                        recursive_input_ids
                    )
                )
                sampled_hidden = self.mtp_model(
                    recursive_positions,
                    sampled_hidden,
                    recursive_embeddings,
                )
                recursive_logits = self.model.compute_logits(sampled_hidden)
                proposal_ids = recursive_logits.argmax(dim=-1)
            finally:
                reset_context()
            for chain, token_id in zip(
                draft_chains, proposal_ids.tolist()
            ):
                chain.append(token_id)
            sampled_next_positions = [
                position + 1 for position in sampled_next_positions
            ]

        return draft_chains

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        descriptor: BatchDescriptor,
    ):
        if descriptor.execution_mode is ExecutionMode.EAGER:
            return self.model(input_ids, positions)
        if descriptor.execution_mode is ExecutionMode.PIECEWISE:
            if self.piecewise_model is None:
                raise RuntimeError("Piecewise CUDA Graph model is not initialized")
            torch.compiler.cudagraph_mark_step_begin()
            return self.piecewise_model(input_ids, positions)
        if descriptor.execution_mode is ExecutionMode.FULL:
            bs = descriptor.num_tokens
            context = get_context()
            graph = self.graphs[descriptor.num_padded_tokens]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self.config.kv_cache_dtype == "fp8_e4m3":
                graph_vars["query_tile_seq_ids"].zero_()
                graph_vars["query_tile_starts"].zero_()
                graph_vars["query_tile_lens"].zero_()
                graph_vars["query_tile_positions"].zero_()
                num_tiles = context.query_tile_seq_ids.numel()
                graph_vars["query_tile_seq_ids"][:num_tiles] = context.query_tile_seq_ids
                graph_vars["query_tile_starts"][:num_tiles] = context.query_tile_starts
                graph_vars["query_tile_lens"][:num_tiles] = context.query_tile_lens
                graph_vars["query_tile_positions"][:num_tiles] = context.query_tile_positions
            graph.replay()
            return graph_vars["outputs"][:bs]
        raise ValueError(f"unsupported execution mode: {descriptor.execution_mode}")

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int] | SpeculativeBatchOutput:
        self.last_speculative_stats = {
            "drafted": 0,
            "proposed": 0,
            "accepted": 0,
            "rejected": 0,
            "bonus": 0,
            "verification_rounds": 0,
            "accepted_position_1": 0,
            "accepted_position_2": 0,
            "accepted_position_3": 0,
        }
        sampled_seqs = [seq for seq in seqs if seq.will_sample]
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        is_uniform_decode = all(
            seq.num_scheduled_tokens == 1 and not seq.is_prefill for seq in seqs
        )
        descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            len(seqs),
            is_uniform_decode,
        )

        use_speculative = self.mtp_model is not None
        if use_speculative and any(seq.temperature != 0 for seq in sampled_seqs):
            raise ValueError(
                "MTP v1 currently supports greedy decoding only "
                "(temperature=0)"
            )
        has_verification = use_speculative and any(
            seq.is_speculative for seq in sampled_seqs
        )
        if has_verification:
            self._prepare_working_delta(seqs)

        input_ids, positions = self.prepare_inputs(
            seqs,
            descriptor,
            use_working_delta=has_verification,
        )
        try:
            hidden_states = self.run_model(input_ids, positions, descriptor)
            self.last_execution_mode = descriptor.execution_mode.value
            if not sampled_seqs:
                if use_speculative:
                    self._run_mtp_proposal(seqs, hidden_states, [], [])
                return [] if self.rank == 0 else None

            logits = self.model.compute_logits(hidden_states)
            if not use_speculative:
                temperatures = (
                    self.prepare_sample(sampled_seqs)
                    if self.rank == 0
                    else None
                )
                return (
                    self.sampler(logits, temperatures).tolist()
                    if self.rank == 0
                    else None
                )

            token_groups: list[list[int]] = []
            accepted_counts: list[int] = []
            rejected_seqs: list[tuple[Sequence, int]] = []
            committed_working_seqs: list[Sequence] = [
                seq for seq in seqs if not seq.will_sample
            ]
            logit_offset = 0
            for seq in sampled_seqs:
                if seq.is_speculative:
                    num_drafts = len(seq.draft_token_ids)
                    if num_drafts != self.config.num_speculative_tokens:
                        raise ValueError(
                            "MTP draft chain length does not match "
                            "num_speculative_tokens"
                        )
                    num_verification_logits = num_drafts + 1
                    verification_logits = logits[
                        logit_offset :
                        logit_offset + num_verification_logits
                    ]
                    logit_offset += num_verification_logits
                    target_tokens = (
                        verification_logits.argmax(dim=-1).tolist()
                    )
                    outputs, accepted = greedy_accept(
                        target_tokens, seq.draft_token_ids
                    )
                    token_groups.append(outputs)
                    accepted_counts.append(accepted)
                    if accepted == num_drafts:
                        committed_working_seqs.append(seq)
                    else:
                        rejected_seqs.append((seq, accepted))
                else:
                    token_groups.append(
                        [int(logits[logit_offset].argmax().item())]
                    )
                    accepted_counts.append(0)
                    logit_offset += 1
                    if has_verification:
                        committed_working_seqs.append(seq)
            if logit_offset != logits.size(0):
                raise ValueError("target verification logits were not fully consumed")

            if has_verification:
                self._commit_working_delta(committed_working_seqs)
                self._replay_rejected_prefixes(rejected_seqs)

            proposals = self._run_mtp_proposal(
                seqs,
                hidden_states,
                token_groups,
                accepted_counts,
            )
            if len(proposals) != len(sampled_seqs):
                raise ValueError("MTP proposal count does not match sampled requests")
            proposed = sum(
                len(seq.draft_token_ids) for seq in sampled_seqs
            )
            accepted = sum(accepted_counts)
            self.last_speculative_stats = {
                "drafted": sum(
                    len(chain) for chain in proposals if chain is not None
                ),
                "proposed": proposed,
                "accepted": accepted,
                "rejected": proposed - accepted,
                "bonus": sum(
                    int(
                        seq.is_speculative
                        and accepted_count == len(seq.draft_token_ids)
                    )
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                "verification_rounds": sum(
                    int(seq.is_speculative) for seq in sampled_seqs
                ),
                "accepted_position_1": sum(
                    int(seq.is_speculative and accepted_count >= 1)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                "accepted_position_2": sum(
                    int(seq.is_speculative and accepted_count >= 2)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                "accepted_position_3": sum(
                    int(seq.is_speculative and accepted_count >= 3)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
            }
            return SpeculativeBatchOutput(
                token_ids=token_groups,
                accepted_counts=accepted_counts,
                next_draft_token_ids=proposals,
            )
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_piecewise_cudagraphs(self):
        if self.piecewise_model is None:
            return
        for size in reversed(self.piecewise_capture_sizes):
            input_ids = torch.zeros(size, dtype=torch.int64, device="cuda")
            positions = torch.arange(size, dtype=torch.int64, device="cuda")
            cu_seqlens = torch.tensor([0, size], dtype=torch.int32, device="cuda")
            set_context(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=size,
                max_seqlen_k=size,
                num_actual_tokens=size,
                num_padded_tokens=size,
            )
            try:
                for _ in range(2):
                    torch.compiler.cudagraph_mark_step_begin()
                    outputs = self.piecewise_model(input_ids, positions)
                    del outputs
                torch.cuda.synchronize()
            finally:
                reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        query_tile_seq_ids = torch.arange(max_bs, dtype=torch.int32, device="cuda")
        query_tile_starts = torch.arange(max_bs, dtype=torch.int32, device="cuda")
        query_tile_lens = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        query_tile_positions = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
            device="cuda",
        )
        outputs = torch.zeros(
            max_bs,
            hf_config.hidden_size,
            dtype=hf_config.dtype,
            device="cuda",
        )
        self.graph_bs = self.full_capture_sizes
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                query_tile_seq_ids=query_tile_seq_ids[:bs],
                query_tile_starts=query_tile_starts[:bs],
                query_tile_lens=query_tile_lens[:bs],
                query_tile_positions=query_tile_positions[:bs],
                use_kv_cache=True,
                is_uniform_decode=True,
                num_actual_tokens=bs,
                num_padded_tokens=bs,
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            query_tile_seq_ids=query_tile_seq_ids,
            query_tile_starts=query_tile_starts,
            query_tile_lens=query_tile_lens,
            query_tile_positions=query_tile_positions,
            outputs=outputs,
        )