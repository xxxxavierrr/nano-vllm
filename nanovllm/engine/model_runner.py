import pickle
from dataclasses import dataclass

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.capacity import plan_delta_state_capacity
from nanovllm.engine.batch import (
    AttentionMetadata,
    ExecutionSignature,
    GDNMetadata,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.config import Config
from nanovllm.engine.cudagraph import (
    BatchDescriptor,
    CUDAGraphDispatcher,
    CUDAGraphMode,
    ExecutionMode,
    infer_piecewise_capture_limit,
    make_full_capture_sizes,
    make_piecewise_capture_sizes,
)
from nanovllm.engine.hybrid_state import HybridStateManager
from nanovllm.engine.kv_capacity import make_kv_cache_layout
from nanovllm.engine.metrics import (
    RunnerStepMetrics,
    RunnerStepOutput,
    SpeculativeStepMetrics,
)
from nanovllm.engine.mtp_proposer import MTPProposer
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import (
    GreedyAcceptance,
    RejectionSamplingAcceptance,
)
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
from nanovllm.utils.context import forward_context
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
        self.hybrid_state = HybridStateManager(self.model, hf_config.dtype)
        self.has_delta_state = self.hybrid_state.enabled
        load_model(self.model, config.model)
        if self.mtp_model is not None:
            load_mtp_model(self.mtp_model, config.mtp_model, hf_config)
        self.speculator = (
            MTPProposer(
                self.model,
                self.mtp_model,
                block_size=self.block_size,
                num_steps=config.num_speculative_tokens,
            )
            if self.mtp_model is not None
            else None
        )
        self.acceptance_policy = GreedyAcceptance()
        self.rejection_sampling_policy = RejectionSamplingAcceptance()
        self.draft_logits: dict[int, torch.Tensor] = {}
        self.request_generators: dict[int, torch.Generator] = {}
        if config.quantization == "fp8":
            quantize_fp8(self.model)
            torch.cuda.empty_cache()
        # Compile and replay with the same ambient default device used by
        # steady-state engine steps. All persistent GPU tensors below specify
        # their device explicitly.
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self.sampler = Sampler()
        # DeltaNet Full Graph replay is currently validated only for a single
        # sequence. Larger uniform batches use Piecewise until the packed
        # recurrent-state batch path is proven numerically equivalent.
        max_full_batch_size = min(
            self.config.max_num_seqs,
            1 if self.has_delta_state else 512,
        )
        max_piecewise_tokens = infer_piecewise_capture_limit(
            requested_max_tokens=self.config.piecewise_max_tokens,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            speculative_tokens=(
                self.config.num_speculative_tokens
                if self.mtp_model is not None
                else 0
            ),
        )
        self.full_capture_sizes = make_full_capture_sizes(max_full_batch_size)
        self.full_query_lengths = [1]
        if self.mtp_model is not None:
            self.full_query_lengths.append(
                1 + self.config.num_speculative_tokens
            )
        self.full_query_lengths = sorted(set(self.full_query_lengths))
        self.piecewise_capture_sizes = make_piecewise_capture_sizes(max_piecewise_tokens)
        self.cudagraph_dispatcher = CUDAGraphDispatcher(
            self.cudagraph_mode,
            self.full_capture_sizes,
            self.piecewise_capture_sizes,
            self.full_query_lengths,
        )
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
            self.warmup_deltanet_chunk()
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
        self.speculator = None
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
        self.hybrid_state.close()
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
        return self.hybrid_state.get(seq_id, working=working)

    def release_sequences(self, seq_ids):
        self.hybrid_state.release(seq_ids)
        for seq_id in seq_ids:
            self.draft_logits.pop(seq_id, None)
            self.request_generators.pop(seq_id, None)

    def _generator_for(self, seq: Sequence) -> torch.Generator:
        generator = self.request_generators.get(seq.seq_id)
        if generator is None:
            generator = torch.Generator(
                device=f"cuda:{self.config.device_ids[self.rank]}"
            )
            seed = (
                seq.seed
                if seq.seed is not None
                else (torch.initial_seed() + seq.seq_id) % (2**63 - 1)
            )
            generator.manual_seed(seed)
            self.request_generators[seq.seq_id] = generator
        return generator

    @property
    def delta_state_capacity(self) -> int | None:
        return self.hybrid_state.capacity

    @property
    def max_active_delta_states(self) -> int:
        return self.hybrid_state.max_active

    @property
    def delta_state_slab(self):
        return self.hybrid_state.committed

    @property
    def delta_working_slab(self):
        return self.hybrid_state.working

    @property
    def delta_state_slots(self):
        return self.hybrid_state.slots

    @property
    def free_delta_state_slots(self):
        return self.hybrid_state.free_slots


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

    def warmup_deltanet_chunk(self):
        """Compile/autotune the production chunk path before cache sizing."""
        if not self.has_delta_state:
            return
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        if seq_len < DELTA_CHUNK_MIN_TOKENS:
            return
        seq = Sequence([0] * DELTA_CHUNK_MIN_TOKENS)
        seq.temperature = 0.0
        seq.num_scheduled_tokens = DELTA_CHUNK_MIN_TOKENS
        descriptor = BatchDescriptor(
            num_tokens=DELTA_CHUNK_MIN_TOKENS,
            num_padded_tokens=DELTA_CHUNK_MIN_TOKENS,
            num_seqs=1,
            uniform_query_len=None,
            execution_mode=ExecutionMode.EAGER,
        )
        prepared = self.prepare_inputs([seq], descriptor)
        try:
            with forward_context(prepared):
                self.model(prepared.input_ids, prepared.positions)
            torch.cuda.synchronize()
        finally:
            self.release_sequences((seq.seq_id,))
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # Piecewise CUDAGraph capture can leave several GiB in the ordinary
        # caching allocator. Live graph-private pools survive empty_cache(),
        # while releasable compilation/autotune blocks must not reduce the
        # subsequent DeltaNet state and KV capacity calculation.
        torch.cuda.empty_cache()
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
        layout = make_kv_cache_layout(
            kv_cache_dtype=config.kv_cache_dtype,
            target_layers=num_attention_layers,
            mtp_layers=len(mtp_attention_modules),
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            activation_bytes=torch.tensor([], dtype=hf_config.dtype).element_size(),
            native_dtype=str(hf_config.dtype).removeprefix("torch."),
        )
        mtp_payload_bytes = layout.mtp_payload_bytes_per_block
        scale_bytes = layout.scale_bytes_per_block
        payload_bytes = (
            layout.target_payload_bytes_per_block + mtp_payload_bytes
        )
        block_bytes = layout.total_bytes_per_block
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
            branch_slots_per_sequence = (
                1 + config.num_speculative_tokens
                if self.mtp_model is not None
                else 0
            )
            state_copies = 1 + branch_slots_per_sequence
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
            config.max_num_seqs = capacity_plan.capacity
            delta_state_bytes_per_sequence = (
                capacity_plan.state_bytes_per_sequence
            )
            guaranteed_kv_blocks_per_sequence = (
                capacity_plan.kv_blocks_per_sequence
            )
            minimum_delta_kv_blocks = capacity_plan.minimum_kv_blocks
            reserved_delta_bytes = capacity_plan.capacity * (
                delta_state_bytes_per_sequence
            )
            self.hybrid_state.allocate(
                capacity_plan.capacity,
                device="cuda",
                branch_slots_per_sequence=branch_slots_per_sequence,
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
            scale_mode = layout.scale_mode
            print(
                "KV cache: "
                f"requested_dtype={config.kv_cache_dtype}, "
                f"dtype={config.kvcache_storage_dtype}, "
                f"scale_mode={scale_mode}, "
                f"payload_bytes_per_block={payload_bytes}, "
                f"scale_bytes_per_block={scale_bytes}, "
                f"mtp_bytes_per_block={mtp_payload_bytes}, "
                f"blocks={config.num_kvcache_blocks}, "
                f"delta_request_capacity={self.delta_state_capacity or 0}, "
                f"delta_total_state_slots="
                f"{self.hybrid_state.total_slot_capacity or 0}, "
                f"delta_branch_slots_per_request="
                f"{self.hybrid_state.branch_slots_per_sequence}, "
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
            uniform_query_len=None,
            execution_mode=ExecutionMode.EAGER,
        )
        prepared = self.prepare_inputs([seq], prefill)
        with forward_context(prepared):
            self.model(prepared.input_ids, prepared.positions)

        seq.num_cached_tokens = num_tokens
        seq.append_token(0)
        seq.num_scheduled_tokens = 1
        decode = BatchDescriptor(
            num_tokens=1,
            num_padded_tokens=1,
            num_seqs=1,
            uniform_query_len=1,
            execution_mode=ExecutionMode.EAGER,
        )
        prepared = self.prepare_inputs([seq], decode)
        try:
            with forward_context(prepared):
                self.model(prepared.input_ids, prepared.positions)
        finally:
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
    ) -> PreparedBatch:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
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
            sequence_index = len(cu_seqlens_q) - 2
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
        query_lengths = [seq.num_scheduled_tokens for seq in seqs]
        uniform_query_len = (
            query_lengths[0]
            if use_kv_cache
            and all(not seq.is_prefill for seq in seqs)
            and all(length == query_lengths[0] for length in query_lengths)
            else None
        )
        if uniform_query_len != descriptor.uniform_query_len:
            raise ValueError(
                "dispatcher and prepared attention metadata disagree on "
                "uniform query length"
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        delta_conv_slab = None
        delta_recurrent_slab = None
        delta_state_slots = None
        delta_branch_state_slots = None
        delta_chunk_indices = None
        delta_cu_chunks = None
        delta_chunk_sequences = None
        delta_recurrent_sequences = None
        if self.has_delta_state:
            state_view = self.hybrid_state.batch_view(
                (seq.seq_id for seq in seqs),
            )
            if state_view is None:
                raise RuntimeError("hybrid state manager returned no state view")
            delta_conv_slab, delta_recurrent_slab, delta_state_slots = state_view
            branch_width = (
                max_seqlen_q if self.hybrid_state.branches else 1
            )
            delta_branch_state_slots = self.hybrid_state.branch_slots_view(
                (seq.seq_id for seq in seqs),
                branch_width,
            )
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
        attention = AttentionMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_tile_seq_ids=query_tile_seq_ids,
            query_tile_starts=query_tile_starts,
            query_tile_lens=query_tile_lens,
            query_tile_positions=query_tile_positions,
            use_kv_cache=use_kv_cache,
        )
        gdn = None
        if self.has_delta_state:
            if any(
                item is None
                for item in (
                    delta_conv_slab,
                    delta_recurrent_slab,
                    delta_state_slots,
                    delta_chunk_indices,
                    delta_cu_chunks,
                    delta_chunk_sequences,
                    delta_recurrent_sequences,
                    delta_branch_state_slots,
                )
            ):
                raise RuntimeError("runner produced incomplete GDN metadata")
            gdn = GDNMetadata(
                cu_seqlens=cu_seqlens_q,
                conv_slab=delta_conv_slab,
                recurrent_slab=delta_recurrent_slab,
                state_slots=delta_state_slots,
                branch_state_slots=delta_branch_state_slots,
                chunk_indices=delta_chunk_indices,
                cu_chunks=delta_cu_chunks,
                chunk_sequences=delta_chunk_sequences,
                recurrent_sequences=delta_recurrent_sequences,
            )
        return PreparedBatch(
            input_ids=input_ids,
            positions=positions,
            signature=descriptor.signature,
            attention=attention,
            sampling=SamplingMetadata(logits_indices=logits_indices),
            gdn=gdn,
        )

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        prepared: PreparedBatch,
        descriptor: BatchDescriptor,
    ):
        input_ids = prepared.input_ids
        positions = prepared.positions
        if descriptor.execution_mode is ExecutionMode.EAGER:
            return self.model(input_ids, positions)
        if descriptor.execution_mode is ExecutionMode.PIECEWISE:
            if self.piecewise_model is None:
                raise RuntimeError("Piecewise CUDA Graph model is not initialized")
            torch.compiler.cudagraph_mark_step_begin()
            return self.piecewise_model(input_ids, positions)
        if descriptor.execution_mode is ExecutionMode.FULL:
            num_tokens = descriptor.num_tokens
            num_requests = descriptor.num_seqs
            padded_tokens = descriptor.num_padded_tokens
            padded_requests = descriptor.padded_requests
            attention = prepared.attention
            graph = self.graphs[descriptor.full_key]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:padded_tokens].zero_()
            graph_vars["positions"][:padded_tokens].zero_()
            graph_vars["input_ids"][:num_tokens] = input_ids
            graph_vars["positions"][:num_tokens] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:num_tokens] = attention.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:num_requests] = attention.context_lens
            graph_vars["cu_seqlens_q"][: padded_requests + 1].copy_(
                graph_vars["cu_seqlens_by_key"][descriptor.full_key]
            )
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][
                :num_requests, : attention.block_tables.size(1)
            ] = attention.block_tables
            if self.has_delta_state:
                if prepared.gdn is None:
                    raise RuntimeError("Full GDN replay requires prepared metadata")
                graph_vars["delta_state_slots"][:num_requests] = (
                    prepared.gdn.state_slots
                )
                graph_vars["delta_branch_state_slots"].fill_(-1)
                graph_vars["delta_branch_state_slots"][
                    :num_requests,
                    : prepared.gdn.branch_state_slots.size(1),
                ] = prepared.gdn.branch_state_slots
                num_dummy_slots = padded_requests - num_requests
                if num_dummy_slots:
                    dummy_slots = self.hybrid_state.dummy_slots(num_dummy_slots)
                    graph_vars["delta_state_slots"][
                        num_requests:padded_requests
                    ] = (
                        torch.tensor(
                            dummy_slots,
                            dtype=torch.int32,
                            device="cuda",
                        )
                    )
            if self.config.kv_cache_dtype == "fp8_e4m3":
                graph_vars["query_tile_seq_ids"].zero_()
                graph_vars["query_tile_starts"].zero_()
                graph_vars["query_tile_lens"].zero_()
                graph_vars["query_tile_positions"].zero_()
                num_tiles = attention.query_tile_seq_ids.numel()
                graph_vars["query_tile_seq_ids"][:num_tiles] = (
                    attention.query_tile_seq_ids
                )
                graph_vars["query_tile_starts"][:num_tiles] = (
                    attention.query_tile_starts
                )
                graph_vars["query_tile_lens"][:num_tiles] = (
                    attention.query_tile_lens
                )
                graph_vars["query_tile_positions"][:num_tiles] = (
                    attention.query_tile_positions
                )
            graph.replay()
            return graph_vars["outputs"][:num_tokens]
        raise ValueError(f"unsupported execution mode: {descriptor.execution_mode}")

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> RunnerStepOutput:
        sampled_seqs = [seq for seq in seqs if seq.will_sample]
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        query_lengths = [seq.num_scheduled_tokens for seq in seqs]
        uniform_query_len = (
            query_lengths[0]
            if all(not seq.is_prefill for seq in seqs)
            and all(length == query_lengths[0] for length in query_lengths)
            else None
        )
        descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            len(seqs),
            uniform_query_len,
        )

        use_speculative = self.speculator is not None
        has_verification = use_speculative and any(
            seq.is_speculative for seq in sampled_seqs
        )

        def step_output(
            result,
            speculative: SpeculativeStepMetrics | None = None,
        ) -> RunnerStepOutput:
            return RunnerStepOutput(
                result=result,
                metrics=RunnerStepMetrics(
                    execution_mode=descriptor.execution_mode.value,
                    real_tokens=descriptor.num_tokens,
                    padded_tokens=descriptor.num_padded_tokens,
                    num_requests=descriptor.num_seqs,
                    speculative=speculative or SpeculativeStepMetrics(),
                ),
            )
        state_transaction = self.hybrid_state.transaction(
            {
                seq.seq_id: seq.num_scheduled_tokens
                for seq in sampled_seqs
                if seq.is_speculative
            },
            enabled=has_verification,
        )
        state_transaction.begin()

        prepared = self.prepare_inputs(
            seqs,
            descriptor,
        )
        with forward_context(prepared):
            hidden_states = self.run_model(prepared, descriptor)
            if not sampled_seqs:
                if use_speculative:
                    self.speculator.propose(
                        seqs, hidden_states, [], [], [], []
                    )
                return step_output([] if self.rank == 0 else None)

            logits = self.model.compute_logits(hidden_states)
            if not use_speculative:
                temperatures = (
                    self.prepare_sample(sampled_seqs)
                    if self.rank == 0
                    else None
                )
                result = (
                    self.sampler(logits, temperatures).tolist()
                    if self.rank == 0
                    else None
                )
                return step_output(result)

            token_groups: list[list[int]] = []
            accepted_counts: list[int] = []
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
                    if seq.temperature == 0:
                        target_tokens = (
                            verification_logits.argmax(dim=-1).tolist()
                        )
                        outputs, accepted = self.acceptance_policy.accept(
                            target_tokens, seq.draft_token_ids
                        )
                        self.draft_logits.pop(seq.seq_id, None)
                    else:
                        draft_logits = self.draft_logits.pop(
                            seq.seq_id, None
                        )
                        if draft_logits is None:
                            raise RuntimeError(
                                "probabilistic MTP verification is missing "
                                "draft logits"
                            )
                        outputs, accepted = (
                            self.rejection_sampling_policy.accept(
                                verification_logits,
                                draft_logits,
                                seq.draft_token_ids,
                                seq.temperature,
                                generator=self._generator_for(seq),
                            )
                        )
                    token_groups.append(outputs)
                    accepted_counts.append(accepted)
                else:
                    token_groups.append(
                        [int(logits[logit_offset].argmax().item())]
                    )
                    accepted_counts.append(0)
                    logit_offset += 1
            if logit_offset != logits.size(0):
                raise ValueError("target verification logits were not fully consumed")

            if has_verification:
                state_transaction.commit(
                    {
                        seq.seq_id: 1 + accepted
                        for seq, accepted in zip(
                            sampled_seqs, accepted_counts
                        )
                        if seq.is_speculative
                    }
                )

            proposals = self.speculator.propose(
                seqs,
                hidden_states,
                token_groups,
                accepted_counts,
                [seq.temperature for seq in sampled_seqs],
                [self._generator_for(seq) for seq in sampled_seqs],
            )
            if len(proposals) != len(sampled_seqs):
                raise ValueError("MTP proposal count does not match sampled requests")
            proposed = sum(
                len(seq.draft_token_ids) for seq in sampled_seqs
            )
            next_draft_token_ids: list[list[int] | None] = []
            for seq, proposal in zip(sampled_seqs, proposals):
                if proposal is None:
                    next_draft_token_ids.append(None)
                    self.draft_logits.pop(seq.seq_id, None)
                else:
                    next_draft_token_ids.append(proposal.token_ids)
                    if seq.temperature > 0:
                        self.draft_logits[seq.seq_id] = proposal.logits
                    else:
                        self.draft_logits.pop(seq.seq_id, None)
            accepted = sum(accepted_counts)
            speculative_metrics = SpeculativeStepMetrics(
                drafted=sum(
                    len(proposal.token_ids)
                    for proposal in proposals
                    if proposal is not None
                ),
                proposed=proposed,
                accepted=accepted,
                rejected=proposed - accepted,
                bonus=sum(
                    int(
                        seq.is_speculative
                        and accepted_count == len(seq.draft_token_ids)
                    )
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                verification_rounds=sum(
                    int(seq.is_speculative) for seq in sampled_seqs
                ),
                accepted_position_1=sum(
                    int(seq.is_speculative and accepted_count >= 1)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                accepted_position_2=sum(
                    int(seq.is_speculative and accepted_count >= 2)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
                accepted_position_3=sum(
                    int(seq.is_speculative and accepted_count >= 3)
                    for seq, accepted_count in zip(
                        sampled_seqs, accepted_counts
                    )
                ),
            )
            return step_output(
                SpeculativeBatchOutput(
                    token_ids=token_groups,
                    accepted_counts=accepted_counts,
                    next_draft_token_ids=next_draft_token_ids,
                ),
                speculative_metrics,
            )

    @torch.inference_mode()
    def capture_piecewise_cudagraphs(self):
        if self.piecewise_model is None:
            return
        for size in reversed(self.piecewise_capture_sizes):
            seq = Sequence([0] * size)
            seq.temperature = 0.0
            seq.num_scheduled_tokens = size
            descriptor = BatchDescriptor(
                num_tokens=size,
                num_padded_tokens=size,
                num_seqs=1,
                uniform_query_len=None,
                execution_mode=ExecutionMode.PIECEWISE,
            )
            prepared = self.prepare_inputs([seq], descriptor)
            try:
                with forward_context(prepared):
                    for _ in range(2):
                        torch.compiler.cudagraph_mark_step_begin()
                        outputs = self.piecewise_model(
                            prepared.input_ids,
                            prepared.positions,
                        )
                        del outputs
                torch.cuda.synchronize()
            finally:
                self.release_sequences((seq.seq_id,))

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_requests = max(self.full_capture_sizes)
        max_query_len = max(self.full_query_lengths)
        max_tokens = max_requests * max_query_len
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        input_ids = torch.zeros(max_tokens, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_tokens, dtype=torch.int64, device="cuda")
        slot_mapping = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device="cuda"
        )
        context_lens = torch.zeros(
            max_requests, dtype=torch.int32, device="cuda"
        )
        cu_seqlens_q = torch.zeros(
            max_requests + 1, dtype=torch.int32, device="cuda"
        )
        query_tile_seq_ids = torch.zeros(
            max_tokens, dtype=torch.int32, device="cuda"
        )
        query_tile_starts = torch.zeros(
            max_tokens, dtype=torch.int32, device="cuda"
        )
        query_tile_lens = torch.zeros(
            max_tokens, dtype=torch.int32, device="cuda"
        )
        query_tile_positions = torch.zeros(
            max_tokens, dtype=torch.int32, device="cuda"
        )
        delta_state_slots = torch.arange(
            max_requests, dtype=torch.int32, device="cuda"
        )
        delta_branch_state_slots = torch.full(
            (max_requests, max_query_len),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        empty_delta_pairs = torch.empty(
            0, 2, dtype=torch.int32, device="cuda"
        )
        empty_delta_sequences = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        delta_cu_chunks = torch.zeros(1, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(
            max_requests,
            max_num_blocks,
            dtype=torch.int32,
            device="cuda",
        )
        outputs = torch.zeros(
            max_tokens,
            hf_config.hidden_size,
            dtype=hf_config.dtype,
            device="cuda",
        )
        self.graphs = {}
        self.graph_pool = None
        cu_seqlens_by_key = {}

        for query_len in reversed(self.full_query_lengths):
            for num_requests in reversed(self.full_capture_sizes):
                num_tokens = num_requests * query_len
                cu_seqlens_q[: num_requests + 1] = (
                    torch.arange(
                        num_requests + 1,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    * query_len
                )
                cu_seqlens_by_key[(query_len, num_requests)] = (
                    cu_seqlens_q[: num_requests + 1].clone()
                )
                positions[:num_tokens] = torch.arange(
                    query_len, dtype=torch.int64, device="cuda"
                ).repeat(num_requests)
                slot_mapping[:num_tokens] = torch.arange(
                    query_len, dtype=torch.int32, device="cuda"
                ).repeat(num_requests)
                context_lens[:num_requests].fill_(query_len)
                delta_branch_state_slots.fill_(-1)
                if self.has_delta_state and query_len > 1:
                    for sequence in range(num_requests):
                        branch_begin = (
                            max_requests + sequence * max_query_len
                        )
                        delta_branch_state_slots[
                            sequence, :query_len
                        ] = torch.arange(
                            branch_begin,
                            branch_begin + query_len,
                            dtype=torch.int32,
                            device="cuda",
                        )

                tile_sequence_ids = []
                tile_starts = []
                tile_lens = []
                tile_positions = []
                for sequence in range(num_requests):
                    for tile_offset in range(
                        0, query_len, FP8_QUERY_TILE_SIZE
                    ):
                        tile_sequence_ids.append(sequence)
                        tile_starts.append(sequence * query_len + tile_offset)
                        tile_lens.append(
                            min(
                                FP8_QUERY_TILE_SIZE,
                                query_len - tile_offset,
                            )
                        )
                        tile_positions.append(tile_offset)
                num_tiles = len(tile_sequence_ids)
                query_tile_seq_ids[:num_tiles] = torch.tensor(
                    tile_sequence_ids, dtype=torch.int32, device="cuda"
                )
                query_tile_starts[:num_tiles] = torch.tensor(
                    tile_starts, dtype=torch.int32, device="cuda"
                )
                query_tile_lens[:num_tiles] = torch.tensor(
                    tile_lens, dtype=torch.int32, device="cuda"
                )
                query_tile_positions[:num_tiles] = torch.tensor(
                    tile_positions, dtype=torch.int32, device="cuda"
                )

                gdn = (
                    GDNMetadata(
                        cu_seqlens=cu_seqlens_q[: num_requests + 1],
                        conv_slab=self.delta_state_slab[0],
                        recurrent_slab=self.delta_state_slab[1],
                        state_slots=delta_state_slots[:num_requests],
                        branch_state_slots=delta_branch_state_slots[
                            :num_requests, :query_len
                        ],
                        chunk_indices=empty_delta_pairs,
                        cu_chunks=delta_cu_chunks,
                        chunk_sequences=empty_delta_sequences,
                        recurrent_sequences=delta_state_slots[:num_requests],
                    )
                    if self.has_delta_state
                    else None
                )
                prepared = PreparedBatch(
                    input_ids=input_ids[:num_tokens],
                    positions=positions[:num_tokens],
                    signature=ExecutionSignature(
                        num_tokens=num_tokens,
                        num_requests=num_requests,
                        num_padded_tokens=num_tokens,
                        uniform_query_len=query_len,
                    ),
                    attention=AttentionMetadata(
                        cu_seqlens_q=cu_seqlens_q[: num_requests + 1],
                        max_seqlen_q=query_len,
                        slot_mapping=slot_mapping[:num_tokens],
                        context_lens=context_lens[:num_requests],
                        block_tables=block_tables[:num_requests],
                        query_tile_seq_ids=query_tile_seq_ids[:num_tiles],
                        query_tile_starts=query_tile_starts[:num_tiles],
                        query_tile_lens=query_tile_lens[:num_tiles],
                        query_tile_positions=query_tile_positions[:num_tiles],
                        use_kv_cache=True,
                    ),
                    sampling=SamplingMetadata(),
                    gdn=gdn,
                )
                graph = torch.cuda.CUDAGraph()
                with forward_context(prepared):
                    outputs[:num_tokens] = self.model(
                        prepared.input_ids,
                        prepared.positions,
                    )
                    with torch.cuda.graph(graph, self.graph_pool):
                        outputs[:num_tokens] = self.model(
                            prepared.input_ids,
                            prepared.positions,
                        )
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[(query_len, num_requests)] = graph
                torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_by_key=cu_seqlens_by_key,
            block_tables=block_tables,
            query_tile_seq_ids=query_tile_seq_ids,
            query_tile_starts=query_tile_starts,
            query_tile_lens=query_tile_lens,
            query_tile_positions=query_tile_positions,
            delta_state_slots=delta_state_slots,
            delta_branch_state_slots=delta_branch_state_slots,
            outputs=outputs,
        )
