import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.capacity import plan_delta_state_capacity
from nanovllm.engine.batch_planner import BatchPlanner
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
from nanovllm.engine.speculative_step import SpeculativeStepCoordinator
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


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.cudagraph_mode = CUDAGraphMode.parse(config.cudagraph_mode)
        self.enforce_eager = self.cudagraph_mode is CUDAGraphMode.NONE
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._init_distributed()
        self._validate_cache_dtypes()
        self._build_and_load_models()
        self._init_graph_policy()
        self._warmup_runtime()
        self._init_tensor_parallel_worker()

    def _init_distributed(self) -> None:
        init_method = f"tcp://{self.config.master_host}:{self.config.master_port}"
        dist.init_process_group(
            "nccl", init_method, world_size=self.world_size, rank=self.rank
        )
        torch.cuda.set_device(self.config.device_ids[self.rank])

    def _validate_cache_dtypes(self) -> None:
        config = self.config
        hf_config = config.hf_config
        if config.delta_state_dtype == "fp8_e4m3":
            capability = torch.cuda.get_device_capability()
            if capability < (8, 9):
                raise RuntimeError(
                    "FP8 DeltaNet state validation requires SM89 or newer; "
                    f"current capability is SM{capability[0]}{capability[1]}"
                )
            raise RuntimeError(
                "FP8 DeltaNet state runtime is fail-closed pending fused "
                "conv/recurrent SM89 correctness and CUDA Graph validation; "
                "use delta_state_dtype='auto' for production"
            )
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

    def _build_and_load_models(self) -> None:
        config = self.config
        hf_config = config.hf_config
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            if config.model_family == "qwen3_5":
                self.model = Qwen3_5ForConditionalGeneration(
                    hf_config, config.gptq_config
                )
            else:
                self.model = Qwen3ForCausalLM(hf_config, config.gptq_config)
            self.mtp_model = (
                Qwen3_5MTP(hf_config)
                if config.speculative_method == "mtp"
                else None
            )
            self.hybrid_state = HybridStateManager(self.model, hf_config.dtype)
            self.has_delta_state = self.hybrid_state.enabled
            self.batch_planner = BatchPlanner(
                block_size=self.block_size,
                use_fp8_kv=config.kv_cache_dtype == "fp8_e4m3",
                hybrid_state=self.hybrid_state,
            )
            load_model(self.model, config.model)
            if self.mtp_model is not None:
                load_mtp_model(self.mtp_model, config.mtp_model, hf_config)
            self.speculator = (
                MTPProposer(
                    self.model, self.mtp_model, block_size=self.block_size,
                    num_steps=config.num_speculative_tokens,
                )
                if self.mtp_model is not None
                else None
            )
            self.speculative_step = SpeculativeStepCoordinator(
                num_drafts=config.num_speculative_tokens,
                device=f"cuda:{config.device_ids[self.rank]}",
            )
            if config.quantization == "fp8":
                quantize_fp8(self.model)
                torch.cuda.empty_cache()
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)
        self.sampler = Sampler()

    def _init_graph_policy(self) -> None:
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

    def _warmup_runtime(self) -> None:
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

    def _init_tensor_parallel_worker(self) -> None:
        if self.world_size > 1:
            if self.rank == 0:
                self.shm = SharedMemory(
                    name=self.config.shm_name, create=True, size=2**20
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.config.shm_name)
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
        seq_ids = tuple(seq_ids)
        self.hybrid_state.release(seq_ids)
        self.speculative_step.release(seq_ids)

    def reset_metrics(self) -> None:
        self.hybrid_state.reset_metrics()

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

    def prepare_inputs(
        self,
        seqs: list[Sequence],
        descriptor: BatchDescriptor,
    ) -> PreparedBatch:
        return self.batch_planner.prepare(seqs, descriptor)

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

    def _execution_descriptor(self, seqs: list[Sequence]) -> BatchDescriptor:
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        query_lengths = [seq.num_scheduled_tokens for seq in seqs]
        uniform_query_len = (
            query_lengths[0]
            if all(not seq.is_prefill for seq in seqs)
            and all(length == query_lengths[0] for length in query_lengths)
            else None
        )
        return self.cudagraph_dispatcher.dispatch(
            num_tokens, len(seqs), uniform_query_len
        )

    @staticmethod
    def _step_output(
        descriptor: BatchDescriptor,
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

    def _begin_speculative_state(self, sampled_seqs: list[Sequence], enabled: bool):
        transaction = self.hybrid_state.transaction(
            {
                seq.seq_id: seq.num_scheduled_tokens
                for seq in sampled_seqs
                if seq.is_speculative
            },
            enabled=enabled,
        )
        transaction.begin()
        return transaction

    def _sample_ordinary(self, logits, sampled_seqs: list[Sequence]):
        temperatures = self.prepare_sample(sampled_seqs) if self.rank == 0 else None
        return self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    def _run_speculative_step(
        self,
        seqs: list[Sequence],
        sampled_seqs: list[Sequence],
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        transaction,
        has_verification: bool,
    ) -> tuple[object, SpeculativeStepMetrics]:
        verified = self.speculative_step.verify(sampled_seqs, logits)
        if has_verification:
            self.speculative_step.commit(transaction, sampled_seqs, verified)
        return self.speculative_step.propose(
            self.speculator, seqs, hidden_states, sampled_seqs, verified
        )

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> RunnerStepOutput:
        sampled_seqs = [seq for seq in seqs if seq.will_sample]
        descriptor = self._execution_descriptor(seqs)
        use_speculative = self.speculator is not None
        has_verification = use_speculative and any(
            seq.is_speculative for seq in sampled_seqs
        )
        transaction = self._begin_speculative_state(
            sampled_seqs, has_verification
        )
        prepared = self.prepare_inputs(seqs, descriptor)
        with forward_context(prepared):
            hidden_states = self.run_model(prepared, descriptor)
            if not sampled_seqs:
                if use_speculative:
                    self.speculator.propose(seqs, hidden_states, [], [], [], [])
                result = [] if self.rank == 0 else None
                return self._step_output(descriptor, result)
            logits = self.model.compute_logits(hidden_states)
            if not use_speculative:
                result = self._sample_ordinary(logits, sampled_seqs)
                return self._step_output(descriptor, result)
            result, metrics = self._run_speculative_step(
                seqs, sampled_seqs, hidden_states, logits,
                transaction, has_verification,
            )
            return self._step_output(descriptor, result, metrics)

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
