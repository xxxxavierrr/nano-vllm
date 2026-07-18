import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

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
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from nanovllm.layers.linear import quantize_fp8
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


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
        self.has_delta_state = hasattr(self.model, "create_delta_state")
        self.delta_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.delta_state_capacity: int | None = None
        load_model(self.model, config.model)
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
        self.delta_states.clear()
        torch.cuda.synchronize()
        dist.destroy_process_group()

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
        self, seq_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.has_delta_state:
            return None
        state = self.delta_states.get(seq_id)
        if state is not None:
            return state
        if (
            self.delta_state_capacity is not None
            and len(self.delta_states) >= self.delta_state_capacity
        ):
            raise RuntimeError(
                "Qwen3.5/3.6 DeltaNet state capacity exhausted: "
                f"{self.delta_state_capacity} active sequences"
            )
        state = self.model.create_delta_state(
            device="cuda", dtype=self.config.hf_config.dtype
        )
        self.delta_states[seq_id] = state
        self.max_active_delta_states = max(
            self.max_active_delta_states, len(self.delta_states)
        )
        return state

    def release_sequences(self, seq_ids):
        for seq_id in seq_ids:
            self.delta_states.pop(seq_id, None)


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
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        num_attention_layers = len(attention_modules)
        block_bytes = (
            2 * num_attention_layers * self.block_size
            * num_kv_heads * head_dim * hf_config.dtype.itemsize
        )
        available_bytes = int(
            total * config.gpu_memory_utilization - used - peak + current
        )
        reserved_delta_bytes = 0
        if self.has_delta_state:
            state_bytes = self.model.delta_state_bytes(hf_config.dtype)
            total_capacity = (available_bytes - block_bytes) // state_bytes
            if total_capacity < 1:
                raise RuntimeError("insufficient GPU memory for one DeltaNet state")
            half_capacity = max(1, available_bytes // 2 // state_bytes)
            self.delta_state_capacity = min(
                config.max_num_seqs, total_capacity, half_capacity
            )
            config.max_num_seqs = self.delta_state_capacity
            reserved_delta_bytes = self.delta_state_capacity * state_bytes
        cache_bytes = available_bytes - reserved_delta_bytes
        config.num_kvcache_blocks = cache_bytes // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            num_attention_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.dtype,
            device="cuda",
        )
        for layer_id, module in enumerate(attention_modules):
            module.k_cache.tensor = self.kv_cache[0, layer_id]
            module.v_cache.tensor = self.kv_cache[1, layer_id]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_inputs(self, seqs: list[Sequence], descriptor: BatchDescriptor):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        sequence_slices = []
        slot_mapping = []
        logits_indices = []
        context_lens = []
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
            if seqlen_q == 1 and start == seq.num_tokens - 1:
                # Decode sequences only serialize last_token to TP workers.
                input_ids.append(seq.last_token)
            else:
                scheduled_token_ids = seq[start:end]
                if len(scheduled_token_ids) != seqlen_q:
                    raise ValueError(f"sequence {seq_id} is missing scheduled token ids")
                input_ids.extend(scheduled_token_ids)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            sequence_slices.append((cu_seqlens_q[-2], cu_seqlens_q[-1]))
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(seqlen_k)
            if seq.will_sample:
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
        if self.has_delta_state:
            delta_states = tuple(
                self.get_delta_state(seq.seq_id) for seq in seqs
            )
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
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
            use_kv_cache=use_kv_cache,
            sequence_slices=tuple(sequence_slices),
            delta_states=delta_states,
            is_uniform_decode=is_uniform_decode,
            num_actual_tokens=num_actual_tokens,
            num_padded_tokens=model_num_tokens,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

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
            graph.replay()
            return graph_vars["outputs"][:bs]
        raise ValueError(f"unsupported execution mode: {descriptor.execution_mode}")

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
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
        input_ids, positions = self.prepare_inputs(seqs, descriptor)
        try:
            hidden_states = self.run_model(input_ids, positions, descriptor)
            self.last_execution_mode = descriptor.execution_mode.value
            if not sampled_seqs:
                return [] if self.rank == 0 else None
            temperatures = self.prepare_sample(sampled_seqs) if self.rank == 0 else None
            logits = self.model.compute_logits(hidden_states)
            return self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
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
            outputs=outputs,
        )
