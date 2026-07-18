import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass(slots=True)
class EngineOutput:
    seq_id: int
    token_id: int
    finished: bool
    finish_reason: str | None
    cached_tokens: int


@dataclass(slots=True)
class EngineStepStats:
    prefill_tokens: int
    decode_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self._closed = False
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def abort_request(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def step(self):
        batch = self.scheduler.schedule()
        previous_completion_tokens = {
            seq.seq_id: seq.num_completion_tokens for seq in batch.sequences
        }
        token_ids = self.model_runner.call("run", batch.sequences)
        self.scheduler.postprocess(batch, token_ids)
        outputs = [
            EngineOutput(
                seq.seq_id,
                seq.last_token,
                seq.is_finished,
                seq.finish_reason,
                seq.num_prefix_cached_tokens,
            )
            for seq in batch.sequences
            if seq.num_completion_tokens > previous_completion_tokens[seq.seq_id]
        ]
        return outputs, EngineStepStats(batch.prefill_tokens, batch.decode_tokens)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, stats = self.step()
            elapsed = perf_counter() - t
            if stats.prefill_tokens:
                prefill_throughput = stats.prefill_tokens / elapsed
            if stats.decode_tokens:
                decode_throughput = stats.decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for item in output:
                outputs.setdefault(item.seq_id, []).append(item.token_id)
                if item.finished:
                    pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
