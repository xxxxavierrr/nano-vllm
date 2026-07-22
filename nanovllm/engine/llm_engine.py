import atexit
from dataclasses import dataclass, fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative_step import SpeculativeBatchOutput


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
    execution_mode: str
    actual_scheduled_tokens: int = 0
    padded_scheduled_tokens: int = 0
    running_requests: int = 0
    speculative_drafted_tokens: int = 0
    speculative_proposed_tokens: int = 0
    speculative_accepted_tokens: int = 0
    speculative_rejected_tokens: int = 0
    speculative_bonus_tokens: int = 0
    speculative_verification_rounds: int = 0
    speculative_accepted_position_1: int = 0
    speculative_accepted_position_2: int = 0
    speculative_accepted_position_3: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
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
        aborted = self.scheduler.abort(seq_id)
        if aborted:
            self.model_runner.call("release_sequences", (seq_id,))
        return aborted

    def execute_batch(self, batch):
        if batch.reset_sequence_ids:
            self.model_runner.call(
                "release_sequences", batch.reset_sequence_ids
            )
        runner_output = self.model_runner.call("run", batch.sequences)
        result = runner_output.result
        if isinstance(result, SpeculativeBatchOutput):
            self.scheduler.postprocess(
                batch,
                result.token_ids,
                accepted_counts=result.accepted_counts,
                next_draft_token_ids=result.next_draft_token_ids,
            )
            token_ids = result.token_ids
        else:
            self.scheduler.postprocess(batch, result)
            token_ids = result
        finished_ids = tuple(
            seq.seq_id for seq in batch.sequences if seq.is_finished
        )
        if finished_ids:
            self.model_runner.call("release_sequences", finished_ids)
        return token_ids, runner_output.metrics

    def step(self):
        batch = self.scheduler.schedule()
        previous_lengths = {
            seq.seq_id: seq.num_tokens for seq in batch.sequences
        }
        _, runner_metrics = self.execute_batch(batch)
        outputs = []
        for seq in batch.sequences:
            new_tokens = seq.token_ids[previous_lengths[seq.seq_id] :]
            for token_index, token_id in enumerate(new_tokens):
                is_last = token_index == len(new_tokens) - 1
                outputs.append(
                    EngineOutput(
                        seq.seq_id,
                        token_id,
                        seq.is_finished and is_last,
                        seq.finish_reason if is_last else None,
                        seq.num_prefix_cached_tokens,
                    )
                )
        speculative = runner_metrics.speculative
        return outputs, EngineStepStats(
            batch.prefill_tokens,
            batch.decode_tokens,
            runner_metrics.execution_mode,
            actual_scheduled_tokens=runner_metrics.real_tokens,
            padded_scheduled_tokens=runner_metrics.padded_tokens,
            running_requests=runner_metrics.num_requests,
            speculative_drafted_tokens=speculative.drafted,
            speculative_proposed_tokens=speculative.proposed,
            speculative_accepted_tokens=speculative.accepted,
            speculative_rejected_tokens=speculative.rejected,
            speculative_bonus_tokens=speculative.bonus,
            speculative_verification_rounds=speculative.verification_rounds,
            speculative_accepted_position_1=speculative.accepted_position_1,
            speculative_accepted_position_2=speculative.accepted_position_2,
            speculative_accepted_position_3=speculative.accepted_position_3,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
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
        return [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
