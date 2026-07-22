# Speculative-decoding engineering knowledge

Originating spec: [Speculative decoding and Qwen MTP](../specs/speculative-decoding.md)

## Stable V1 protocol

- Pending draft tokens belong to request state and consume the same scheduler
  token budget and cache capacity as other scheduled model inputs.
- Proposal, target verification, acceptance, state commit/rollback, and next
  proposal are separate ownership phases even when fused for performance.
- Running a rejected draft through the target does not commit it. Request token
  counters, KV cache, prefix-cache visibility, and recurrent/conv state must all
  end at the accepted prefix.
- A proposer is method-specific; the scheduler and acceptance policy are not.
  MTP, a draft model, or a future tree proposer must share one request protocol.
- Greedy prefix comparison is not probabilistic rejection sampling. Lossless
  sampling claims require the latter and its distributional tests.
- The next proposal consumes post-processed accepted/rejected state. It may
  overlap independent output transfer, but not state-commit ordering.

## Metrics

- Record rounds, proposed/drafted/accepted/rejected tokens, bonus tokens,
  proposal and verification latency, replay latency, throughput, memory, and
  per-position acceptance.
- Mean acceptance length conventionally includes the target bonus token:
  `1 + accepted_draft_tokens / draft_rounds`.
- Choose `k` using end-to-end throughput/latency under concurrency, not raw
  acceptance rate.

## Source baseline and exclusions

The V1 study used vLLM commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd).
The V0 `SpecDecodeWorker`, `MultiStepWorker`, and separate draft/target worker
topology are obsolete design references for nano-vLLM.
