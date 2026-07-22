# Design

## 1. Indexed GDN state branches

Implement this before probabilistic sampling because non-greedy verification
will reject more often and would magnify the current replay cost.

- Replace committed/working slab transactions with a physical state-slot pool
  and a `StateBranchTable` owned by `HybridStateManager`.
- A request owns one committed base slot. For `k` drafts, the target step
  reserves `q = 1 + k` candidate slots representing the state after each
  scheduled input position.
- Packed convolution and recurrent kernels load the base state once, advance
  sequentially, and write each prefix state to its candidate slot.
- If `a` draft tokens are accepted, the committed model frontier has consumed
  `1 + a` scheduled inputs. Commit atomically remaps the request to candidate
  slot `1 + a`; the old base and all unselected candidates return to the pool.
- The recovery token sampled at the rejection position is output now but is
  processed in the next scheduler step, matching the existing token frontier.
- Graph metadata uses stable `[request_bucket, max_spec_tokens + 1]` state-index
  buffers. Invalid branches use a sentinel and kernels must not touch them.
- Branch-capacity exhaustion reduces `k` or disables speculation for selected
  requests; it never aliases state.

This exchanges bounded state memory for eliminating whole-model replay and
state-slab copies.

## 2. Lossless probabilistic rejection sampler

Introduce typed `DraftBatch` and `VerificationBatch` records.

- `DraftBatch` carries token IDs and draft logits `[requests, k, vocab]` (or an
  explicitly documented compressed equivalent later).
- Greedy acceptance remains a separate cheap policy.
- The probabilistic policy applies the same temperature transform to target and
  draft logits, then accepts proposed token `x_i` when
  `log(u_i) <= log p_i(x_i) - log q_i(x_i)`.
- At the first rejection it samples from normalized `relu(p_i - q_i)`.
- If all drafts are accepted it samples the bonus token from target `p`.
- A Triton blockwise implementation computes logsumexp, residual mass, and
  Gumbel selection without writing additional normalized `[R,k,V]` softmax
  tensors.
- Per-request RNG state is explicit and deterministic across batching and
  Graph replay.

Initial scope supports greedy and temperature-only sampling. Top-k/top-p wait
until target and draft can provably apply identical transforms.

## 3. W4A16 runtime backend

Keep the current raw-layout Triton operator as a correctness fallback while
introducing an explicit loader/runtime boundary.

### Load-time representation

- Normalize the supported symmetric group-size-128 checkpoint once.
- For `desc_act`, compute `perm = argsort(g_idx)`, repack weights in permuted K
  order, and store only metadata required by the runtime kernel.
- Interleave packed weights and scales in the order consumed by the kernel.
- Fuse the matching activation permutation into A-tile loads; do not launch a
  standalone gather.
- Include repack workspace in startup memory accounting and release it before
  cache capacity is finalized.

### Runtime representation

- One custom operator owns packed decode, dequantization, and W4A16 GEMM.
- No global BF16 dequantized weight or scratch tensor is allowed.
- Backend selection is shape/hardware based (`M`, `K`, `N`, SM), not based on
  scheduler labels.
- First optimize a repacked Triton backend to remove raw `g_idx`/layout costs.
  Because Triton `tl.dot` still consumes a widened tile, the production target
  is an Ada SM89/RTX 4090D-specific Marlin-style CUDA kernel with pipelined
  packed loads, dequantization, and tensor-core accumulation. Do not assume
  Hopper-only WGMMA/TMA features.
- Bias and TP restrictions remain at the linear-method boundary.

## Integration boundaries

- Scheduler owns token and branch-state reservation, but no numerical policy.
- ModelRunner owns ordering: target verify -> acceptance -> state commit -> next
  proposal.
- Acceptance policy owns distribution correctness and RNG.
- HybridStateManager owns slots, branch lifetime, and atomic remap.
- Quantization method owns checkpoint validation, post-load repack, workspace,
  and kernel selection.

## Risks

- Candidate state slots reduce maximum concurrency; capacity must be measured
  before enabling large `k` broadly.
- Full draft logits are memory-expensive. Correctness comes first; compressed
  logits are a separate optimization requiring proof that residual sampling is
  unchanged.
- A CUDA W4 kernel has substantially higher maintenance cost than Triton and
  needs real RTX 4090D SM89 validation.
- Graph replay requires stable RNG/state-index buffers and must not capture
  stale pointers.
- The 24 GB device budget makes speculative state branches, KV cache, Graph
  pools, draft logits, and W4 workspace direct competitors. Capacity planning
  and concurrency measurement are acceptance requirements, not follow-up
  observability.

## Revised optimization sequence and current status

The order is based on removing known computation waste before trading numerical
precision for memory. Already implemented foundations are not new development
milestones; they remain GPU validation gates.

1. **Marlin-style W4A16 small-M and large-M -- opt-in source scaffolded, GPU
   validation and optimization pending.** The native extension, SM89/layout
   gates, `M<=64`/large-M dispatch, packed in-operator decode, and fallback are
   present. `auto` deliberately remains Triton. CUDA build correctness,
   tensor-core dataflow optimization, Graph capture, and target-shape
   performance are not yet established and must not be represented as done.
2. **DeltaNet speculative state branches -- implemented locally, GPU validation
   pending.** Indexed prefix states and commit-by-remap replace partial-
   rejection target replay. Do not implement this a second time.
3. **Lossless probability-difference rejection -- correctness implementation
   complete, optimized GPU path pending.** Greedy remains a fast policy;
   temperature sampling already uses `p/q` acceptance and normalized `(p-q)+`
   recovery with deterministic request RNG. Remaining work is integrated GPU,
   Graph, distribution, and blockwise-kernel validation/optimization.
4. **Fused W4A8 large-M -- experimental source scaffolded, disabled.** The
   native operator performs per-row/per-group activation quantization while
   consuming packed W4 weights without a global activation scratch. It is not
   selected by production dispatch until numerical and goodput evidence exists.
5. **FP8 KV capacity experiment -- runtime implemented, experiment pending.**
   Compare native and existing FP8 KV at each mode's maximum stable concurrency
   and long-context workloads. Adopt only if extra concurrency raises SLO
   goodput enough to repay the FP8 attention cost.
6. **FP8 DeltaNet-state capacity experiment -- not implemented.** Quantize
   committed and branch state with explicit scales/lifetime. Adopt only if it
   increases effective scheduler batch and SLO goodput within numerical limits.
7. **FFN W3A16 -- deferred and not implemented.** Consider W3 only after memory
   accounting proves W4 weights, Graph pools, KV, draft, and state still block
   desired concurrency. Kernel complexity and quality cost require a measured
   capacity need, not just a smaller checkpoint.

DSpark target/draft formats use the same runtime-method boundary. The paired
AWQ target is the acceptance baseline; a GPTQ target and an INT4 draft are
introduced as separate variables so target mismatch and draft quantization
loss cannot be conflated. Draft compression reuses the W4A16 runtime backend
rather than creating a DSpark-specific GEMM kernel.

The 8.8 GB BF16 DSpark draft cannot coexist with the target and runtime state on
the 24 GB GPU. It is therefore not an online baseline. The BF16 checkpoint is
loaded alone only for calibration and sequential offline agreement evidence:

```text
run target alone -> persist real DSpark input/hidden-state corpus on CPU/disk
unload target -> load BF16 draft alone -> calibrate and record reference logits
unload BF16 draft -> load INT4 draft alone -> compare logits/top-1/KL
online runnable cells -> paired AWQ target + INT4 draft
                      -> nano GPTQ target + the same INT4 draft
```

Those two online cells isolate target-format/checkpoint mismatch because the
draft is held fixed. Exact BF16-versus-INT4 end-to-end acceptance loss cannot be
measured on this 24 GB device; author-published BF16 acceptance is an external
reference, not a local apples-to-apples result. A larger GPU would be required
for that exact cell. On the available GPU, compare runnable INT4 variants
(group size and mixed-precision exceptions) and use offline BF16 agreement as
the quantization-quality guardrail.

## Goodput experiment design

Microbenchmarks remain phase-local diagnostics. The production decision uses:

```text
output tokens/s
accepted speculative tokens/s
requests/s
SLO-good output tokens/s and requests/s
time-weighted average/max running requests
scheduled actual/padded tokens per step
GPU compute and memory utilization
TTFT p50/p99
TPOT p50/p99
maximum passing offered load under the declared SLO
```

Closed-loop concurrency sweeps reveal saturation and memory capacity. An
open-loop arrival-rate sweep establishes maximum SLO throughput: increase
offered load until p99/error SLO failure, then refine around the highest
passing point. Offline runs isolate engine and scheduler effects; online runs
are authoritative for serving claims.
