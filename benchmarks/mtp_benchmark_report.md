# nano-vLLM Qwen3.6 MTP 投机解码实验报告

日期：2026-07-19

代码基线：`7ed7f6c`，另含本次尚待提交的 DeltaNet 状态容量规划与 benchmark harness 改动

目标模型：`Qwen3.6-27B GPTQ INT4`

MTP 模型：`Qwen3.6-27B-mtp`

## 1. 结论摘要

本轮实验已经完成。结论不是简单地把并发或 `k` 拉到最大：

- **吞吐最高点**：`k=3, concurrency=6, token budget=256`，达到 **52.603 output tok/s**。
- **更均衡的在线配置**：`k=2, concurrency=4, token budget=256`，达到 **48.875 output tok/s**，同时 `TTFT p95=512.9 ms`、`TPOT p95=88.9 ms`。
- 相对同并发的非投机基线，`k=2` 在 `concurrency=1/2/4` 分别获得 **2.04× / 1.89× / 1.77×** 吞吐提升；`k=3,c=6` 相对 `k=0,c=6` 为 **1.58×**。
- DeltaNet 状态容量已经从固定的“吃掉一半可用显存”改为按请求状态和最低 KV 需求联合规划，27B 模型现可实际容纳 **8 个 active request**，且实测无 preemption。
- **8 个槽位只是容量，不是最佳执行并发**。`c=8` 的尾延迟显著恶化，主实验中 `k=2/3` 的 `TTFT p95` 都超过 13 秒。
- `max_num_batched_tokens=256` 是当前测试空间内的最佳预算。`128` 限制过紧；`512` 会引入形状/编译路径抖动并显著恶化尾延迟，因此没有继续测试 `1024`。
- MTP 接受率不是越长越高。主实验中 `k=2` 总接受率约 86%，`k=3` 约 76%～82%；更长草稿带来更多潜在 token，也带来更多无效验证计算，最佳 `k` 会随 active batch 改变。
- 当前 W4 路径在不同 GEMM `M` 形状下并不保证逐 token bit-exact。MTP 与 baseline 的完整回答一致率较低，不能据此断言语义质量下降，但也不能宣称准确率已验证；这需要独立质量集和 logit-margin 分析。

## 2. 实验问题与思路

我们需要回答三个不同问题：

1. **请求容量**：显存最多能同时保存多少条 Qwen3.6 请求的 DeltaNet recurrent/conv state？
2. **调度预算**：每轮允许调度多少 token，才能平衡 prefill、MTP verify 和 decode？
3. **投机长度**：在不同 active concurrency 下，`k=0/1/2/3` 哪个吞吐和延迟最好？

原实现把一半可用显存固定分给 DeltaNet state，在短上下文测试中只得到 4 个槽位。这个分法没有依据实际 KV 需求，导致我们测到的是人为容量上限，而不是模型/硬件上限。因此先实施容量规划：

- 计算每条序列的 DeltaNet committed state 和 MTP working state。
- 给每条序列保证覆盖 `max_model_len + speculative_tokens` 所需的最低 KV blocks。
- 在满足最低 KV 容量后，用剩余显存决定 state slot 数。
- 缓存完成后再次校验 KV block 保证，避免 silent overcommit。

然后采用两阶段搜索：

1. 固定 `token budget=256`，做 `k × concurrency` 主矩阵，确定真实吞吐峰值和在线甜点位。
2. 只在有竞争力的 `k=2/3`、`c=4/6/8` 上扫描 `token budget=128/256/512`；当 512 已明显劣于 256 时早停，不再跑 1024。

benchmark 使用 **closed-loop constant concurrency**：每完成一个请求立即补入下一个请求，直到完成指定总请求数。这样不会把只有一波请求的尾部排空阶段误当成稳态吞吐。每个 case 还会预热从 1 到目标并发的所有 active batch shape，尽量排除首次 Triton autotune/编译时间。

## 3. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24,564 MiB，SM 8.9 |
| NVIDIA driver | 580.76.05 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| 主模型 | `/root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4` |
| MTP 模型 | `/root/autodl-tmp/huggingface/Qwen3.6-27B-mtp` |
| 权重量化 | GPTQ W4A16 |
| KV Cache | BF16 (`kv-cache-dtype=auto`) |
| TP | 1 |
| 执行模式 | eager |
| 采样 | greedy，ignore EOS |

### 3.1 容量计算

Qwen3.6-27B 包含 48 个 DeltaNet 层。实测每请求：

- 一份 DeltaNet state：147.75 MiB。
- MTP 路径需要 committed + working 两份：295.50 MiB/request。
- 8 个请求的 DeltaNet state：2,364 MiB。

对 `max_model_len=256, k=3`，规划器结果为：

```text
capacity=8
state_bytes_per_sequence=309,854,208
kv_blocks_per_sequence=2
minimum_kv_blocks=16
```

实际 `k=3,c=8` smoke test 观测到 8 条 active request，未发生 preemption，证明“8 槽位”已经真实达成。

## 4. 主实验

### 4.1 Workload

| 参数 | 值 |
|---|---:|
| input length | 128 tokens |
| output length | 64 tokens |
| requests | 32 |
| concurrency | 1, 2, 4, 6, 8 |
| speculative `k` | 0, 2, 3 |
| max batched tokens | 256 |
| max state/request slots | 8 |
| prompt | 内置自然语言 prompt，扩展到精确 token 长度 |
| repeats | 1 个长 closed-loop run/case |

`k=0` 是逐 token baseline。`k=1` 已在前置 smoke/screening 中验证功能，但没有进入耗时最长的主矩阵；正式选择聚焦于吞吐更有竞争力的 `k=2/3`。

### 4.2 完整结果

| k | concurrency | Output tok/s | TTFT p95 (ms) | TPOT p95 (ms) | 接受率 | Preemptions |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 7.837 | 135.4 | 129.1 | — | 0 |
| 0 | 2 | 15.012 | 152.1 | 134.7 | — | 0 |
| 0 | 4 | 27.589 | 304.5 | 145.9 | — | 0 |
| 0 | 6 | 33.302 | 535.3 | 215.9 | — | 0 |
| 0 | 8 | 34.926 | 2,428.9 | 365.7 | — | 0 |
| 2 | 1 | 16.006 | 146.0 | 84.4 | 86.10% | 0 |
| 2 | 2 | 28.306 | 272.6 | 94.0 | 86.21% | 0 |
| 2 | 4 | 48.875 | 512.9 | 88.9 | 86.81% | 0 |
| 2 | 6 | 39.583 | 2,581.2 | 296.8 | 86.63% | 0 |
| 2 | 8 | 38.703 | 13,349.2 | 506.7 | 87.86% | 0 |
| 3 | 1 | 18.504 | 152.0 | 64.8 | 82.44% | 0 |
| 3 | 2 | 27.479 | 283.6 | 103.4 | 75.88% | 0 |
| 3 | 4 | 44.937 | 524.4 | 107.6 | 75.64% | 0 |
| 3 | 6 | **52.603** | 800.6 | 152.0 | 75.86% | 0 |
| 3 | 8 | 30.315 | 13,834.9 | 523.7 | 76.76% | 0 |

### 4.3 解读

- `c=1` 时，调度和 batch 计算都很轻，`k=3` 能最大化一次主模型验证得到的 token 数。
- `c=2/4` 时，`k=2` 更均衡。`k=3` 的第三个草稿位置接受收益不足以抵消额外 verify 成本。
- `c=6` 时，更大的 batch 改善了 MTP verify 的 GPU 利用率，`k=3` 再次成为吞吐峰值。
- `c=8` 虽然没有 preemption，但所有模式的 TTFT/TPOT 都显著恶化。容量瓶颈已经解除，此时暴露的是执行形状、调度预算和 kernel 效率问题。

建议把“可驻留请求数”和“每轮可运行请求数”拆开：state capacity 可以保留 8，但在线 scheduler 的 runnable cap 暂设为 4；追求离线最大吞吐时使用 6。

## 5. Token budget 扫描

预算扫描使用 `input=128, output=32, requests=16`，扫描 `k=2/3` 与 `c=4/6/8`。不同预算各自独立初始化 engine 并预热对应 active shapes。

### 5.1 Budget = 128

| k | c | Output tok/s | TTFT p95 (ms) | TPOT p95 (ms) | 接受率 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 14.882 | 7,938.6 | 622.8 | 82.47% |
| 2 | 6 | 14.057 | 13,364.9 | 523.0 | 83.59% |
| 2 | 8 | 10.505 | 14,599.3 | 966.2 | 83.85% |
| 3 | 4 | 29.842 | 2,050.9 | 223.1 | 75.16% |
| 3 | 6 | 35.233 | 2,497.0 | 234.0 | 76.52% |
| 3 | 8 | 19.409 | 7,140.3 | 652.2 | 73.66% |

128 对 mixed/prefill + verify 太紧，调度轮数增加，整体吞吐和尾延迟都不理想。

### 5.2 Budget = 256

| k | c | Output tok/s | TTFT p95 (ms) | TPOT p95 (ms) | 接受率 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 37.494 | 600.1 | 119.0 | 79.00% |
| 2 | 6 | 22.611 | 4,699.1 | 374.4 | 81.12% |
| 2 | 8 | 30.718 | 5,012.9 | 395.3 | 82.99% |
| 3 | 4 | 41.148 | 610.6 | 106.8 | 71.66% |
| 3 | 6 | **51.061** | 733.1 | 126.8 | 70.63% |
| 3 | 8 | 17.608 | 7,940.0 | 652.6 | 73.21% |

256 明显优于 128，并再次给出 `k=3,c=6` 的吞吐峰值。

### 5.3 Budget = 512

| k | c | Output tok/s | TTFT p95 (ms) | TPOT p95 (ms) | 接受率 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 19.531 | 13,655.7 | 526.0 | 79.29% |
| 2 | 6 | 18.310 | 13,783.6 | 523.5 | 81.54% |
| 2 | 8 | 24.011 | 14,037.9 | 556.3 | 81.28% |
| 3 | 4 | 11.818 | 13,784.9 | 585.6 | 73.25% |
| 3 | 6 | 36.081 | 1,644.6 | 253.9 | 70.66% |
| 3 | 8 | 20.820 | 13,891.2 | 542.6 | 76.79% |

512 的所有点都低于相应的最佳 256 点，且多数 case 的 TTFT p95 超过 13 秒。说明“更大的 token budget”没有自动转化为更高利用率，反而触发了代价更高的 active shape/编译或调度路径。因为 512 已被 256 支配，本轮按顺序搜索原则早停，没有继续测试 1024。

## 6. 输出一致性观察

主实验记录了 baseline 与 MTP 的 position-wise token agreement 和完整请求一致率：

| concurrency | 吞吐优胜 k | k=2 token agreement | k=3 token agreement | k=2 完整请求一致率 | k=3 完整请求一致率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 53.12% | 53.12% | 3.12% | 3.12% |
| 2 | 2 | 71.88% | 69.27% | 6.25% | 6.25% |
| 4 | 2 | 68.75% | 68.75% | 6.25% | 6.25% |
| 6 | 3 | 76.04% | 84.64% | 6.25% | 15.62% |
| 8 | 2 | 69.27% | 76.04% | 6.25% | 6.25% |

这组数字必须谨慎解释：自回归生成中只要某一步发生一个 token 分歧，后续上下文就不同，position-wise agreement 会把一次早期分歧放大成整段差异。已定位的一个 `k=1` 样本在前 7 个输出 token 完全相同，第 8 个 token 才出现 `198` 与 `271` 的差异，两者都位于换行/思考模板附近。另一个短 prompt 上 `k=1/2/3` 与 baseline 完全一致。

当前更可能的来源是 W4 GEMM 随 active batch 的 `M` 形状改变而采用不同 tile/累加顺序，近似并列的 argmax 被 BF16/FP32 舍入扰动，而不是 MTP state rollback 的结构性错误。但这仍只是工程假设，尚未完成 logit 级证明。

因此：

- 本报告的 `k` 选择基于性能，不把 token exactness 当成语义质量结论。
- 在默认开启 MTP 前，需要固定质量集，报告 exact match、语义/任务指标、首个分歧位置和该位置 top-2 logit margin。
- 后续还应让 baseline 与 MTP 尽可能走相同的 W4 kernel accumulation/layout，减少由 batch shape 引入的数值漂移。

## 7. 推荐配置

### 在线低延迟默认值

```text
state/request capacity = 8
effective runnable concurrency = 4
speculative k = 2
max_num_batched_tokens = 256
```

该配置在主实验达到 48.875 tok/s，TTFT p95 512.9 ms，TPOT p95 88.9 ms。额外请求应在 EngineClient/EngineProc 队列中等待，不要一次把 8 条都变为 GPU runnable。

### 离线或吞吐优先

```text
active concurrency = 6
speculative k = 3
max_num_batched_tokens = 256
```

该配置达到本轮峰值 52.603 tok/s。它的尾延迟高于在线甜点位，但吞吐更高。

### 可选自适应策略

当前数据支持一个简单的 active-count policy：

```text
active=1       -> k=3
active=2..4    -> k=2
active=5..6    -> k=3
active>=7      -> 不继续扩大 runnable batch，先排队
```

这应作为后续实验策略，而不是立即硬编码的最终策略；需要用更多 prompt、上下文长度和至少 3 次重复验证边界。

## 8. 局限与下一步

本轮结果的适用范围：单张 4090 D、Qwen3.6-27B GPTQ W4A16、BF16 KV、128-token 输入、32/64-token 输出、greedy、eager。它尚未覆盖：

- 多种上下文长度和真实生产请求分布。
- 在线 HTTP/ZMQ 网络开销与请求到达过程。
- CUDA Graph 模式。
- FP8 KV Cache 与 MTP 组合。
- 三次以上独立重复和置信区间。
- 正式语义/任务准确率。

建议下一步按顺序做：

1. 把 8 个 state slots 与 4/6 个 runnable slots 显式拆成两个调度参数。
2. 增加 active-count adaptive `k`，先只记录 shadow decision，再开启执行。
3. 用多个 input/output length、真实 prompt JSONL 做三次重复，输出均值与置信区间。
4. 对首个分歧位置保存 baseline/MTP logits、top-2 margin 和 kernel shape，定位 W4 数值漂移。
5. 再测试 CUDA Graph、FP8 KV 和在线接口下的 TTFT/TPOT/ITL。

## 9. 可复现命令与原始数据

主实验：

```bash
/root/miniconda3/bin/python tools/bench_mtp_sweep.py \
  --model /root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4 \
  --mtp-model /root/autodl-tmp/huggingface/Qwen3.6-27B-mtp \
  --k 0,2,3 --concurrency 1,2,4,6,8 \
  --input-len 128 --output-len 64 --num-requests 32 \
  --max-num-batched-tokens 256 --max-num-seqs 8 \
  --output-json results/mtp-steady-i128-o64-r32-b256.json \
  --no-print-json
```

预算扫描只需把 `--k` 改为 `2,3`，把并发改为 `4,6,8`，`output-len/num-requests` 改为 `32/16`，分别设置 `--max-num-batched-tokens 128/256/512`。

版本化原始数据：

- `results/mtp-steady-i128-o64-r32-b256.json`
- `results/mtp-budget-i128-o32-r16-b128.json`
- `results/mtp-budget-i128-o32-r16-b256.json`
- `results/mtp-budget-i128-o32-r16-b512.json`
