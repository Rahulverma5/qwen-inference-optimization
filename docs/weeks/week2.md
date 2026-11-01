# Week 2: Benchmark harness and capacity math

**Goal:** Measure how the Week 1 fine-tuned model (fp16, vLLM, one T4) behaves under load: where throughput stops scaling, what latency each load level costs, and how many users one GPU can serve within an SLO. Every Week 3 optimization is reported as a change against these numbers.

**Status: harness built and unit-tested; Kaggle run pending.** Predictions are written down below *before* the run, so the measurements can confirm or refute them.

---

## What I built and why

| Piece | File | Why |
|---|---|---|
| Async streaming load generator | [bench/client.py](../../bench/client.py) | Speaks the OpenAI `/v1/completions` API, so the same client measures vLLM, llama.cpp and the Week 4 gateway. Closed loop (N in flight) gives capacity curves. Open loop (Poisson arrivals) shows queueing, which closed loop hides because the client waits for a reply before sending more. |
| Metric definitions | [bench/metrics.py](../../bench/metrics.py) | TTFT, ITL, TPOT, E2E, p50/90/95/99. TPOT uses the server's token count because one SSE chunk can carry several tokens. |
| Frozen workloads | [bench/workloads.py](../../bench/workloads.py), [configs/workloads.yaml](../../configs/workloads.yaml) | short (128/128), long (2048/256), shared-prefix (1024 of 1536 tokens shared), and Spider (300 real dev questions in the exact serving format, natural output length). Exact token counts verified after re-tokenization; sha256 manifest per run. |
| Server-side sampling | [bench/server_metrics.py](../../bench/server_metrics.py) | Polls vLLM `/metrics` and `nvidia-smi` every second during each load level: KV-cache usage, running/waiting requests, preemptions, prefix-cache hit rate, GPU memory. This explains *why* latency bends, not just that it does. |
| Server launcher | [bench/server.py](../../bench/server.py), [configs/serving.yaml](../../configs/serving.yaml) | vLLM flags live in named profiles, with defaults spelled out. Week 3 ablations override one flag (`--set enable-prefix-caching=false`). Parses vLLM's reported KV capacity from its startup log. |
| KV-cache calculator | [bench/kv_cache.py](../../bench/kv_cache.py) | Predicts capacity from `config.json` before running anything. |
| Capacity report | [bench/report.py](../../bench/report.py) | Knee, max concurrency under SLO, Little's-law check, plots. |
| Kaggle notebook | [notebooks/week2_kaggle.ipynb](../../notebooks/week2_kaggle.ipynb) | Merges the Week 1 adapter, serves on one T4, runs all sweeps, open-loop sweep, and the `vllm bench serve` cross-check. |

### Key decisions

| Decision | Why |
|---|---|
| One T4, TP=1 | This is the unit of capacity. TP=2 vs two replicas is a Week 3 experiment. |
| `ignore_eos` on synthetic workloads, natural length on Spider | Fixed output length compares configs on equal work. Natural length is what production sees. |
| Knee = max of throughput / latency (Kleinrock's power) | A definition that doesn't need eyeballing. Past the knee, extra load buys less throughput than it costs in latency. |
| SLO capacity = highest level where it *and every lower level* pass | One lucky pass above a failing level isn't capacity. |
| SLOs fixed in config before the run | Picking thresholds after seeing the curves would make "meets SLO" meaningless. |
| Client cross-checked with `vllm bench serve` | An independent tool on the same server. If they disagree by more than ~10%, my client is the bottleneck or mismeasures. |

---

## Predictions (written before the run)

**KV cache.** Qwen3-1.7B: 28 layers, 8 KV heads (GQA, vs 16 query heads), head dim 128.
bytes/token = 2 (K, V) × 28 × 8 × 128 × 2 B = **114,688 B = 112 KiB**.
Budget ≈ 0.9 × 15 GiB − 3.8 GiB weights − ~1 GiB activations ≈ 8.7 GiB → **~81k tokens**, so **~19 concurrent 4k sequences, ~4 at 16k**. vLLM reports its own figure at startup; the gap measures my activation guess.

**Decode floor.** At batch 1, each token reads every weight once: ~4.06 GB / 320 GB/s (T4 peak) ≈ **12.7 ms/token**. Expect ~16–20 ms in practice (60–80% of peak bandwidth).

**Why batching is nearly free at first.** One weight read serves the whole batch, so arithmetic intensity is about B FLOP/byte at batch B. The T4's ridge point is ~65 TFLOPS / 320 GB/s ≈ 200 FLOP/byte, so decode stays memory-bound far past any batch the KV cache allows. Expect output throughput to scale nearly linearly with concurrency at first, then bend as per-request KV reads grow and prefills interrupt decode. That bend sets the knee, not compute.

**Workload ordering.** long_v1 knees first (2k-token prefills stall decode; ~40 such sequences fill the cache). shared_prefix_v1 should beat long_v1 at the same length, because prefix caching is on and 2/3 of each prompt is cached.

---

## Results

*To fill from `reports/week2/kaggle/results/`: capacity table, plots, KV predicted vs reported, open-loop curve, cross-check agreement, final SLOs.*
