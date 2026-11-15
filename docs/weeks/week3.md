# Week 3: Compression and serving optimization

**Goal:** Make the Week 1 model cheaper to serve without making it worse at SQL. Compress it four ways and measure what each costs in accuracy. Then change one serving knob at a time against a baseline, and end with an experiment matrix (speed × quality × memory × cost) and a Pareto plot of what's worth shipping.

**Status: done.** GGUF Q4_K_M is 3.1× smaller than fp16 with no measurable accuracy loss: +0.2 points through vLLM (p = 0.92), and −0.4 points when served live by llama.cpp on CPU (p = 0.76, all 1,034 questions). That's the deployment format. One 4-vCPU box serves it at **~0.2–0.3 req/s within a 10 s p95**, i.e. one request at a time. A host-RAM prompt cache is the only knob worth 40%. As in Week 2, predictions are written down *before* each run.

### Plan change (2026-11-09): CPU serving replaces GPU serving

The deployment target has no GPU: free cloud tiers (GCP trial, AWS free plan) offer none. So the vLLM-on-T4 serving experiments (**part b**) were cancelled. Kaggle's CLI can't stop a queued run, so it ran anyway, on the pre-fix commit `71dc390` (no AWQ). Its results are below as a GPU reference; they don't drive any deployment decision. The serving study now runs the GGUF model on `llama-server`, on a Kaggle CPU-only notebook (**part c**: 4 vCPUs, no GPU quota). That machine is closer to a Cloud Run or EC2 instance than to a T4. The part b code (`configs/experiments.yaml`, `bench/experiments.py`) stays in the repo.

---

## What I built and why

| Piece | File | Why |
|---|---|---|
| Quantization variants | [configs/quant.yaml](../../configs/quant.yaml), [quantize/compress.py](../../quantize/compress.py) | llm-compressor `oneshot` with AWQ W4A16 (asymmetric), GPTQ W4A16 (symmetric) and SmoothQuant + GPTQ W8A8-INT8. Recipes live in YAML, so a variant is a config change. Output is compressed-tensors, which vLLM loads with no flag. |
| Calibration set | [quantize/calibration.py](../../quantize/calibration.py) | 256 Spider **train** rows rendered exactly like the SFT rows (a test asserts byte-identity). Scales are fit on the traffic the model serves. Dev is never touched, so dev accuracy stays held out. |
| GGUF export | [quantize/gguf.py](../../quantize/gguf.py) | llama.cpp at a pinned tag (b11600), CPU-only build, HF → f16 → Q8_0 / Q4_K_M. The converter uses the `gguf-py` from the same checkout, not vLLM's pip `gguf`. |
| Perplexity | [evaluation/perplexity.py](../../evaluation/perplexity.py) | Through vLLM prompt logprobs, so fp16, compressed-tensors and GGUF all go through one engine. Two corpora: WikiText-2 (general damage) and **gold SQL given the serving prompt** (task damage, scored on completion tokens only). |
| Significance | [evaluation/compare.py](../../evaluation/compare.py) | Exact McNemar p-value on the flip matrix vs fp16. A −0.4-point change means nothing without it: Week 1 measured noise at ±0.3. |
| Remote eval | [evaluation/remote.py](../../evaluation/remote.py) | Spider accuracy against any OpenAI-compatible server: the GGUF on llama.cpp CPU here, the deployed endpoint's canary in Week 5. |
| Experiment plan | [configs/experiments.yaml](../../configs/experiments.yaml) | 19 experiments, each one flag (or one model) away from `fp16`, each with the question it answers. |
| Experiment runner | [bench/experiments.py](../../bench/experiments.py) | Starts the servers on the right GPUs (TP=2 or two replicas), runs the Week 2 client per workload, and stops the whole process group. A config that won't start on sm75 is recorded with its log tail; that's a result, not a crash. Re-runs skip finished experiments. |
| Matrix + Pareto | [bench/matrix.py](../../bench/matrix.py) | Variant table (accuracy, Δ, McNemar p, perplexity, size, load time), serving table (SLO capacity, $/1M tokens, **$/1k queries**, KV tokens, spec acceptance), every point vs fp16 at equal load, and the accuracy-vs-cost Pareto frontier. |
| Harness extensions | [bench/client.py](../../bench/client.py), [bench/server_metrics.py](../../bench/server_metrics.py), [bench/server.py](../../bench/server.py) | Round-robin over replicas with summed server metrics, spec-decode counters, weight memory/load-time parsing, JSON flags (`--speculative-config`), and per-level request counts so c=1 doesn't take minutes. |
| Mixed workload | `mixed_v1` in [configs/workloads.yaml](../../configs/workloads.yaml) | 20% 2k-token prompts among 128-token ones, for chunked prefill. The extra random draw only happens for this spec, so v1 workload files stay byte-identical (tested). |
| Kaggle notebooks | [week3a](../../notebooks/week3a_kaggle.ipynb), [week3b](../../notebooks/week3b_kaggle.ipynb), [week3c](../../notebooks/week3c_kaggle.ipynb) | **3a** quantizes and evaluates quality, and keeps the models as kernel output. **3b** mounts them for GPU serving (cancelled, ran anyway). **3c** mounts the GGUFs on a CPU-only machine. Split so a serving failure doesn't cost the 1–2 h of quantization. |
| CPU serving plan + runner | [configs/cpu_experiments.yaml](../../configs/cpu_experiments.yaml), [bench/cpu_experiments.py](../../bench/cpu_experiments.py) | 10 llama-server experiments, one flag each vs Q4_K_M: quant level, threads (2-vCPU shape), slots, prompt cache, host-RAM cache, n-gram speculation, flash attention, KV q8_0. A **fresh server per load level**, the only reset that clears both the slot caches and the host cache, so no level warms the next (Week 2's leakage bug can't happen by construction). Also measures live Spider accuracy of a config. Same result layout as the GPU runner, so `bench.matrix` reads both. |

### Key decisions

| Decision | Why |
|---|---|
| AWQ asymmetric, GPTQ symmetric | Each algorithm in its standard form. Both run on the same T4 kernel (Exllama), so a speed gap between them isolates the zero-point cost. |
| W8A8 with **dynamic** per-token activation scales | Static scales would need calibrated activation ranges, and the SQL prompts' outliers vary per schema. Dynamic costs one reduction per token and removes that failure mode. |
| `lm_head` not quantized | Qwen3-1.7B ties it to the embedding (Week 2: 3.2 GiB of weights, not 3.8). Quantizing it hurts most per byte saved, but it stays fp16 and is read every step: ~0.58 GiB of every W4A16 step. |
| GGUF quality measured through vLLM's GGUF loader; Q4_K_M also on llama.cpp CPU (200 questions) | vLLM gives all 1,034 questions in minutes with the same prompts and engine. llama.cpp is what Week 5 deploys, so its numbers are the deployment's. If the two disagree, that's a kernel/numerics finding. |
| Baseline re-measured in session, not copied from Week 2 | The client changed (prefix-cache reset per level), and Kaggle machines differ run to run. Comparisons are only made within one session. |
| `min(file, max(40, 4 × c))` requests per level | Stable p95s at high load (≥ 4 waves) without spending 5 minutes on c=1. Every config sees the same first prompts. |
| Cost at an **on-demand T4 price ($0.35/GPU-h, GCP list, GPU only)** | Kaggle is free, so cost compares configs, not bills. Priced at the SLO-capacity throughput, not the peak: you can't sell throughput that breaks the SLO. |

### T4 (sm75) facts checked in the pinned versions' source, before spending GPU hours

| Question | Answer (vLLM 0.19.0 / llm-compressor 0.10.0.4) |
|---|---|
| Which kernel runs W4A16 on T4? | **Exllama** (min capability 6.0, fp16 activations, supports zero points). Marlin, Machete, AllSpark and Conch all need sm80+. |
| W8A8-INT8? | CUTLASS int8 `scaled_mm`; the compressed-tensors W8A8 scheme's minimum is 7.5 ("turing and up"). |
| fp8 KV cache with TRITON_ATTN? | The backend lists `fp8`, `fp8_e4m3`, `fp8_e5m2`, but Triton has no e4m3 conversions below sm89, so e4m3 should fail and e5m2 may work. Both are run. |
| Speculative decoding methods | `ngram`, `draft_model`, `suffix`, EAGLE/MTP (need trained heads). `draft_model` with Qwen3-0.6B works because the tokenizer is shared. |
| llm-compressor vs the pinned stack | Requires torch ≤ 2.10.0 and transformers ≤ 4.57.6, exactly what's pinned. Has AWQ and SmoothQuant mappings for `Qwen3ForCausalLM`. |

---

## Predictions (written before the run)

Week 2's model of a decode step: **TPOT ≈ 17.7 ms + 3.2 ms per 1k tokens of batch context**. The first term is the weight read, the second the attention kernel reading the KV cache at ~11% of peak bandwidth. Most predictions follow from which term a change touches.

**Size** (weights on disk; Week 2 measured fp16 at 3.2 GiB):

| Variant | Predicted | Reasoning |
|---|---|---|
| AWQ / GPTQ W4A16 | ~1.3 GiB (2.5×) | 1.4B non-embedding params × ~4.25 bits (group-128 scales) ≈ 0.7 GiB, + 0.58 GiB fp16 tied embedding |
| W8A8-INT8 | ~1.9 GiB (1.7×) | 1.4B × 1 byte + embedding |
| GGUF Q8_0 / Q4_K_M | ~1.8 / ~1.1 GiB | GGUF also quantizes the embedding |

**Accuracy** (fp16 = 64.6%; noise ±0.3 points):
- W8A8 and Q8_0: **within noise** (≤ 0.5 points, McNemar p > 0.05).
- W4A16 (AWQ, GPTQ) and Q4_K_M: **−0.5 to −2 points**. Small models lose more to 4 bits than 7B+ ones, and SQL is unforgiving: one wrong identifier is a wrong answer. AWQ vs GPTQ: no confident call; within 1 point of each other.
- Spider-SQL perplexity separates variants more clearly than WikiText. Calibration was on SQL, so WikiText may *rise* more than SQL perplexity does.
- fp8 e5m2 KV: **measurable loss possible (−0.5 to −2)**. Two mantissa bits is coarse for keys, and attention logits amplify key error.

**GPU speed** (part b; spider and short at the 4 levels; fp16 c=1 TPOT ≈ 16.4 ms):
- **W4A16:** the weight read shrinks ~2.5× → c=1 TPOT **11–14 ms**, not ~7 ms. Exllama dequantizes on CUDA cores and the fp16 lm_head is still read every step. At c=64 W4A16 is **no faster, possibly slower** than fp16: dequant work scales with batch, the bandwidth saving doesn't. Nothing changes on long_v1, where attention dominates.
- **W8A8:** c=1 TPOT ~12–14 ms. Best of the three at high batch (INT8 tensor cores, 2× fp16 peak). This is the variant to beat.
- **attn_flex:** not faster than TRITON_ATTN for decode on sm75 (FlexAttention also compiles to Triton). Low confidence.
- **eager (no CUDA graphs):** c=1 TPOT **+30–60%**. Launch overhead for ~28 layers × several kernels is a big share of a 16 ms step at batch 1, and shrinks at high batch.
- **kv_fp8_e5m2:** reported KV capacity **~2× (≈170k tokens)**. long_v1 TPOT improves only if the attention kernel's cost is bytes, not dequant math. Expect a small gain (≤ 20%), possibly none, because Triton converts fp8 in software on sm75. No effect on spider (11% KV use).
- **kv_fp8_e4m3:** server fails to start (Triton fp8e4nv unsupported below sm89).
- **util_0.5:** KV ≈ 0.5 × 15 − 3.2 − 1.07 ≈ 3.2 GiB ≈ 30k tokens, less than long_v1 c=16 needs (~37k), so **preemptions appear** and throughput drops.
- **capacity_awq_kv_fp8:** KV ≈ 0.9 × 15 − 1.3 − 1.07 ≈ 11 GiB at 56 KiB/token ≈ **200k tokens ≈ 48 × 4k**, so the README's 32 × 4k target fits *in memory*. Week 2 already says it won't meet a latency SLO at that context (attention term).
- **prefix_cache_off:** spider shares 91.6% of each prompt per DB. **TTFT p95 +50–150%** at c ≥ 4. Throughput only −5–15%, because spider prompts are ~300 tokens, so prefill is a small part of each request.
- **max_seqs_16:** at c=64 TPOT stays near its c=16 value while TTFT p95 jumps by seconds (requests queue instead of joining the batch). SLO capacity unchanged or lower.
- **Chunk size (mixed_v1):** 512 gives the best ITL p99 for short requests and the worst TTFT for long ones; 8192 the reverse (a 2k prefill takes a whole ~0.9 s step, so every running decode stalls). `chunked_prefill_off` may be refused by the V1 engine.
- **spec_ngram (spider):** SQL copies table and column names from the schema in the prompt, so prompt-lookup drafts should hit often. **Acceptance 40–60%, c=1 TPOT −20–35%.** At c=64 it hurts (+5–20% TPOT): the batch is already big, and verifying rejected drafts takes compute from real tokens.
- **spec_draft_0.6b:** **worse than ngram.** The stock 0.6B doesn't know the fine-tuned SQL style, and its own weight read (~1.2 GiB) costs ~35% of a target step. Small gain at c=1, loss from c=16.
- **tp2 vs replicas2:** **replicas win on spider throughput** (~2× fp16 at SLO, at 2× the cost, so equal $/query), and also on cost/query vs TP2. TP2 wins c=1 latency (weights and KV split: ~10–12 ms TPOT; PCIe all-reduce of 2 × 28 small messages per step adds ~2–3 ms) and long_v1 (attention halved per GPU).

**Shipping guess (GPU):** W8A8 + prefix caching (on by default) + ngram speculation at low load. Replicas, not TP, when a second GPU is available. W4A16 only if memory, not latency, is the constraint.

### CPU predictions (part c, written before its run)

Prior evidence: part a's llama.cpp eval on the same Kaggle 4-vCPU machine logged **~4.2 decode tok/s per slot with 4 slots busy (~17 tok/s total)** and **15–27 tok/s per slot for uncached prefill**. In that eval 92.5% of requests computed < 100 prompt tokens: dev questions arrive grouped by database, so a slot's cached schema got reused. `spider_v1` samples databases at random, so it gets far fewer cache hits.

- **llama-bench:** prefill (pp512) ~60–100 tok/s at 4 threads, near-linear from 1 to 2 threads and flatter from 2 to 4 (4 vCPUs are likely 2 physical cores with hyper-threading). Decode (tg128) ~10–15 tok/s at 4 threads, flattening early (bandwidth-bound).
- **Baseline SLO capacity:** an uncached spider request is ~300–900 prompt tokens at ~60–100 tok/s (3–15 s) plus ~25 tokens at ~10 tok/s (2.5 s). **The 10 s p95 SLO fails or barely passes even at c=1.** One 4-vCPU box serves ≲ 0.3 req/s within SLO. The SLO stays as registered; a miss is the finding.
- **q8_0:** decode ~1.4× slower (1.7× the bytes, partly offset by cheaper dequant); prefill about equal. Accuracy identical (−0.2, p 0.81, from part a).
- **threads_2:** prefill ~1.6–2× slower (compute-bound), decode only ~1.2× slower. Real cloud 2-vCPU shapes will look like this.
- **parallel_1 vs 4 vs 8:** on CPU there is little idle compute to batch into. 4 slots give **≤ 1.5×** the throughput of 1 at much worse per-request latency; 8 slots no more throughput than 4.
- **no_prompt_cache:** small effect on `spider_v1` (random DB order, few slot hits). It would be large on database-grouped traffic.
- **host_cache_1g: the biggest win.** 1 GiB holds ~20 schemas of ~400 tokens × 112 KiB, i.e. every dev database. After the first request per DB, prefill drops to the ~25-token question: **E2E −50% or more**.
- **spec_ngram:** CPU decode is bandwidth-bound like a GPU's at batch 1, so verifying 4 drafted tokens costs little more than 1. **Acceptance ~40–60%, decode +20–40% faster at c=1**, a smaller gain at c=8.
- **flash_attn_off:** within ±10% (short contexts).
- **kv_q8_0:** within ±10% speed. Halves KV memory; that matters only for a < 2 GiB instance.

---

## How to run

```bash
# 3a: quantize + quality (~2.5-3 h of T4 time)
python scripts/kaggle_run.py run --week 3 --part a --attach models/week1_run3/sft/adapter=qwen-sql-week1-adapter
# 3c: CPU serving study on 3a's GGUFs (CPU-only notebook, no GPU quota; ~4 h)
python scripts/kaggle_run.py run --week 3 --part c --cpu --kernel-source qwen-sql-week3a
# results land in reports/week3/kaggle_a/ and reports/week3/kaggle_c/
```

Locally, from fetched results: `python -m bench.matrix --plan configs/cpu_experiments.yaml --experiments reports/week3/kaggle_c/results/experiments --quality reports/week3/kaggle_a/results reports/week3/kaggle_c/results --out reports/week3/report`.

---

## Results

### Part a: compression and quality

Commit `c9b6f56`, 2× Tesla T4, vLLM 0.19.0, llm-compressor 0.10.0.4, llama.cpp b11600. Kaggle queued the run for ~2 h, so a second copy (`-r2`) was pushed. Both eventually ran, which turned an accident into a reproducibility test. Raw results: [run 1](../../reports/week3/kaggle_a/results/), [run 2](../../reports/week3/kaggle_a_r2/results/).

| Variant | Size GiB (× smaller) | Exec acc, run 1 / run 2 | Δ vs fp16 (McNemar p) | WikiText PPL Δ | Spider-SQL PPL Δ |
|---|---|---|---|---|---|
| fp16 | 3.20 (1.0×) | 64.7% / 64.7% | — | 15.91 | 1.386 |
| **GGUF Q4_K_M** | **1.03 (3.1×)** | 64.9% / 64.9% | **+0.2 (0.92)** | +8.3% | +4.0% |
| GGUF Q8_0 | 1.71 (1.9×) | 64.5% / 64.5% | −0.2 (0.81) | +0.3% | +0.4% |
| W8A8-INT8 | 1.89 (1.7×) | 64.0% / 63.1% | −0.7 (0.51) / −1.6 (0.053) | +0.9% | +0.1% |
| GPTQ W4A16 | 1.26 (2.6×) | 62.4% / 63.2% | −2.3 (**0.025**) / −1.5 (0.14) | +17.5% | +0.2% |
| AWQ W4A16 | — | failed (see below) | — | — | — |
| Q4_K_M on llama.cpp CPU, 200 q | — | 56.0% / 52.5% | +1.5 (0.61) / −2.0 (0.42), vs fp16 on the same 200 | — | — |

**Findings**
1. **GGUF Q4_K_M loses nothing measurable at 3.1× compression.** It beats both 4-bit llm-compressor formats here. K-quants use mixed precision (the most sensitive tensors keep 6 bits), and their 32-weight sub-blocks get finer scales than group-128.
2. **llm-compressor results aren't reproducible run to run.** fp16 and GGUF scored identically twice (the eval is deterministic), but GPTQ moved 0.8 points and W8A8 0.9 between identical runs. Cause: `oneshot` shuffles calibration samples with an unseeded `RandomSampler`. Same samples, different order, so different float rounding in the accumulated statistics. That flips rounding decisions for weights near a quantization boundary, and accuracy moves by about one point. **One quantization run can't support a 1-point claim.** Fixed: our fixed-seed sample, `shuffle_calibration_samples=False`, `torch.manual_seed`.
3. **AWQ failed in fp16:** every candidate scale in layer 0's grid search had loss `inf`. The squared output error summed over 138k calibration tokens exceeds fp16's maximum (65,504). GPTQ survived because it accumulates its Hessian in fp32. Fixed by calibrating AWQ in fp32 and casting to fp16 before saving. It wasn't re-run: part b was going to rebuild it, and the CPU deployment doesn't need it.
4. **Perplexity didn't predict SQL accuracy.** GPTQ's Spider-SQL perplexity moved +0.2% while its accuracy fell 1.5–2.3 points. Perplexity averages over every token, and an execution failure comes from one wrong identifier. WikiText moved the most (+17.5%), as predicted: calibration was on SQL.
5. **llama.cpp CPU accuracy varied ±3.5 points between runs** on the same 200 questions. With 4 slots, batch composition changes the float summation order, and 200 questions is a small sample. Part c measures all 1,034.

**Predictions vs measurements (part a)**

| Prediction | Measured | Verdict |
|---|---|---|
| Sizes: W4A16 ~1.3, W8A8 ~1.9, Q8_0 ~1.8, Q4_K_M ~1.1 GiB | 1.26, 1.89, 1.71, 1.03 | ✅ |
| W8A8 and Q8_0 within noise | Q8_0 −0.2 (p 0.81) ✅; W8A8 −0.7 / −1.6 (p 0.51 / 0.053) | ⚠️ borderline in one run |
| W4A16 and Q4_K_M −0.5 to −2 | GPTQ −2.3 / −1.5 ✅; Q4_K_M **+0.2** | ❌ for Q4_K_M: better than predicted |
| SQL perplexity separates variants more than WikiText | the reverse: SQL PPL flat, WikiText +17.5% | ❌ |
| AWQ vs GPTQ within 1 point | AWQ didn't run (fp16 overflow) | — |

**Other costs:** GPTQ took 9.3 min and SmoothQuant+GPTQ 10.6 min to calibrate on one T4. Building llama.cpp took 5.6 min, then 16 s to convert and 1.4 min to quantize to Q4_K_M.

### Part b: GPU serving on T4 (reference only)

Commit `71dc390` (before the AWQ fix, so AWQ and `capacity_awq_kv_fp8` had no model), 1–2× T4, vLLM 0.19.0. Raw results: [kaggle_b](../../reports/week3/kaggle_b/results/). The SLOs are Week 2's (TTFT/TPOT/E2E), and cost is at $0.35 per GPU-hour.

| Config | Max c @SLO | req/s @SLO | $/1k queries | c=1 TPOT p95 | c=64 tok/s |
|---|---|---|---|---|---|
| fp16 (baseline) | 4 | 4.5 | 0.022 | 16.7 ms | 481 |
| **GPTQ W4A16** | **16** | **11.9** | **0.008** | **9.2 ms** | 505 |
| W8A8-INT8 | 4 | 5.4 | 0.018 | 14.7 ms | 513 |
| TP=2 (2 GPUs) | 16 | 14.5 | 0.013 | 9.5 ms | 754 |
| 2 replicas (2 GPUs) | 16 | 11.1 | 0.018 | 17.1 ms | 700 |
| prefix cache off | 4 | 3.8 | 0.026 | 17.3 ms | 230 |
| max-num-seqs 16 | 4 | 4.3 | 0.022 | 17.2 ms | 324 |
| ngram spec (25% accepted) | 4 | 4.6 | 0.021 | 20.2 ms | 480 |
| draft 0.6B spec (43% accepted) | 0 | — | — | 53.3 ms | 264 |
| FlexAttention | 1 | 1.4 | 0.069 | 22.8 ms | 275 |
| eager (no CUDA graphs) | 1 | 0.95 | 0.102 | 33.0 ms | 458 |
| fp8 KV e4m3 / e5m2 | server failed | | | | |

**Findings**
1. **GPTQ is the cheapest GPU config by 2.7×**: c=1 TPOT 16.7 → 9.2 ms, which moves the TPOT SLO from c=4 to c=16. At c=64 it's no faster than fp16 (505 vs 481 tok/s): the dequant work grows with batch. But it costs 1.5–2.3 accuracy points (part a), and W8A8 costs none, so **W8A8 is the quality-safe GPU pick** and GPTQ the cost pick.
2. **Two GPUs: TP=2 beat replicas** on SLO throughput (14.5 vs 11.1 req/s). At batch 1–16 on a 1.7B model, the PCIe all-reduce costs less than halving each GPU's weight read saves, and TP halves per-request latency, which is what the SLO binds on.
3. **Prefix caching is the free lunch:** turning it off cuts c=64 throughput by 52% (481 → 230 tok/s).
4. **Speculation didn't pay.** N-gram accepted 25% (predicted 40–60%) and made c=1 TPOT 20% *worse*. The 0.6B draft accepted 43% but costs a second weight read per step, and took KV space (35k tokens vs 86k).
5. **On sm75, CUDA graphs and the default Triton attention are essential.** Eager doubled c=1 TPOT, and FlexAttention was 1.4× slower at c=1 and 1.7× slower at c=64.
6. **fp8 KV doesn't run on T4.** e4m3 failed as predicted ("fp8e4nv not supported"). e5m2 also failed, on a `torch.compile` Unsupported assertion.
7. **Chunked prefill (mixed_v1):** a 512-token budget cut c=16 ITL p99 from 856 to 349 ms, at +8% long-prompt TTFT. That's the predicted trade-off. No config met mixed_v1's SLO at c ≥ 4.

| Prediction (GPU) | Measured | Verdict |
|---|---|---|
| W4A16 c=1 TPOT 11–14 ms, no gain at c=64 | 9.2 ms; 505 vs 481 tok/s | ✅ (better at c=1) |
| W8A8 best at high batch | 513 tok/s, best single-GPU, by 2% | ✅ (marginal) |
| eager +30–60% c=1 TPOT | +98% | ⚠️ bigger |
| kv e4m3 fails; e5m2 may work | both fail | ⚠️ |
| util_0.5 → ~30k tokens, preemptions | 32k tokens | ✅ |
| prefix off: TTFT +50–150%, throughput −5–15% | c=1 TTFT +68%; c=64 throughput **−52%** | ❌ much bigger |
| spec ngram 40–60% accept, c=1 −20–35% | 25%, +20% | ❌ |
| draft 0.6B worse than ngram | yes | ✅ |
| replicas beat TP2 on throughput | TP2 won | ❌ |

### Part c: CPU serving (the deployment engine)

Commit `4355d82` plus the scoped-glob fix, a Kaggle CPU notebook (AMD EPYC 7B12, **4 vCPUs = 2 cores × 2 hyper-threads**, AVX2, 31 GiB), llama.cpp b11600. 24 requests per level; every level started a fresh server. Raw results: [kaggle_c](../../reports/week3/kaggle_c/results/).

**Quality, served live:** Q4_K_M on llama-server (4 slots), all 1,034 dev questions: **64.3% vs fp16 64.7% (−0.4, McNemar p = 0.76)**. 47 questions were fixed and 51 broken, so ~10% of answers flip while the net stays at noise. Through vLLM's GGUF loader it scored 64.9%. The 0.6-point engine gap is numerics (CPU kernels, batch-dependent summation), not a real loss. The ±3.5-point swing in part a was 200-question sample noise.

**llama-bench (Q4_K_M / Q8_0, tok/s):**

| Threads | Prefill pp512 | Decode tg128 |
|---|---|---|
| 1 | 31.1 / 22.5 | 10.9 / 7.5 |
| 2 | 60.0 / 44.0 | 18.4 / 13.2 |
| 4 | 65.7 / 45.3 | 24.1 / 17.5 |

**Serving.** SLO = e2e p95 ≤ 10 s (spider), 20 s (short); $0.38 per 4-vCPU instance-hour. One problem in the data: 1–3 of 24 requests at some levels failed with `RemoteProtocolError`. llama-server closed a pooled keep-alive connection just as the client reused it, so the request failed before any byte came back. The matrix counts any error as an SLO miss, which zeroes almost every row of [serving.md](../../reports/week3/kaggle_c/results/report/serving.md). The table below gates only on latency. It's the same data, and latencies are over the successful requests. The client now retries such a request once ([bench/client.py](../../bench/client.py)); the Week 4 gateway needs the same retry.

| Config | Max c @SLO | req/s @SLO | $/1k queries | c=1 e2e p50 / p95 | c=1 TTFT p95 | Prompt tokens computed / sent |
|---|---|---|---|---|---|---|
| **q4_k_m** (4 slots, 4 threads) | 1 | 0.21 | 0.51 | 4.3 / 8.6 s | 6.8 s | 185 / 301 |
| q8_0 | **0** | (0.16) | — | 5.8 / 10.8 s | 9.0 s | |
| threads_2 | 1 | 0.19 | 0.56 (0.28 at a 2-vCPU price) | 4.8 / 9.4 s | 6.8 s | |
| parallel_1 | 1 | 0.19 | 0.55 | 4.5 / 8.3 s | 7.2 s | 240 / 301 |
| parallel_8 | 1 | 0.21 | 0.50 | 4.2 / 8.2 s | 6.2 s | |
| no_prompt_cache | 1 | 0.16 | 0.65 | 6.0 / 8.6 s | 7.6 s | 281 / 306 |
| **host_cache_1g** | 1 | **0.30** | **0.36** | **2.4** / 8.0 s | 6.1 s | **111** / 291 |
| flash_attn_off | 1 | 0.21 | 0.50 | 4.3 / 8.2 s | 6.5 s | |
| kv_q8_0 | 1 | 0.19 | 0.55 | 4.8 / 8.9 s | 6.9 s | |
| spec_ngram | server failed: b11600 removed `--draft-max` (now `--spec-ngram-simple-size-m`) | | | | | |

Throughput by concurrency (spider, req/s, c = 1/2/4/8): baseline 0.21/0.22/0.19/0.20; parallel_1 0.19/0.19/0.19/0.20; host_cache_1g 0.30/0.28/0.27/0.27. At c=4, e2e p95 is 46.8 s (baseline), 24.9 s (parallel_1) and 29.8 s (host cache).

**Findings**
1. **One 4-vCPU box is saturated by one request.** Throughput is flat from c=1 to c=8, and every config's e2e p95 jumps from ~8–9 s at c=1 to 20–35 s at c=2. Capacity is ~0.2 req/s (12 queries/min) per instance, so scaling is horizontal: one in-flight request per instance, more instances.
2. **Prefill is the request.** An uncached spider prompt is ~300 tokens at ~65 tok/s ≈ 4.6 s, versus ~25 output tokens at ~22 tok/s ≈ 1.1 s. Everything that skips prefill wins. **The host-RAM cache (1 GiB) is the only big lever:** it cut prompt tokens computed per request from 185 to 111 and median e2e from 4.3 to 2.4 s (−44%), and raised throughput 42%. p95 barely moved (8.6 → 8.0 s), because the tail is the first request to each database, which is always a miss.
3. **Batching buys nothing on CPU.** 1, 4 and 8 slots give the same throughput (0.19–0.21 req/s), because there's no idle compute to batch into. Slots only decide how the queue is shared. With 4 slots, concurrent requests share the CPU (processor sharing) and all finish late. With 1 slot they run FIFO, which nearly halves the p95 at c ≥ 2 (24.9 vs 46.8 s at c=4) with the same mean.
4. **Hyper-threads help decode, not prefill.** Going from 2 to 4 threads gave prefill +9% (60 → 66 tok/s; the two cores' FMA units are already busy) but decode +31% (18 → 24; the extra threads hide memory latency). That's the reverse of what I predicted. So **2 vCPUs keep ~90% of a 4-vCPU box's request throughput** (0.19 vs 0.21 req/s) at half the price, which makes the 2-vCPU shape twice as cost-efficient.
5. **Q8_0 is strictly worse:** 1.38× slower decode, 1.45× slower prefill, misses the SLO at c=1, and buys −0.2 points of accuracy that aren't significant. Q4_K_M stays.
6. **Flash attention and KV q8_0 don't matter here** (±2% and −10%), since contexts are ≤ 1k tokens. Keep FA on and the KV cache at f16.

**Predictions vs measurements (part c)**

| Prediction | Measured | Verdict |
|---|---|---|
| pp512 60–100 tok/s at 4 threads, near-linear 1→2, flat 2→4 | 66; 31 → 60 → 66 | ✅ |
| tg128 10–15 tok/s, flattening early | 24 tok/s, still +31% from 2→4 | ❌ faster, and doesn't flatten |
| 10 s p95 fails or barely passes at c=1; ≲ 0.3 req/s | passes at c=1 only (8.6 s); 0.21 req/s | ✅ |
| q8_0 decode ~1.4× slower, prefill equal | 1.38× / 1.45× | ✅ / ❌ |
| threads_2: prefill 1.6–2× slower, decode 1.2× | prefill 1.09×, decode 1.31× | ❌ reversed (hyper-threading) |
| 4 slots ≤ 1.5× the throughput of 1; 8 no better than 4 | 1.08×; 1.0× | ✅ |
| no_prompt_cache: small effect | −21% req/s, +40% median e2e | ⚠️ bigger: slots keep 4 recent schemas |
| host_cache_1g: biggest win, e2e −50% or more | biggest win; median −44%, p95 −7% | ✅ median / ❌ tail |
| spec_ngram: 40–60% accept, decode +20–40% | not run (renamed flag) | — |
| flash_attn_off, kv_q8_0 within ±10% | −2%, −10% | ✅ |

### Ship decision (Week 4–5)

**GGUF Q4_K_M on llama-server: `--threads <all vCPUs> --parallel 1 --ctx-size 2048 --cache-ram 1024 --flash-attn on --metrics`, with one request in flight per instance** (Cloud Run concurrency 1). The gateway queues or returns 429 beyond that, and autoscaling adds instances. Expect ~0.2–0.3 req/s per instance at a ~8 s p95. Prefer a 2-vCPU instance (≈ same throughput, half the cost) if Week 5's measurement confirms it.

Untested combination: one slot plus the host cache. It's in the plan as `ship_p1_cache1g`, together with the fixed `spec_ngram`, and is worth ~40 min of a free CPU notebook before Week 5 bakes the flags in.

---

## Interview Q&A

**Why doesn't 4-bit quantization give 4× faster decode?**
Decode reads every weight once per step, so the ideal is bytes saved. But (1) the tied embedding/lm_head stays fp16 and is read each step, (2) on a T4 there are no Marlin kernels, so Exllama dequantizes on CUDA cores and that work grows with batch, and (3) attention's KV read doesn't shrink at all. Week 2 showed it dominates once context adds up.
*Spoken:* "Weight quantization only shrinks one term of the step time. On T4 the dequant kernel isn't free, the LM head stays in fp16, and attention cost doesn't change. GPTQ got 1.8× at batch 1 and 5% at batch 64."

**Why doesn't continuous batching help on CPU when it's the main win on GPU?**
Batching amortizes the weight read over many sequences, which pays when a step is memory-bound and compute sits idle. A GPU at batch 1 uses a few % of its FLOPs. On a 2-core CPU, prefill already saturates the FMA units, and prefill is ~80% of a text-to-SQL request. Measured: 1, 4 and 8 slots all gave 0.19–0.21 req/s. More slots only change queueing: processor sharing makes every request late, while FIFO with one slot halves p95.
*Spoken:* "There's no idle compute on a CPU to batch into, so slots just reshuffle the queue. I run one request per instance and scale out."

**AWQ vs GPTQ vs SmoothQuant: what does each actually do?**
GPTQ quantizes weights column by column and uses the inverse Hessian (from calibration activations) to push each column's rounding error onto the columns not yet quantized. AWQ doesn't change rounding: it finds per-channel scales that protect the ~1% of weight channels hit by large activations before rounding. SmoothQuant is for *activation* quantization: it divides activation outlier channels by a scale and multiplies the weights by it, moving the difficulty to the weights, where int8 handles it.
*Spoken:* "GPTQ is error compensation, AWQ is outlier-aware scaling, and SmoothQuant moves activation outliers into the weights so W8A8 is possible at all."

**Why calibrate on Spider train instead of a generic corpus?**
GPTQ minimizes layer-output error *on the calibration activations*, and AWQ/SmoothQuant pick scales from them. Calibrating on WikiText optimizes for text the model never serves. Using train, not dev, keeps the dev accuracy honest.

**How do you know a −0.6-point drop is real?**
Paired test on the same 1,034 questions: exact McNemar on the discordant pairs. Week 1's re-run put noise at ±0.3 points, so a drop is real only if p < 0.05 *and* it exceeds that floor.

**TP=2 or two replicas?**
TP splits each layer across GPUs: per-token latency falls (half the weights and KV per GPU), but every layer pays two all-reduces. Over PCIe without NVLink that overhead is fixed per step. Replicas don't communicate, so throughput doubles and per-request latency doesn't change. Pick TP when the model doesn't fit or single-request latency is the SLO. Pick replicas when you're throughput- or cost-bound, as a 1.7B model on two T4s is.

**When does speculative decoding hurt?**
It trades compute for fewer sequential steps. At low batch the GPU is memory-bound and has idle compute, so verifying k drafted tokens in one step is nearly free. At high batch compute is busy, and rejected drafts waste it. N-gram (prompt lookup) is a natural fit for text-to-SQL, because the answer copies identifiers from the schema in the prompt.
