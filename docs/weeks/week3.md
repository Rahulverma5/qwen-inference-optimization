# Week 3: Compression and serving optimization

**Goal:** Make the Week 1 model cheaper to serve without making it worse at SQL. Compress it four ways and measure what each costs in accuracy. Then change one vLLM serving knob at a time against an fp16 baseline re-measured in the same session. The result is an experiment matrix (speed × quality × memory × cost) and a Pareto plot of what's worth shipping.

**Status: code built and unit-tested (52 tests); Kaggle runs pending.** As in Week 2, predictions are written down *before* the run.

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
| Kaggle notebooks | [week3a](../../notebooks/week3a_kaggle.ipynb), [week3b](../../notebooks/week3b_kaggle.ipynb) | **3a** quantizes and evaluates quality, and keeps the models as kernel output. **3b** mounts them (`--kernel-source`) and runs the serving experiments. Split so a serving failure doesn't cost the 1–2 h of quantization. |

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

**Speed** (spider and short at the 4 levels; fp16 c=1 TPOT ≈ 16.4 ms):
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

**Shipping guess:** W8A8 + prefix caching (on by default) + ngram speculation at low load. Replicas, not TP, when a second GPU is available. W4A16 only if memory, not latency, is the constraint.

---

## How to run

```bash
# 3a: quantize + quality (~2.5-3 h of T4 time)
python scripts/kaggle_run.py run --week 3 --part a --attach models/week1_run3/sft/adapter=qwen-sql-week1-adapter
# 3b: serving experiments on 3a's models (~3 h)
python scripts/kaggle_run.py run --week 3 --part b --kernel-source qwen-sql-week3a
# results land in reports/week3/kaggle_a/ and reports/week3/kaggle_b/
```

Locally (CPU), from fetched results: `python -m bench.matrix --experiments reports/week3/kaggle_b/results/experiments --quality reports/week3/kaggle_a/results reports/week3/kaggle_b/results --out reports/week3/report`.

Total ≈ 6 GPU-hours of the 30/week. Each part fits one 12 h session with margin.

---

## Results

*To fill from the two runs: variants table, serving table, Pareto plot, predictions vs measurements, ship decision.*

---

## Interview Q&A (theory; numbers to come from the run)

**Why doesn't 4-bit quantization give 4× faster decode?**
Decode reads every weight once per step, so the ideal is bytes saved. But (1) the tied embedding/lm_head stays fp16 and is read each step, (2) on a T4 there are no Marlin kernels, so Exllama dequantizes on CUDA cores and that work grows with batch, and (3) attention's KV read doesn't shrink at all. Week 2 showed it dominates once context adds up.
*Spoken:* "Weight quantization only shrinks one term of the step time. On T4 the dequant kernel isn't free, the LM head stays in fp16, and attention cost doesn't change, so 4-bit buys maybe 1.3× at batch 1 and nothing at high batch."

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
