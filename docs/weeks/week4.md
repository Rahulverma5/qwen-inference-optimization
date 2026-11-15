# Week 4: Productionize the service

**Goal:** Put the Week 3 ship decision (GGUF Q4_K_M on `llama-server`, one request in flight per instance) behind a production gateway, package it as containers, monitor it, and break it on purpose to check it fails safely.

**Status: done.** The gateway adds no measurable latency (−1.6 ms at p50, within noise). Under 3× overload its 429 back-pressure keeps the requests it accepts at a **7.3 s median, versus 100 s** with an unbounded queue, at the same throughput. Every failure drill passed after one bug fix: a client that gave up used to keep its backend slot busy for 27 s; now it's freed in 0.09 s. A real crash recovers on its own in 8.8 s.

### Scope change (simplified on purpose)

Kept: the gateway (auth, rate limit, 429 back-pressure, timeouts, streaming, health/ready, JSON logs, metrics, graceful shutdown), Docker images, compose with Prometheus + Grafana, load and failure tests, CI image build.

Dropped or moved, because CPU-only deployment (2026-11-09) made them irrelevant, or they cost time without changing a decision:
- `Dockerfile.vllm` and the HF/vLLM/llama.cpp engine comparison: there's no GPU in the deployment.
- Backend failover (primary → fallback): one backend per instance; scaling is horizontal (more instances), which Week 5's platforms do.
- Locust: [serving/loadtest.py](../../serving/loadtest.py) does the same job and also checks that the SQL is correct.
- Trivy scan and the GHCR push: these move to Week 5, where an image is actually deployed.
- `Dockerfile.llamacpp`: the official image (`ghcr.io/ggml-org/llama.cpp:server-b11610`) with the model mounted is enough locally. Week 5 bakes the model in for Cloud Run.

---

## What I built and why

| Piece | File | Why |
|---|---|---|
| Gateway | [serving/gateway/app.py](../../serving/gateway/app.py) | FastAPI in front of llama-server. `POST /v1/sql` is the product: `{schema, question} → {sql}`, with the prompt built by `finetune.prompt` exactly as in SFT and eval. `/v1/completions` and `/v1/chat/completions` are OpenAI-compatible passthroughs (streaming or not), so the Week 2–3 bench and eval clients, and Week 5's canary, run through the gateway unchanged. |
| Limits | [serving/gateway/limits.py](../../serving/gateway/limits.py) | Token bucket **per API key** (fairness between clients) and **admission** (protects the backend): 1 request in flight, at most 2 waiting, each for at most 6 s. Beyond that the gateway returns 429 with `Retry-After`. |
| Settings | [serving/gateway/config.py](../../serving/gateway/config.py) | All `GATEWAY_*` env vars, so the same image runs in compose, Cloud Run and EC2. Refuses to start without API keys unless auth is disabled explicitly. |
| Metrics | [serving/gateway/metrics.py](../../serving/gateway/metrics.py) | Requests by route and status, e2e latency, TTFT, queue wait (histograms), queue depth, in-flight, rejections by reason, backend errors by kind, retries, readiness. |
| Prompt render without `transformers` | [finetune/prompt.py](../../finetune/prompt.py) `render_generation_prompt` | The gateway image stays at 223 MB with 4 dependencies. A test checks it **byte-for-byte** against the real Qwen3 chat template (thinking off), so the served prompt can't drift from the evaluated one. |
| Images | [serving/docker/Dockerfile.gateway](../../serving/docker/Dockerfile.gateway) | Multi-stage, `python:3.12-slim`, non-root user, pinned deps ([requirements-gateway.txt](../../serving/requirements-gateway.txt)), exec-form `CMD` so uvicorn is PID 1 and receives SIGTERM, Docker `HEALTHCHECK`. |
| Stack | [serving/compose.yml](../../serving/compose.yml) | gateway + llama-server (Week 3 ship flags: `--parallel 1 --ctx-size 2048 --cache-ram 1024 --flash-attn on --metrics`) + Prometheus + Grafana. Healthchecks, `restart: unless-stopped`, `stop_grace_period` longer than the gateway's drain timeout. |
| Dashboard | [serving/grafana/dashboards/gateway.json](../../serving/grafana/dashboards/gateway.json) | Auto-provisioned. 8 panels: traffic by status, p50/p95 vs the 10 s SLO, queue depth/in-flight, rejections by reason, backend errors/retries, queue wait/TTFT, llama prefill/cache/decode tok/s, busy/ready. Every query checked against live Prometheus data. |
| Load test | [serving/loadtest.py](../../serving/loadtest.py) | Real Spider dev questions in random DB order (seeded). Closed loop (capacity) or open loop (Poisson arrivals, i.e. real users). Executes each returned query against the Spider DB, so correctness under load is measured, not assumed. |
| Failure drills | [serving/failure_drills.py](../../serving/failure_drills.py) | Five scripted drills against the running stack; results in [reports/week4/failure_drills.json](../../reports/week4/failure_drills.json). |
| Tests | [tests/test_gateway.py](../../tests/test_gateway.py) | 19 tests against a fake llama-server (httpx `MockTransport`): auth, rate limit, queue full vs queue timeout, timeout → 504, down → 502, retry-once, streaming, mid-stream failure, client disconnect, the prompt render. |
| CI | [.github/workflows/ci.yml](../../.github/workflows/ci.yml) | New `gateway-image` job: build, then smoke-test (liveness 200, readiness 503 with no backend, 401 without a key, runs as non-root), plus `compose config` validation. |

### Key decisions

| Decision | Why |
|---|---|
| **1 in flight, queue of 2, 6 s max wait, then 429** | Week 3: CPU batching buys no throughput, and one slot (FIFO) halves the p95 compared with shared slots. A request that would wait longer than ~6 s would miss the 10 s SLO anyway, so the gateway tells the client right away (`Retry-After`) instead of making it wait to fail. The autoscaler (Week 5) reacts to the load; the queue only absorbs short bursts. |
| Two different 429s: `queue_full` (instant) and `queue_timeout` (after the wait) | A full queue means more waiting won't help. A timeout means it was worth a try. Both appear in the metrics by reason, so the dashboard can show *why* clients are being refused. |
| Rate limit per API key, before admission | One noisy client shouldn't fill the shared queue. The bucket allows bursts of 5, refilling at 1/s. |
| Retry once, only if the connection failed before any byte | Week 3c: llama-server sometimes drops a reused keep-alive socket. Nothing was generated yet, so a retry is safe. Any later failure isn't retried, because that could generate twice. |
| `/health` never calls the backend; `/ready` does | Liveness answers "is the process alive?" (if not, restart it). Readiness answers "can it serve?" (if not, stop routing to it). Mixing them up makes a slow model load restart the gateway over and over. |
| Error *inside* the stream when the backend dies mid-stream | The 200 status line is already sent, so the gateway can't change it. It emits an SSE `error` event and closes, so the client knows the answer is incomplete. |
| Non-streaming requests are cancelled when the client leaves (HTTP 499) | Found by a drill (below). Without it, an abandoned request holds the only slot until the model finishes. |
| Graceful shutdown = uvicorn's drain (30 s) + compose `stop_grace_period: 35s` | On SIGTERM uvicorn stops accepting connections and lets in-flight requests finish. The grace period must be longer than the drain, or Docker SIGKILLs mid-answer. Cloud Run gives 10 s, which covers a ~4–9 s request. |
| Official llama.cpp image `server-b11610` | Week 3 pinned b11600, which has no published image. b11610 is the nearest published build (10 commits later). Same flags, same GGUF. |
| API key logged as an 8-character SHA-256 prefix | Logs can say who sent a request without the key itself ever being written down. |

---

## Results

All measured on this laptop: **Intel i5-10210U** (4 cores / 8 threads, 15 W mobile part), 7.6 GiB RAM with swap full, llama-server at 4 threads. Two local deviations from the ship config: **host prompt cache 256 MiB instead of 1 GiB** (not enough RAM), and the per-key rate limit was raised for the load runs so that it wouldn't mask the queue. Raw results: [reports/week4/](../../reports/week4/).

**This laptop is ~2.5–3× slower than Week 3's Kaggle 4-vCPU box.** llama-server computed prompts at 22 tok/s (Kaggle: 66) and generated at 9.3 tok/s (Kaggle: 24). So absolute latencies here are pessimistic. What carries over to the cloud is the *behaviour*: who gets refused, what the queue does, how failures look. Week 5 measures the real latency.

### 1. Baseline through the gateway (closed loop, 1 at a time, 40 Spider questions)

| Metric | Value |
|---|---|
| Answered | 40 / 40 |
| Throughput | 0.10 req/s (Kaggle box, Week 3: 0.21) |
| Latency p50 / p95 / max | 8.5 / 19.2 / 32.4 s |
| Schema already in the prompt cache (11 of 40) vs not (29) | median **2.3 s** vs 11.1 s |
| Within the 10 s SLO | 58% (this laptop; Week 3's box passed at c=1) |
| Exec accuracy of the answers | 82.5% on this 40-question sample (Week 3, all 1,034: 64.3%; 40 questions is too few to compare, ±12 points) |

The prompt cache is still the biggest lever: a question about an already-seen database is ~5× faster.

### 2. Gateway overhead

Same cached request (`max_tokens=1`), 30 times each: direct to llama-server **104.8 ms** p50, through the gateway **103.3 ms**. The difference (−1.6 ms) is noise. The gateway costs nothing measurable next to a multi-second model call.

### 3. Rate limit, live

10 simultaneous requests from one key (rate 1/s, burst 5; 1 in flight, queue 2): **5 × 429 `rate_limited`** (Retry-After 1 s), **3 × 200** (1 ran, 2 queued), **2 × 429 `queue_full`** (Retry-After 6 s). Exactly the designed split.

### 4. Overload: back-pressure on vs off

Poisson arrivals at 0.3 req/s for 120 s, about 3× this laptop's capacity. Same seed, so the same 41 requests at the same times. "Off" = unbounded queue and wait.

| | Back-pressure ON (queue 2, wait ≤ 6 s) | OFF (unbounded) |
|---|---|---|
| Answered | 16 / 41 (others: 13 `queue_full` + 12 `queue_timeout`, each told within ≤ 6 s) | 41 / 41 |
| Goodput | 0.13 req/s | 0.12 req/s |
| Latency of answered p50 / p95 | **7.3 / 18.5 s** | **100 / 232 s** |
| Answered within 10 s / 20 s | 11 / 15 | 2 / 3 |
| Exec accuracy of the answers | 87.5% | 90.2% |
| Test wall time | 124 s | 349 s (queue drains for 4 min after arrivals stop) |

**Finding:** an overloaded box does the same amount of work either way (~0.12–0.13 req/s). Back-pressure only decides *who waits*. Without it, every user waits ~100 s, far past the point where anyone is still there, so "100% success" is really ~95% failure. With it, 15 users get a usable answer and 25 get a fast "busy, retry in N s" that a client, a load balancer or an autoscaler can act on.

The 18.5 s p95 for accepted requests is up to 6 s of queue plus up to ~13 s of model time on this laptop. On the cloud box (Week 3: 8.6 s p95 model time), a 6 s wait could still push some accepted requests past 10 s. **For Week 5:** set `GATEWAY_MAX_QUEUE_WAIT_S` ≈ SLO − measured model p50, i.e. ~5 s, once the real instance has been measured.

### 5. Failure drills

| Drill | What happened | Verdict |
|---|---|---|
| **Backend SIGKILLed mid-stream** (like an OOM kill) | The client had 7 chunks, then got an in-band `backend_failed_mid_stream` event and a clean end (HTTP 200 already sent). `/ready` → 503 after **0.06 s**. New requests: **502 `backend_unavailable`**. No leaked slot (in-flight back to 0). | ✅ |
| ↳ did Docker restart it? | **No.** `docker kill` counts as a *manual* stop, which restart policies ignore by design. After a manual `start`: ready in 5.9 s. | expected; see next row |
| **Backend process exits by itself** (crash) | `/ready` → 503 after 0.05 s. Docker **restarted it automatically** (restart count +1). Serving again after **8.8 s**; first request 200. | ✅ |
| **Oversized input** | 20k-char schema: **413** from the gateway in 12 ms. 3,900 numbers (7,799 tokens, under the char limit but over the 2,048-token context): llama-server's **400 `exceed_context_size_error`** passed through in 13 ms. No model time spent on either. | ✅ |
| **Client disconnects mid-stream** | Slot freed **0.07 s** after the client hung up; next request 200 in 0.16 s. | ✅ |
| **Client gives up on a non-streaming request** | **Bug:** the slot stayed busy for **26.8 s**, until the model finished an answer nobody would read, and everyone queued behind it waited. **Fixed** (`call_backend` races the backend call against client disconnect, cancels it, returns 499): now freed after **0.09 s**, and llama-server stopped too (`requests_processing` 0). | ❌ → ✅ |
| **SIGTERM to the gateway during a request** | The in-flight request **finished with 200** 4.8 s after SIGTERM. A new connection during the drain was refused. Gateway exited with code 0 after 5.5 s, ready again 1.8 s after start. | ✅ |

No pre-run predictions this week (they were registered for Weeks 2–3 only). The one result that surprised me was the non-streaming disconnect leak.

---

## How to run

```bash
cp serving/.env.example serving/.env          # set GATEWAY_API_KEYS
docker compose -f serving/compose.yml up -d --build
curl -s localhost:8000/v1/sql -H "X-API-Key: $KEY" \
     -d '{"schema": "CREATE TABLE singer(id int, name text);", "question": "How many singers?"}'
# Grafana: http://localhost:3000 -> "Qwen SQL gateway"; Prometheus: http://localhost:9090
python -m serving.loadtest --key $KEY --n 40 --concurrency 1 --out reports/week4/load/c1.json
python -m serving.loadtest --key $KEY --n 200 --rate 0.3 --duration 120 --seed 1 --out reports/week4/load/overload.json
python -m serving.failure_drills --key $KEY --out reports/week4/failure_drills.json
```

On a small machine, set `LLAMA_CACHE_RAM_MIB=256` in `serving/.env`.

---

## Interview Q&A

**Why return 429 instead of queueing everything?** An overloaded server does the same work either way; the queue only decides who waits. Measured: with an unbounded queue every user waited ~100 s (p50), and only 2 of 41 got an answer within 10 s. With a bounded queue, 15 of 41 got a useful answer and the rest heard "busy, retry in N s" within 6 s, a signal a client can back off on and an autoscaler can scale on.

**Liveness vs readiness?** Liveness: is the process alive? It fails → restart it. Readiness: can it serve right now (model loaded, backend reachable)? It fails → stop routing traffic but don't restart. If a readiness-style check is used for liveness, a 60 s model load gets the container killed in a loop.

**How do you shut down without dropping requests?** Stop accepting new connections, let in-flight requests finish within a drain timeout, then exit. The orchestrator's grace period must exceed the drain timeout. Tested: a request in flight at SIGTERM finished with 200.

**What can't you fix after streaming has started?** The status code: it's already sent. So mid-stream failures are signalled in-band (an SSE error event), and the client must treat a stream without `[DONE]` as incomplete.

**A bug you found by testing failure paths?** Non-streaming requests weren't cancelled when the client disconnected. With one backend slot, one impatient user blocked everyone for ~27 s. Fixed by racing the backend call against client disconnect.

**When is it safe to retry an LLM request?** Only when nothing could have been generated, i.e. the connection failed before any response byte. Retrying later can bill or act twice.

**Spoken version (30 s):** "I put the quantized model behind a FastAPI gateway with API keys, per-key rate limits, and a bounded queue that returns 429 with Retry-After. Under 3× overload that kept accepted requests at a 7-second median instead of 100 seconds, at the same throughput. I containerized it with Prometheus and Grafana, and ran failure drills: backend killed mid-stream, crash and restart, oversized prompts, client disconnects, SIGTERM drain. One drill found that abandoned requests held the only model slot for 27 seconds, and I fixed it."
