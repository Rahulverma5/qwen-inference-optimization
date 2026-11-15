"""FastAPI gateway in front of llama-server (OpenAI-compatible, CPU, GGUF Q4_K_M).

    GATEWAY_API_KEYS=dev-key GATEWAY_BACKEND_URL=http://localhost:8080 \
        uvicorn serving.gateway.app:app --port 8000 --timeout-graceful-shutdown 30 --no-access-log

Routes
  POST /v1/sql               {schema, question} -> {sql}: the product endpoint (prompt built here,
                             identical to SFT/eval via finetune.prompt)
  POST /v1/completions       OpenAI-compatible passthrough, streaming or not (bench/eval clients)
  POST /v1/chat/completions  same, chat format
  GET  /health               liveness: the process answers (no backend call)
  GET  /ready                readiness: backend model loaded and reachable
  GET  /metrics              Prometheus

Request path: request id + JSON access log -> API key -> token bucket (per key) -> size check ->
admission (1 in flight, bounded queue, 429 on full/too-long wait) -> backend, with one retry if
the connection failed before any byte (Week 3c: llama-server drops reused keep-alive sockets).

Graceful shutdown is uvicorn's: on SIGTERM it stops accepting connections and waits up to
--timeout-graceful-shutdown for in-flight requests (streams included) before exiting.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from finetune.prompt import build_messages, extract_sql, render_generation_prompt
from serving.gateway.config import Settings
from serving.gateway.limits import Admission, Rejected, TokenBucket
from serving.gateway.metrics import Metrics

ROUTES = {"/v1/sql", "/v1/completions", "/v1/chat/completions", "/health", "/ready", "/metrics"}
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
log = logging.getLogger("gateway")


# ---------- logging ----------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": round(record.created, 3), "level": record.levelname, "msg": record.getMessage(),
               "request_id": request_id_var.get()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging(level: str) -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    log.handlers[:] = [h]
    log.setLevel(level)
    log.propagate = False


class ObservabilityMiddleware:
    """Pure ASGI (not @app.middleware) so a streamed response is timed to its LAST byte."""

    def __init__(self, app, metrics: Metrics):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        rid = headers.get(b"x-request-id", b"").decode()[:64] or uuid.uuid4().hex
        token = request_id_var.set(rid)
        route = scope["path"] if scope["path"] in ROUTES else "other"
        t0, status = time.perf_counter(), [500]

        async def wrapped_send(msg):
            if msg["type"] == "http.response.start":
                status[0] = msg["status"]
                msg["headers"] = [*msg.get("headers", []), (b"x-request-id", rid.encode())]
            await send(msg)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            dt = time.perf_counter() - t0
            if route not in {"/metrics", "/health", "/ready"}:
                self.metrics.latency.labels(route).observe(dt)
            self.metrics.requests.labels(route, str(status[0])).inc()
            log.info("request", extra={"fields": {
                "method": scope["method"], "path": scope["path"], "status": status[0],
                "latency_ms": round(dt * 1000, 1), "key": scope.get("state", {}).get("key_id", "-")}})
            request_id_var.reset(token)


# ---------- app ----------

def create_app(settings: Settings | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)
    metrics = Metrics()
    bucket = TokenBucket(settings.rate_per_s, settings.burst)
    admission = Admission(settings.max_in_flight, settings.max_queue, settings.max_queue_wait_s)
    client = httpx.AsyncClient(
        base_url=settings.backend_url, transport=transport,
        timeout=httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s))

    @asynccontextmanager
    async def lifespan(_app):
        log.info("startup", extra={"fields": {"backend": settings.backend_url,
                                              "max_in_flight": settings.max_in_flight,
                                              "max_queue": settings.max_queue}})
        yield
        await client.aclose()
        log.info("shutdown complete")

    app = FastAPI(title="qwen-sql gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings, app.state.metrics, app.state.admission = settings, metrics, admission
    app.add_middleware(ObservabilityMiddleware, metrics=metrics)

    def reject(reason: str, status: int, detail: str, retry_after: str | None = None):
        metrics.rejections.labels(reason).inc()
        headers = {"Retry-After": retry_after} if retry_after else None
        raise HTTPException(status, detail={"error": reason, "message": detail}, headers=headers)

    # ---- auth + per-key rate limit (a dependency, so /health etc. stay open) ----
    async def guard(request: Request) -> str:
        key = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if not settings.auth_disabled:
            if not key or not any(hmac.compare_digest(key, k) for k in settings.api_keys):
                reject("auth", 401, "missing or invalid API key")
        key_id = hashlib.sha256(key.encode()).hexdigest()[:8] if key else "anon"
        request.scope.setdefault("state", {})["key_id"] = key_id  # logged; never the key itself
        wait = bucket.try_acquire(key_id)
        if wait:
            reject("rate_limited", 429, "too many requests for this API key",
                   Rejected("rate_limited", wait).retry_after_header)
        return key_id

    async def admit() -> float:
        t0 = time.perf_counter()
        metrics.queue_depth.set(admission.waiting + 1)
        try:
            await admission.acquire()
        except Rejected as e:
            reject(e.reason, 429, "server busy, retry later", e.retry_after_header)
        finally:
            metrics.queue_depth.set(admission.waiting)
        metrics.in_flight.set(admission.in_flight)
        wait = time.perf_counter() - t0
        metrics.queue_wait.observe(wait)
        return wait

    def release() -> None:
        admission.release()
        metrics.in_flight.set(admission.in_flight)

    async def send_backend(path: str, body: dict, stream: bool) -> httpx.Response:
        req = client.build_request("POST", path, json=body)
        for attempt in (0, 1):
            try:
                return await client.send(req, stream=stream)
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                # failed before any response byte: nothing was generated, safe to retry once
                if attempt:
                    raise
                metrics.retries.inc()
                await asyncio.sleep(0.2)
        raise AssertionError("unreachable")

    async def call_backend(request: Request, path: str, body: dict) -> httpx.Response:
        """Non-streaming backend call bounded by the request timeout AND by the client staying
        connected. Without the second bound, a client that gives up leaves its slot busy until
        the model finishes an answer nobody reads (drill: 27 s), and the whole queue waits."""
        call = asyncio.ensure_future(send_backend(path, body, stream=False))
        gone = asyncio.ensure_future(wait_disconnect(request))
        try:
            done, _ = await asyncio.wait({call, gone}, timeout=settings.request_timeout_s,
                                         return_when=asyncio.FIRST_COMPLETED)
        finally:
            gone.cancel()
        if call in done:
            return call.result()
        call.cancel()   # closes the backend connection; llama-server then stops generating
        if gone in done:
            metrics.backend_errors.labels("client_gone").inc()
            raise HTTPException(499, {"error": "client_closed_request"})
        raise asyncio.TimeoutError

    def backend_failure(e: Exception) -> HTTPException:
        if isinstance(e, httpx.TimeoutException | asyncio.TimeoutError):
            metrics.backend_errors.labels("timeout").inc()
            return HTTPException(504, {"error": "backend_timeout", "message": "model took too long"})
        metrics.backend_errors.labels("unavailable").inc()
        log.warning("backend unavailable", extra={"fields": {"error": type(e).__name__}})
        return HTTPException(502, {"error": "backend_unavailable", "message": "model server unreachable"})

    def check_size(body: dict) -> None:
        n = len(str(body.get("prompt", ""))) + sum(len(str(m.get("content", "")))
                                                   for m in body.get("messages", []) or [])
        if n > settings.max_input_chars:
            reject("too_large", 413, f"input is {n} chars; limit {settings.max_input_chars}")
        mt = body.get("max_tokens")
        body["max_tokens"] = min(int(mt), settings.max_tokens_cap) if mt else settings.max_tokens_cap

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            reject("bad_request", 400, "body must be a JSON object")
        return body

    # ---------- routes ----------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        try:
            r = await client.get("/health", timeout=2.0)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        metrics.ready.set(int(ok))
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def prom():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    async def proxy(request: Request, path: str) -> Response:
        body = await json_body(request)
        check_size(body)
        body.setdefault("model", settings.model_name)
        stream = bool(body.get("stream"))
        await admit()
        if not stream:
            try:
                r = await call_backend(request, path, body)
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                raise backend_failure(e) from None
            finally:
                release()
            if r.status_code >= 500:
                metrics.backend_errors.labels(f"http_{r.status_code}").inc()
            return Response(r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type", "application/json"))

        try:
            r = await send_backend(path, body, stream=True)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            release()
            raise backend_failure(e) from None
        except BaseException:
            release()
            raise
        done = False

        async def close() -> None:  # idempotent: runs from the generator AND the response wrapper
            nonlocal done
            if not done:
                done = True
                await r.aclose()
                release()

        if r.status_code != 200:
            content = await r.aread()
            await close()
            return Response(content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        return _ClosingStream(stream_body(r, path, time.perf_counter(), close), close)

    async def stream_body(r: httpx.Response, route: str, t0: float, close):
        first = True
        try:
            async for chunk in r.aiter_raw():
                if first:
                    metrics.ttft.labels(route).observe(time.perf_counter() - t0)
                    first = False
                yield chunk
        except httpx.HTTPError as e:
            # headers (200) are already sent: tell the client in-band, then end the stream
            metrics.backend_errors.labels("mid_stream").inc()
            log.warning("backend failed mid-stream", extra={"fields": {"error": type(e).__name__}})
            yield b'data: {"error": {"type": "backend_failed_mid_stream"}}\n\n'
        finally:
            await close()

    @app.post("/v1/completions")
    async def completions(request: Request, _key: str = Depends(guard)):
        return await proxy(request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat(request: Request, _key: str = Depends(guard)):
        return await proxy(request, "/v1/chat/completions")

    @app.post("/v1/sql")
    async def sql(request: Request, _key: str = Depends(guard)):
        body = await json_body(request)
        schema, question = body.get("schema"), body.get("question")
        if not isinstance(schema, str) or not isinstance(question, str) or not question.strip():
            reject("bad_request", 400, "need string fields 'schema' and 'question'")
        req = {"model": settings.model_name, "temperature": 0.0, "max_tokens": body.get("max_tokens", 128),
               "prompt": render_generation_prompt(build_messages(schema, question))}
        check_size(req)
        queue_wait = await admit()
        t0 = time.perf_counter()
        try:
            r = await call_backend(request, "/v1/completions", req)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            raise backend_failure(e) from None
        finally:
            release()
        if r.status_code != 200:
            if r.status_code >= 500:
                metrics.backend_errors.labels(f"http_{r.status_code}").inc()
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        out = r.json()
        text = out["choices"][0]["text"]
        return {"sql": extract_sql(text), "raw": text, "usage": out.get("usage"),
                "timings": {"queue_wait_s": round(queue_wait, 3),
                            "model_s": round(time.perf_counter() - t0, 3)}}

    return app


async def wait_disconnect(request: Request, poll_s: float = 0.25) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(poll_s)


class _ClosingStream(StreamingResponse):
    """Releases the backend slot even if the client disconnects before the body generator
    starts (a never-started generator's `finally` never runs: a leaked slot = a dead gateway)."""

    def __init__(self, gen, close):
        super().__init__(gen, media_type="text/event-stream")
        self._close = close

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._close()


def __getattr__(name: str):
    # `uvicorn serving.gateway.app:app` builds the app from the environment on first access, so
    # importing this module (tests) doesn't require GATEWAY_* to be set.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
