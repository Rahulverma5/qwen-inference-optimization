"""Gateway behaviour against a fake llama-server (httpx.MockTransport): no model, no network."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from finetune.prompt import build_messages, render_generation_prompt
from serving.gateway.app import create_app
from serving.gateway.config import Settings
from serving.gateway.limits import Admission, Rejected, TokenBucket

KEY = "test-key"
H = {"Authorization": f"Bearer {KEY}"}


class FakeBackend:
    """Mimics llama-server: /health, /v1/completions (stream or not). Knobs per test."""

    def __init__(self, delay=0.0, fail_first=0, down=False, die_mid_stream=False):
        self.delay, self.fail_first, self.down, self.die_mid_stream = delay, fail_first, down, die_mid_stream
        self.calls: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if self.fail_first:
            self.fail_first -= 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        body = json.loads(request.content)
        self.calls.append(body)
        await asyncio.sleep(self.delay)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=SSE(self.die_mid_stream))
        return httpx.Response(200, json={"choices": [{"text": "```sql\nSELECT count(*) FROM singer;\n```"}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 9}})


class SSE(httpx.AsyncByteStream):
    def __init__(self, die: bool):
        self.die = die

    async def __aiter__(self):
        yield b'data: {"choices": [{"text": "SELECT"}]}\n\n'
        if self.die:
            raise httpx.ReadError("backend killed")
        yield b'data: {"choices": [{"text": " 1"}]}\n\n'
        yield b"data: [DONE]\n\n"


def make(backend: FakeBackend, **kw):
    s = Settings(api_keys=(KEY, "other-key"), **({"burst": 100, "rate_per_s": 100.0} | kw))
    return create_app(s, transport=httpx.MockTransport(backend))


def run(app, fn):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
            return await fn(c)
    return asyncio.run(go())


SQL = {"schema": "CREATE TABLE singer(id int);", "question": "How many singers?"}


# ---------- health / auth / basic proxy ----------

def test_health_needs_no_key_and_ready_follows_backend():
    async def fn(c):
        return (await c.get("/health")).status_code, (await c.get("/ready")).status_code
    assert run(make(FakeBackend()), fn) == (200, 200)
    assert run(make(FakeBackend(down=True)), fn) == (200, 503)


def test_auth_rejects_missing_and_wrong_keys_accepts_both_header_styles():
    async def fn(c):
        return [(await c.post("/v1/sql", json=SQL, headers=h)).status_code for h in
                ({}, {"Authorization": "Bearer nope"}, H, {"X-API-Key": KEY})]
    assert run(make(FakeBackend()), fn) == [401, 401, 200, 200]


def test_sql_endpoint_builds_the_training_prompt_and_extracts_sql():
    be = FakeBackend()

    async def fn(c):
        return await c.post("/v1/sql", json=SQL, headers=H | {"X-Request-ID": "abc"})
    r = run(make(be), fn)
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc"
    assert r.json()["sql"] == "SELECT count(*) FROM singer"
    sent = be.calls[0]
    assert sent["prompt"] == render_generation_prompt(build_messages(SQL["schema"], SQL["question"]))
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 128


def test_bad_bodies():
    async def fn(c):
        return [(await c.post("/v1/sql", json=b, headers=H)).status_code
                for b in ({"schema": "x"}, [1, 2], {"schema": "x", "question": "  "})]
    assert run(make(FakeBackend()), fn) == [400, 400, 400]


def test_input_too_large_is_413_and_max_tokens_is_capped():
    be = FakeBackend()

    async def fn(c):
        big = await c.post("/v1/completions", json={"prompt": "x" * 9000}, headers=H)
        ok = await c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 10_000}, headers=H)
        return big.status_code, ok.status_code
    assert run(make(be), fn) == (413, 200)
    assert be.calls[0]["max_tokens"] == 256


# ---------- rate limit and admission (back-pressure) ----------

def test_token_bucket_is_per_key_and_sets_retry_after():
    async def fn(c):
        codes = [(await c.post("/v1/sql", json=SQL, headers=H)).status_code for _ in range(3)]
        limited = await c.post("/v1/sql", json=SQL, headers=H)
        other = await c.post("/v1/sql", json=SQL, headers={"X-API-Key": "other-key"})
        return codes, limited.headers.get("retry-after"), other.status_code
    codes, retry_after, other = run(make(FakeBackend(), burst=2, rate_per_s=0.1), fn)
    assert codes == [200, 200, 429] and int(retry_after) >= 1 and other == 200


def test_queue_full_and_queue_timeout_return_429_while_one_request_runs():
    app = make(FakeBackend(delay=0.6), max_in_flight=1, max_queue=1, max_queue_wait_s=0.2)

    async def fn(c):
        first = asyncio.create_task(c.post("/v1/sql", json=SQL, headers=H))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(c.post("/v1/sql", json=SQL, headers=H))   # waits, times out
        await asyncio.sleep(0.05)
        third = await c.post("/v1/sql", json=SQL, headers=H)                    # queue full: at once
        r1, r2 = await first, await second
        metrics = (await c.get("/metrics")).text
        return r1, r2, third, metrics
    r1, r2, r3, metrics = run(app, fn)
    assert r1.status_code == 200
    assert (r2.status_code, r2.json()["detail"]["error"]) == (429, "queue_timeout")
    assert (r3.status_code, r3.json()["detail"]["error"]) == (429, "queue_full")
    assert 'gateway_rejections_total{reason="queue_full"} 1.0' in metrics
    assert app.state.admission.in_flight == 0


def test_queued_request_runs_when_slot_frees():
    app = make(FakeBackend(delay=0.2), max_in_flight=1, max_queue=2, max_queue_wait_s=2.0)

    async def fn(c):
        rs = await asyncio.gather(*(c.post("/v1/sql", json=SQL, headers=H) for _ in range(3)))
        return [r.status_code for r in rs], [r.json()["timings"]["queue_wait_s"] for r in rs]
    codes, waits = run(app, fn)
    assert codes == [200, 200, 200] and max(waits) >= 0.35   # third waited for two 0.2 s requests


# ---------- backend failures ----------

def test_retries_once_when_connection_drops_before_any_byte():
    app = make(FakeBackend(fail_first=1))

    async def fn(c):
        return (await c.post("/v1/sql", json=SQL, headers=H)).status_code, (await c.get("/metrics")).text
    code, metrics = run(app, fn)
    assert code == 200 and "gateway_backend_retries_total 1.0" in metrics


def test_backend_down_is_502_and_frees_the_slot():
    app = make(FakeBackend(down=True))

    async def fn(c):
        return [(await c.post("/v1/sql", json=SQL, headers=H)).status_code for _ in range(2)]
    assert run(app, fn) == [502, 502]
    assert app.state.admission.in_flight == 0


def test_backend_timeout_is_504():
    async def fn(c):
        return (await c.post("/v1/sql", json=SQL, headers=H)).status_code
    app = make(FakeBackend(delay=1.0), request_timeout_s=0.2)
    assert run(app, fn) == 504
    assert app.state.admission.in_flight == 0


# ---------- streaming ----------

def test_streaming_passthrough_records_ttft_and_releases():
    app = make(FakeBackend())

    async def fn(c):
        r = await c.post("/v1/completions", json={"prompt": "x", "stream": True}, headers=H)
        return r, (await c.get("/metrics")).text
    r, metrics = run(app, fn)
    assert r.status_code == 200 and r.text.endswith("data: [DONE]\n\n")
    assert 'gateway_ttft_seconds_count{route="/v1/completions"} 1.0' in metrics
    assert app.state.admission.in_flight == 0


def test_backend_dying_mid_stream_sends_error_event_and_releases():
    app = make(FakeBackend(die_mid_stream=True))

    async def fn(c):
        return await c.post("/v1/completions", json={"prompt": "x", "stream": True}, headers=H)
    r = run(app, fn)
    assert r.status_code == 200 and "backend_failed_mid_stream" in r.text
    assert app.state.admission.in_flight == 0


# ---------- units ----------

def test_token_bucket_refills_at_rate():
    t = [0.0]
    b = TokenBucket(rate=2.0, burst=2, clock=lambda: t[0])
    assert [b.try_acquire("k") for _ in range(2)] == [0.0, 0.0]
    assert b.try_acquire("k") == pytest.approx(0.5)
    t[0] = 0.5
    assert b.try_acquire("k") == 0.0


def test_rejected_retry_after_rounds_up_to_whole_seconds():
    assert Rejected("x", 0.01).retry_after_header == "1"
    assert Rejected("x", 2.2).retry_after_header == "3"


def test_admission_counts_return_to_zero():
    async def go():
        a = Admission(1, 0, 0.1)
        await a.acquire()
        with pytest.raises(Rejected):
            await a.acquire()
        a.release()
        return a.in_flight, a.waiting
    assert asyncio.run(go()) == (0, 0)


def test_settings_refuse_to_start_without_keys(monkeypatch):
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("GATEWAY_API_KEYS", "a, b")
    assert Settings.from_env().api_keys == ("a", "b")


TOKENIZER = Path("models/week3/models/fp16")


@pytest.mark.skipif(not (TOKENIZER / "tokenizer_config.json").exists(), reason="local model files only")
def test_render_matches_the_real_qwen3_template():
    transformers = pytest.importorskip("transformers")
    from finetune.prompt import render
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    msgs = build_messages("CREATE TABLE t(a int);\nCREATE TABLE u(b text);", "How many rows in t?")
    assert render_generation_prompt(msgs) == render(tok, msgs, add_generation_prompt=True)


def test_client_that_leaves_cancels_the_backend_call_and_frees_the_slot(monkeypatch):
    import serving.gateway.app as gw

    async def gone_soon(request, poll_s=0.25):
        await asyncio.sleep(0.1)
    monkeypatch.setattr(gw, "wait_disconnect", gone_soon)
    app = make(FakeBackend(delay=5.0))

    async def fn(c):
        t0 = asyncio.get_running_loop().time()
        r = await c.post("/v1/sql", json=SQL, headers=H)
        return r.status_code, asyncio.get_running_loop().time() - t0
    code, dt = run(app, fn)
    assert code == 499 and dt < 1.0 and app.state.admission.in_flight == 0
