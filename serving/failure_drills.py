"""Failure drills against the running compose stack (Week 4). Each drill breaks one thing on
purpose and records what the client saw and how long recovery took.

    docker compose -f serving/compose.yml up -d
    python -m serving.failure_drills --key local-dev-key --out reports/week4/failure_drills.json

Drills: backend killed mid-stream (+ does Docker restart it?), oversized input, client
disconnecting mid-stream and mid-request, SIGTERM drain of the gateway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import httpx

from finetune.prompt import build_messages, render_generation_prompt

COMPOSE = ["docker", "compose", "-f", "serving/compose.yml"]
GW, LLAMA = "http://localhost:8000", "http://localhost:8080"


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True)


def container_state(service: str) -> dict:
    cid = subprocess.run([*COMPOSE, "ps", "-aq", service], capture_output=True, text=True).stdout.strip()
    out = subprocess.run(["docker", "inspect", "-f", "{{json .}}", cid], capture_output=True, text=True)
    c = json.loads(out.stdout)
    return {"status": c["State"]["Status"], "restart_count": c["RestartCount"],
            "exit_code": c["State"]["ExitCode"]}


def wait_for(url: str, want: int, timeout: float) -> float | None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code == want:
                return round(time.perf_counter() - t0, 2)
        except httpx.HTTPError:
            if want == 0:
                return round(time.perf_counter() - t0, 2)
        time.sleep(0.25)
    return None


def gauge(name: str) -> float:
    for line in httpx.get(f"{GW}/metrics", timeout=5).text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise KeyError(name)


def long_body(stream: bool) -> dict:
    prompt = render_generation_prompt(build_messages(
        "CREATE TABLE t(a int);", "Write 300 different SQL queries over t, one per line."))
    return {"prompt": prompt, "max_tokens": 256, "stream": stream, "ignore_eos": True, "temperature": 0}


async def drill_backend_killed_mid_stream(h: dict) -> dict:
    out: dict = {}
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{GW}/v1/completions", json=long_body(True), headers=h) as r:
            chunks, killed_at = 0, None
            t0 = time.perf_counter()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunks += 1
                    if chunks == 5 and killed_at is None:
                        compose("kill", "llama")   # SIGKILL: what an OOM kill or a segfault looks like
                        killed_at = time.perf_counter()
                    if "backend_failed_mid_stream" in line:
                        out["client_got_error_event"] = True
            out |= {"http_status": r.status_code, "chunks_before_end": chunks,
                    "stream_ended_after_kill_s": round(time.perf_counter() - killed_at, 2),
                    "total_s": round(time.perf_counter() - t0, 2)}
    out.setdefault("client_got_error_event", False)
    out["ready_503_after_s"] = wait_for(f"{GW}/ready", 503, 10)
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{GW}/v1/completions", json={"prompt": "x", "max_tokens": 1}, headers=h)
        out["request_while_down"] = f"{r.status_code} {r.json().get('detail', {}).get('error')}"
    time.sleep(5)
    out["state_5s_after_kill"] = container_state("llama")
    out["docker_restarted_it"] = out["state_5s_after_kill"]["status"] == "running"
    if not out["docker_restarted_it"]:
        compose("start", "llama")
    out["ready_again_after_start_s"] = wait_for(f"{GW}/ready", 200, 120)
    out["gateway_in_flight_after"] = gauge("gateway_in_flight")
    return out


async def drill_backend_process_exits(h: dict) -> dict:
    """The process dies on its own (crash, OOM inside the container): unlike `docker kill`
    (a manual stop, which restart policies deliberately ignore), Docker should restart it."""
    before = container_state("llama")["restart_count"]
    t0 = time.perf_counter()
    subprocess.run(["docker", "exec", subprocess.run([*COMPOSE, "ps", "-q", "llama"], capture_output=True,
                    text=True).stdout.strip(), "kill", "-TERM", "1"], check=True)
    down = wait_for(f"{GW}/ready", 503, 20)
    up = wait_for(f"{GW}/ready", 200, 120)
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{GW}/v1/completions", json={"prompt": "x", "max_tokens": 1}, headers=h)
    back = round(time.perf_counter() - t0, 2) if up is not None else None
    return {"ready_503_after_s": down, "ready_200_again_after_s": back,
            "restart_count_delta": container_state("llama")["restart_count"] - before,
            "first_request_after": r.status_code}


async def drill_oversized_input(h: dict) -> dict:
    async with httpx.AsyncClient(timeout=120) as c:
        t0 = time.perf_counter()
        r1 = await c.post(f"{GW}/v1/sql", json={"schema": "x" * 20_000, "question": "q"}, headers=h)
        t1 = time.perf_counter()
        # under the gateway's char limit but over llama's 2,048-token context: numbers ~1 token each
        r2 = await c.post(f"{GW}/v1/completions", headers=h,
                          json={"prompt": " ".join(str(i % 10) for i in range(3_900)), "max_tokens": 4})
        t2 = time.perf_counter()
    return {"gateway_limit": {"status": r1.status_code, "s": round(t1 - t0, 3), "body": r1.json()},
            "backend_context_limit": {"status": r2.status_code, "s": round(t2 - t1, 3),
                                      "body": r2.text[:200]}}


async def drill_client_disconnects(h: dict) -> dict:
    out: dict = {}
    # (a) streaming client reads a few tokens, then hangs up
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{GW}/v1/completions", json=long_body(True), headers=h) as r:
            n = 0
            async for line in r.aiter_lines():
                n += line.startswith("data: ")
                if n == 3:
                    break
    t0 = time.perf_counter()
    while gauge("gateway_in_flight") and time.perf_counter() - t0 < 30:
        await asyncio.sleep(0.1)
    out["stream_slot_freed_after_s"] = round(time.perf_counter() - t0, 2)
    async with httpx.AsyncClient(timeout=120) as c:
        t1 = time.perf_counter()
        r = await c.post(f"{GW}/v1/completions", json={"prompt": "x", "max_tokens": 1}, headers=h)
        out["next_request"] = {"status": r.status_code, "s": round(time.perf_counter() - t1, 2)}
    # (b) non-streaming client gives up after 1 s
    async with httpx.AsyncClient(timeout=1.0) as c:
        try:
            await c.post(f"{GW}/v1/completions", json=long_body(False), headers=h)
        except httpx.TimeoutException:
            pass
    t0 = time.perf_counter()
    while gauge("gateway_in_flight") and time.perf_counter() - t0 < 120:
        await asyncio.sleep(0.2)
    out["non_stream_slot_held_after_client_left_s"] = round(time.perf_counter() - t0, 2)
    return out


async def drill_sigterm_drain(h: dict) -> dict:
    out: dict = {}
    async with httpx.AsyncClient(timeout=120) as c:
        task = asyncio.create_task(c.post(f"{GW}/v1/completions", json=long_body(False) | {"max_tokens": 64},
                                          headers=h))
        await asyncio.sleep(1.5)
        t0 = time.perf_counter()
        stop = await asyncio.create_subprocess_exec(*COMPOSE, "stop", "gateway",
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(1.0)
        try:
            r = await c.post(f"{GW}/health", timeout=3)
            out["new_request_during_drain"] = r.status_code
        except httpx.HTTPError as e:
            out["new_request_during_drain"] = type(e).__name__
        r = await task
        out["in_flight_request"] = {"status": r.status_code, "finished_after_sigterm_s":
                                    round(time.perf_counter() - t0, 2)}
        await stop.wait()
        out["gateway_stopped_after_s"] = round(time.perf_counter() - t0, 2)
        out["gateway_exit"] = container_state("gateway")
    compose("start", "gateway")
    out["gateway_ready_again_s"] = wait_for(f"{GW}/ready", 200, 60)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--key", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    h = {"X-API-Key": a.key}
    drills = {"backend_killed_mid_stream": drill_backend_killed_mid_stream,
              "backend_process_exits": drill_backend_process_exits,
              "oversized_input": drill_oversized_input,
              "client_disconnects": drill_client_disconnects,
              "sigterm_drain": drill_sigterm_drain}
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    for name, fn in drills.items():
        if a.only and name not in a.only:
            continue
        assert wait_for(f"{GW}/ready", 200, 120) is not None, "stack not ready"
        print(f"== {name}", flush=True)
        results[name] = asyncio.run(fn(h))
        print(json.dumps(results[name], indent=1), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
