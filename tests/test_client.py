import asyncio

import httpx

from bench.client import send_one

ITEM = {"prompt": "q", "max_tokens": 4}
SSE = (b'data: {"choices":[{"text":"SELECT"}]}\n\n'
       b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\ndata: [DONE]\n\n')


def run(handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await send_one(c, "http://x", "m", ITEM, 0, ignore_eos=False)
    return asyncio.run(go())


def test_retries_once_on_dropped_keepalive():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, content=SSE)

    rec = run(handler)
    assert rec.ok and rec.output_tokens == 1 and len(calls) == 2


def test_gives_up_after_one_retry_and_does_not_retry_other_errors():
    calls = []

    def dropped(request):
        calls.append(1)
        raise httpx.RemoteProtocolError("gone")

    assert run(dropped).error == "RemoteProtocolError" and len(calls) == 2
    calls.clear()

    def timeout(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    assert run(timeout).error == "ReadTimeout" and len(calls) == 1
