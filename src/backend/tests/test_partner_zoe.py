"""POST /zoe/v1/chat/completions — OpenAI-compatible, stateless, billed on
delivery from the org pool. GET /zoe/v1/models for SDK probes."""

import json
from types import SimpleNamespace

import pytest

from partner_api import zoe
from partner_api.service import PartnerContext

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
CTX = PartnerContext(org_id=ORG_ID, key_id="k-1")
H = {"Authorization": "Bearer mk_live_ok"}
URL = "/zoe/v1/chat/completions"
ASK = {"model": "zoe", "messages": [{"role": "user", "content": "What is a mechanical royalty?"}]}


class FakeStream:
    """A stream=True OpenAI response: content chunks, then the usage-only chunk."""

    def __init__(self, parts):
        self.parts = parts
        self.exhausted = False

    def __iter__(self):
        for p in self.parts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))])
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))
        self.exhausted = True


class FakeOpenAI:
    def __init__(self, answer="A mechanical royalty is paid per reproduction.", parts=None, boom=False):
        self.calls = []
        self.answer = answer
        self.stream = FakeStream(parts or ["A mechanical ", "royalty is paid ", "per reproduction."])
        self.boom = boom
        outer = self

        class completions:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                if outer.boom:
                    raise RuntimeError("openai down")
                if kw.get("stream"):
                    return outer.stream
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=outer.answer), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
                )

        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture
def partner(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.setattr("partner_api.service.resolve_key", lambda sb, b: CTX)
    monkeypatch.setattr("partner_api.service.get_price", lambda sb, action: {"partner_zoe_message": 5}[action])
    monkeypatch.setattr(
        "partner_api.service.check_pool",
        lambda sb, org, price: {"ok": True, "balance": 100, "wallet_id": "w1", "period_end": None},
    )
    monkeypatch.setattr("utils.llm.model_garden.model_for", lambda slot: "gpt-test")


@pytest.fixture
def openai(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr("utils.llm.client.get_openai_client", lambda: fake)
    return fake


@pytest.fixture
def debits(monkeypatch):
    seen = []
    monkeypatch.setattr("partner_api.service.debit_run", lambda sb, **kw: seen.append(kw))
    return seen


def _sse_events(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def test_non_stream_is_openai_shaped_and_bills_the_base_after_delivery(client, partner, openai, debits):
    r = client.post(URL, headers=H, json=ASK)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "zoe"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": openai.answer}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert body["id"].startswith("chatcmpl-")
    # Our persona leads; the partner's messages follow verbatim.
    sent = openai.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": zoe.ZOE_SYSTEM_PROMPT}
    assert sent[1:] == ASK["messages"]
    assert openai.calls[0]["max_completion_tokens"] == zoe.MAX_OUTPUT_TOKENS
    assert "stream" not in openai.calls[0]
    # Billed once, on the Zoe price row, unmeasured -> the base.
    assert len(debits) == 1
    assert debits[0]["action"] == "partner_zoe_message"
    assert debits[0]["amount"] == 5
    assert debits[0]["wallet_id"] == "w1" and debits[0]["key_id"] == "k-1"


def test_stream_emits_openai_chunks_bills_after_content_and_ends_with_done(client, partner, openai, monkeypatch):
    seen = []

    def debit(sb, **kw):
        # Delivery before billing: every content chunk has been read by now.
        assert openai.stream.exhausted
        seen.append(kw)

    monkeypatch.setattr("partner_api.service.debit_run", debit)
    r = client.post(URL, headers=H, json={**ASK, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "zoe" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "A mechanical royalty is paid per reproduction."
    assert chunks[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert len({c["id"] for c in chunks}) == 1
    assert openai.calls[0]["stream"] is True
    assert len(seen) == 1 and seen[0]["action"] == "partner_zoe_message" and seen[0]["amount"] == 5


def test_dry_pool_is_402_before_any_model_call(client, partner, openai, debits, monkeypatch):
    monkeypatch.setattr(
        "partner_api.service.check_pool", lambda sb, org, price: {"ok": False, "balance": 2, "wallet_id": "w1"}
    )
    r = client.post(URL, headers=H, json=ASK)
    assert r.status_code == 402
    assert r.json()["detail"] == {"code": "insufficient_credits", "price": 5, "balance": 2}
    assert openai.calls == [] and debits == []


def test_model_call_failure_is_502_and_unbilled(client, partner, debits, monkeypatch):
    monkeypatch.setattr("utils.llm.client.get_openai_client", lambda: FakeOpenAI(boom=True))
    r = client.post(URL, headers=H, json=ASK)
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "zoe_failed"
    assert debits == []


def test_stream_failure_is_an_error_frame_and_unbilled(client, partner, debits, monkeypatch):
    monkeypatch.setattr("utils.llm.client.get_openai_client", lambda: FakeOpenAI(boom=True))
    r = client.post(URL, headers=H, json={**ASK, "stream": True})
    assert r.status_code == 200
    events = _sse_events(r.text)
    assert json.loads(events[0])["error"]["code"] == "zoe_failed"
    assert events[-1] == "[DONE]"
    assert debits == []


def test_request_validation(client, partner, openai, debits):
    # OpenAI extras are ignored, content parts are flattened, temperature honoured.
    ok = client.post(
        URL,
        headers=H,
        json={
            "model": "zoe",
            "messages": [
                {"role": "system", "content": "House style: terse."},
                {"role": "user", "content": [{"type": "text", "text": "Hi"}], "name": "x"},
            ],
            "temperature": 0.2,
            "n": 1,
            "tools": [],
            "user": "abc",
        },
    )
    assert ok.status_code == 200
    sent = openai.calls[-1]
    assert sent["messages"][1:] == [
        {"role": "system", "content": "House style: terse."},
        {"role": "user", "content": "Hi"},
    ]
    assert sent["temperature"] == 0.2

    assert client.post(URL, headers=H, json={"messages": []}).status_code == 422
    assert client.post(URL, headers=H, json={"messages": [{"role": "tool", "content": "x"}]}).status_code == 422
    assert (
        client.post(URL, headers=H, json={"messages": [{"role": "user", "content": "x" * 100_001}]}).status_code == 422
    )
    assert client.post(URL, headers=H, json={**ASK, "max_tokens": 99_999}).status_code == 422
    r = client.post(URL, headers=H, json={**ASK, "model": "gpt-4o"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "model_not_found"
    # None of the rejected requests reached the model or the pool.
    assert len(openai.calls) == 1 and len(debits) == 1


def test_models_lists_zoe(client, partner):
    r = client.get("/zoe/v1/models", headers=H)
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["data"]] == ["zoe"]


def test_bad_key_is_401(client, partner, monkeypatch):
    monkeypatch.setattr("partner_api.service.resolve_key", lambda sb, b: None)
    assert client.post(URL, headers=H, json=ASK).status_code == 401
    assert client.get("/zoe/v1/models", headers=H).status_code == 401


def test_prompt_is_stateless_no_supabase_reads(client, partner, openai, debits, mock_supabase):
    # No stored contracts, no memory: the only Supabase traffic is billing.
    mock_supabase.table.reset_mock()
    client.post(URL, headers=H, json=ASK)
    assert not any(
        c.args[0] in ("contracts", "zoe_sessions", "project_files") for c in mock_supabase.table.call_args_list
    )
    assert isinstance(debits, list)
