"""The drop-in client we hand partners (examples/partner-api/msanii_partner.py)
must speak the wire format the router actually emits: multipart field names,
the SSE framing with heartbeats, and both error phases."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "examples" / "partner-api" / "msanii_partner.py"
_spec = importlib.util.spec_from_file_location("msanii_partner", _PATH)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


class FakeResp:
    def __init__(self, status, lines=(), body=None, content=b""):
        self.status_code = status
        self.reason = "reason"
        self._lines = list(lines)
        self._body = body
        self.content = content

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client(resp):
    api = mp.MsaniiPartner("mk_live_x", "https://partner.example/")
    calls = []
    api.session.post = lambda url, **kw: calls.append((url, kw)) or resp
    api.session.get = lambda url, **kw: calls.append((url, kw)) or resp
    return api, calls


def test_iter_events_skips_pings_joins_multiline_and_flushes_tail():
    lines = [": ping", "", 'data: {"type":', 'data: "result", "n": 1}', "", ": ping", "", 'data: {"type":"x"}']
    assert list(mp.iter_events(lines)) == [{"type": "result", "n": 1}, {"type": "x"}]


def test_calculate_sends_router_field_names_and_returns_result_event():
    result = {"type": "result", "payments": [], "total_payments": 0, "expense_review_required": False}
    api, calls = _client(FakeResp(200, [": ping", "", "data: " + json.dumps(result), ""]))
    out = api.calculate(
        ("s.csv", b"Title,Net Payable\nA,1\n"),
        contracts=[("a.pdf", b"%PDF"), ("b.pdf", b"%PDF")],
        expenses=[{"amount": 1}],
        idempotency_key="k1",
    )
    assert out == result
    url, kw = calls[0]
    assert url == "https://partner.example/oneclick/v1/royalties"
    # `contracts`, repeated — NOT `contracts[]`, which FastAPI would ignore.
    assert [name for name, _ in kw["files"]] == ["statement", "contracts", "contracts"]
    assert json.loads(kw["data"]["expenses"]) == [{"amount": 1}]
    assert "contract_terms" not in kw["data"]
    assert kw["headers"] == {"Idempotency-Key": "k1"}
    assert kw["stream"] is True


def test_calculate_terms_mode_json_encodes_terms():
    api, calls = _client(FakeResp(200, ['data: {"type":"result","payments":[],"total_payments":0}', ""]))
    api.calculate(("s.csv", b"x"), contract_terms={"parties": []})
    assert json.loads(calls[0][1]["data"]["contract_terms"]) == {"parties": []}
    assert calls[0][1]["headers"] == {}


def test_in_stream_error_raises_with_code_message_suggestion_and_no_status():
    ev = {"type": "error", "code": "NO_SONG_MATCHES", "message": "m", "suggestion": "s", "details": {"x": 1}}
    api, _ = _client(FakeResp(200, ["data: " + json.dumps(ev), ""]))
    with pytest.raises(mp.MsaniiError) as ei:
        api.calculate(("s.csv", b"x"), contract_terms={})
    e = ei.value
    assert (e.code, e.message, e.suggestion, e.status) == ("NO_SONG_MATCHES", "m", "s", None)
    assert e.details == {"details": {"x": 1}}


def test_pre_stream_402_carries_price_and_balance():
    api, _ = _client(FakeResp(402, body={"detail": {"code": "insufficient_credits", "price": 30, "balance": 4}}))
    with pytest.raises(mp.MsaniiError) as ei:
        api.calculate(("s.csv", b"x"), contract_terms={})
    assert ei.value.status == 402
    assert ei.value.code == "insufficient_credits"
    assert ei.value.details == {"price": 30, "balance": 4}


def test_validation_422_and_bare_401_map_to_stable_codes():
    api, _ = _client(FakeResp(422, body={"detail": [{"loc": ["body", "statement"], "msg": "field required"}]}))
    with pytest.raises(mp.MsaniiError) as ei:
        api.models()
    assert ei.value.code == "invalid_request" and "field required" in ei.value.message

    api, _ = _client(FakeResp(401, body={"detail": {"code": "invalid_key"}}))
    with pytest.raises(mp.MsaniiError) as ei:
        api.models()
    assert (ei.value.code, ei.value.status) == ("invalid_key", 401)


def test_models_is_the_free_key_check():
    api, calls = _client(FakeResp(200, body={"object": "list", "data": [{"id": "zoe", "object": "model"}]}))
    assert api.models() == ["zoe"]
    assert calls[0][0] == "https://partner.example/zoe/v1/models"


def test_parse_contract_sends_contracts_repeated_and_returns_both_views():
    result = {
        "type": "result",
        "contract_terms": {"parties": []},
        "splits": {"parties": [], "main_artist_found": False},
    }
    api, calls = _client(FakeResp(200, [": ping", "", "data: " + json.dumps(result), ""]))
    out = api.parse_contract([("a.pdf", b"%PDF"), ("b.pdf", b"%PDF")], main_artist_name="Jane", idempotency_key="p1")
    assert out == result
    url, kw = calls[0]
    assert url == "https://partner.example/registry/v1/splits"
    assert [name for name, _ in kw["files"]] == ["contracts", "contracts"]
    assert kw["data"] == {"main_artist_name": "Jane"}
    assert kw["headers"] == {"Idempotency-Key": "p1"} and kw["stream"] is True


def test_split_sheet_posts_the_document_body_and_returns_bytes():
    api, calls = _client(FakeResp(200, content=b"%PDF-1.4 sheet"))
    pdf = api.split_sheet(
        work_title="Blue Sky",
        date="6 Sept 2026",
        contributors=[{"name": "Jane", "role": "Producer", "master_percentage": 50}],
        idempotency_key="s1",
    )
    assert pdf == b"%PDF-1.4 sheet"
    url, kw = calls[0]
    assert url == "https://partner.example/splitsheet/v1/documents"
    assert kw["json"] == {
        "work_title": "Blue Sky",
        "work_type": "single",
        "split_type": "both",
        "date": "6 Sept 2026",
        "format": "pdf",
        "contributors": [{"name": "Jane", "role": "Producer", "master_percentage": 50}],
    }
    assert kw["headers"] == {"Idempotency-Key": "s1"}

    api, _ = _client(FakeResp(402, body={"detail": {"code": "insufficient_credits", "price": 20, "balance": 1}}))
    with pytest.raises(mp.MsaniiError) as ei:
        api.split_sheet(work_title="x", date="d", contributors=[{"name": "a", "role": "b"}])
    assert ei.value.code == "insufficient_credits" and ei.value.details == {"price": 20, "balance": 1}


def test_stream_that_ends_without_result_is_an_error():
    api, _ = _client(FakeResp(200, [": ping", ""]))
    with pytest.raises(mp.MsaniiError) as ei:
        api.calculate(("s.csv", b"x"), contract_terms={})
    assert ei.value.code == "no_result"


def test_chat_posts_openai_shape_and_returns_the_answer():
    body = {"choices": [{"message": {"role": "assistant", "content": "Paid per copy."}}]}
    api, calls = _client(FakeResp(200, body=body))
    assert api.chat([{"role": "user", "content": "Mechanical?"}], max_tokens=50) == "Paid per copy."
    url, kw = calls[0]
    assert url == "https://partner.example/zoe/v1/chat/completions"
    assert kw["json"] == {"model": "zoe", "messages": [{"role": "user", "content": "Mechanical?"}], "max_tokens": 50}


def test_chat_stream_yields_deltas_until_done_and_surfaces_error_frames():
    def chunk(delta):
        return "data: " + json.dumps({"choices": [{"delta": delta}]})

    api, calls = _client(
        FakeResp(
            200,
            [
                chunk({"role": "assistant", "content": ""}),
                "",
                chunk({"content": "Paid "}),
                "",
                chunk({"content": "per copy."}),
                "",
                chunk({}),
                "",
                "data: [DONE]",
                "",
            ],
        )
    )
    assert "".join(api.chat_stream([{"role": "user", "content": "x"}])) == "Paid per copy."
    assert calls[0][1]["json"]["stream"] is True

    api, _ = _client(
        FakeResp(200, ['data: {"error": {"code": "zoe_failed", "message": "down"}}', "", "data: [DONE]", ""])
    )
    with pytest.raises(mp.MsaniiError) as ei:
        list(api.chat_stream([{"role": "user", "content": "x"}]))
    assert ei.value.code == "zoe_failed"
