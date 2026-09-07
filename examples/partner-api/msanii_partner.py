"""Msanii Partner API — Python client + smoke test.

Copy this file into your project. It needs only `requests` (pip install requests).

    from msanii_partner import MsaniiPartner

    api = MsaniiPartner("mk_live_…", base_url="https://<partner-api-host>")
    api.models()                                                        # free key check -> ["zoe"]
    api.calculate("statement.csv", contract_terms={...})                # no AI, base price
    api.calculate("statement.xlsx", contracts=["deal.pdf"], expenses=[...])
    #   -> {"summary": {...}, "payments": [...], "billing": {"credits": n, …}}
    api.parse_contract(["deal.pdf"], main_artist_name="Jane Doe")       # the deal as data
    #   -> {"contract_terms": {...}, "splits": {"main_artist": …, "parties": [...]}, "billing": {...}}
    api.split_sheet(work_title="Blue Sky", date="6 Sept 2026", contributors=[...])  # PDF bytes
    #   -> what it cost is in api.last_billing
    api.chat([{"role": "user", "content": "What is a mechanical royalty?"}])   # Zoe
    for delta in api.chat_stream([...]): print(delta, end="")

Zoe is OpenAI-compatible, so the official SDK works too — no client needed:

    from openai import OpenAI
    zoe = OpenAI(api_key="mk_live_…", base_url="https://<partner-api-host>/zoe/v1")
    zoe.chat.completions.create(model="zoe", messages=[{"role": "user", "content": "…"}])

Every method returns plain dicts (or bytes, for a split sheet) and raises
MsaniiError on any failure — HTTP errors before a stream opens and in-stream
errors alike.

Smoke test — runs each endpoint the way an external integration would:

    export MSANII_API_URL=https://<partner-api-host>
    export MSANII_API_KEY=mk_live_…
    python msanii_partner.py                       # models, bad key, terms-mode calculate x2, Zoe x2, split sheet
    python msanii_partner.py --contract deal.pdf   # + a contract parse
    python msanii_partner.py --statement s.xlsx --contract deal.pdf   # + a PDF-parse calculation

Every run except the model probe is billed to the team's credit pool (see
docs/partner-api-reference.md). The smoke test sends an Idempotency-Key on
calculations, parses and sheets, so re-running it in the same billing period
is charged once for each; Zoe answers have no idempotency and pay every time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

FilePart = str | Path | tuple[str, bytes]


class MsaniiError(Exception):
    """Any failure. `code` is stable and machine-readable; `message` and
    `suggestion` are safe to show a person. `status` is the HTTP status for
    pre-stream failures and None for in-stream ones (HTTP was already 200)."""

    def __init__(self, code, message=None, suggestion=None, status=None, details=None):
        super().__init__(message or code)
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.status = status
        self.details = details or {}


class MsaniiPartner:
    def __init__(self, api_key: str, base_url: str, timeout: float = 120):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        # Read timeout resets on every chunk; the server sends a heartbeat
        # every 15 s during a long parse, so 120 s never fires on a live run.
        self.timeout = timeout
        # What the last split sheet cost — its body is the document, so the
        # charge rides in headers. Empty until the first sheet.
        self.last_billing: dict = {}

    # ---- endpoints ----------------------------------------------------------

    def models(self) -> list[str]:
        """Free. The models Zoe serves — just "zoe" — and the natural key
        check: a bad, revoked or expired key is a 401 here."""
        return [m["id"] for m in self._json(self.session.get(f"{self.base}/zoe/v1/models", timeout=30))["data"]]

    def calculate(
        self,
        statement: FilePart,
        *,
        contracts: list[FilePart] | None = None,
        contract_terms: dict | None = None,
        expenses: list[dict] | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Royalty calculation. Exactly one of `contracts` (PDFs, parsed by AI)
        or `contract_terms` (structured, no AI). Returns the result event:
        {"summary": {"payments": n, "total_payable": x, "expense_review_required": bool},
         "payments": [{"song", "payee": {...}, "share": {...}, "amounts": {...}}],
         "billing": {"credits": n, "request_id": "…"}}.
        """
        files = [("statement", _part(statement))]
        files += [("contracts", _part(c)) for c in contracts or []]
        data = {}
        if contract_terms is not None:
            data["contract_terms"] = json.dumps(contract_terms)
        if expenses is not None:
            data["expenses"] = json.dumps(expenses)
        return self._stream_result(f"{self.base}/oneclick/v1/royalties", files, data, idempotency_key)

    def parse_contract(
        self,
        contracts: list[FilePart],
        *,
        main_artist_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """The deal as data. Returns the result event:
        {"contract_terms": {...}, "splits": {"main_artist": str | None, "parties": [...]},
         "billing": {...}}.
        `contract_terms` is exactly what calculate() takes, so parse once and
        run every statement against it at the base price."""
        files = [("contracts", _part(c)) for c in contracts]
        data = {"main_artist_name": main_artist_name} if main_artist_name else {}
        return self._stream_result(f"{self.base}/registry/v1/splits", files, data, idempotency_key)

    def split_sheet(
        self,
        *,
        work_title: str,
        date: str,
        contributors: list[dict],
        split_type: str = "both",
        work_type: str = "single",
        format: str = "pdf",
        idempotency_key: str | None = None,
    ) -> bytes:
        """A finished split sheet — the PDF (or DOCX) bytes, ready to save.
        Each contributor is {"name", "role", "publishing_share" | "writer_share" +
        "publisher_share", "master_percentage", ...} as in the reference.
        What the last document cost, from the response headers, is in
        `self.last_billing` (empty when the headers were absent)."""
        body = {
            "work_title": work_title,
            "work_type": work_type,
            "split_type": split_type,
            "date": date,
            "format": format,
            "contributors": contributors,
        }
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        r = self.session.post(f"{self.base}/splitsheet/v1/documents", json=body, headers=headers, timeout=60)
        if r.status_code != 200:
            raise self._error(r)
        self._record_billing(r.headers)
        return r.content

    def chat(self, messages: list[dict], *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Zoe, non-streaming. `messages` is the OpenAI list of {role, content};
        you supply the whole context (Zoe keeps no memory between calls)."""
        body = {"model": "zoe", "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        out = self._json(self.session.post(f"{self.base}/zoe/v1/chat/completions", json=body, timeout=self.timeout))
        return out["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], **kw):
        """Zoe, streaming: yields text deltas as they arrive."""
        body = {"model": "zoe", "messages": messages, "stream": True, **kw}
        with self.session.post(
            f"{self.base}/zoe/v1/chat/completions", json=body, stream=True, timeout=self.timeout
        ) as r:
            if r.status_code != 200:
                raise self._error(r)
            for line in r.iter_lines(decode_unicode=True):
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                event = json.loads(payload)
                if "error" in event:
                    raise MsaniiError(event["error"].get("code", "zoe_failed"), event["error"].get("message"))
                delta = event["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    # ---- plumbing -----------------------------------------------------------

    def _stream_result(self, url: str, files, data: dict, idempotency_key: str | None) -> dict:
        """One multipart POST answered with server-sent events: heartbeats,
        then exactly one result or error event."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        with self.session.post(url, files=files, data=data, headers=headers, stream=True, timeout=self.timeout) as r:
            if r.status_code != 200:
                raise self._error(r)
            for event in iter_events(r.iter_lines(decode_unicode=True)):
                if event.get("type") == "result":
                    return event
                if event.get("type") == "error":
                    raise MsaniiError(
                        event.get("code", "error"),
                        event.get("message"),
                        event.get("suggestion"),
                        details={k: v for k, v in event.items() if k not in ("type", "code", "message", "suggestion")},
                    )
        raise MsaniiError(
            "no_result", "The connection closed before a result arrived. Retry with the same Idempotency-Key."
        )

    def _record_billing(self, headers) -> None:
        """Read the charge off a document response. Never raises: the file is
        already delivered and charged, so a missing or odd header only means
        we cannot report the number."""
        self.last_billing = {}
        try:
            credits = int(headers.get("Msanii-Credits", ""))
        except (TypeError, ValueError):
            return
        self.last_billing = {
            "credits": credits,
            "request_id": headers.get("Msanii-Request-Id"),
            "replayed": headers.get("Msanii-Replayed") == "true",
        }

    @staticmethod
    def _json(r: requests.Response) -> dict:
        if r.status_code != 200:
            raise MsaniiPartner._error(r)
        return r.json()

    @staticmethod
    def _error(r: requests.Response) -> MsaniiError:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, dict):
            extra = {k: v for k, v in detail.items() if k not in ("code", "message", "suggestion")}
            return MsaniiError(
                detail.get("code", f"http_{r.status_code}"),
                detail.get("message"),
                detail.get("suggestion"),
                r.status_code,
                extra,
            )
        # FastAPI validation errors arrive as a list; anything else as a string.
        return MsaniiError(
            "invalid_request" if r.status_code == 422 else f"http_{r.status_code}",
            str(detail or r.reason),
            None,
            r.status_code,
        )


def _part(f: FilePart) -> tuple[str, bytes]:
    if isinstance(f, tuple):
        return f
    p = Path(f)
    return p.name, p.read_bytes()


def iter_events(lines):
    """Server-sent events → dicts. Skips `: ping` heartbeats; joins multi-line
    `data:` frames; flushes a final frame that has no trailing blank line."""
    buf = []
    for line in lines:
        if line == "":
            if buf:
                yield json.loads("\n".join(buf))
                buf = []
        elif line.startswith("data:"):
            buf.append(line[5:].lstrip())
        # ":" comment lines and any other field are ignored
    if buf:
        yield json.loads("\n".join(buf))


# ---- smoke test -------------------------------------------------------------

# Deterministic fixture: two songs, one net-basis share, one project-wide expense.
#   Blue Sky  1000 gross - 200 expense (1000/1500 of 300) = 800 net * 50% = 400
#   Red Sun    500 gross - 100 expense ( 500/1500 of 300) = 400 net * 50% = 200
SMOKE_STATEMENT = ("statement.csv", b"Title,Net Payable\nBlue Sky,1000.00\nRed Sun,500.00\n")
SMOKE_TERMS = {
    "parties": [{"name": "Jane Doe", "role": "producer"}],
    "works": [{"title": "Blue Sky"}, {"title": "Red Sun"}],
    "royalty_shares": [{"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50, "basis": "net"}],
}
SMOKE_EXPENSES = [{"description": "Mastering", "amount": 300}]
SMOKE_EXPECTED = {"Blue Sky": 400.0, "Red Sun": 200.0}
SMOKE_CONTRIBUTORS = [
    {"name": "Jane Doe", "role": "Producer", "publishing_share": 50, "master_percentage": 50},
    {"name": "Sam Ray", "role": "Writer", "publishing_share": 50, "master_percentage": 50},
]


def _sans_billing(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "billing"}


def smoke(base_url: str, api_key: str, statement: str | None, contract: str | None) -> int:
    api = MsaniiPartner(api_key, base_url)
    failures = 0

    def step(title, fn):
        nonlocal failures
        print(f"\n== {title}")
        try:
            fn()
            print("   ok")
        except AssertionError as e:
            failures += 1
            print(f"   FAIL: {e}")
        except MsaniiError as e:
            failures += 1
            print(f"   FAIL: {e.code} ({e.status}) {e.message or ''} {e.suggestion or ''}".rstrip())
        except requests.RequestException as e:
            failures += 1
            print(f"   FAIL: could not reach {base_url}: {e}")

    def check_models():
        assert api.models() == ["zoe"], "unexpected model list"

    def check_bad_key():
        try:
            MsaniiPartner("mk_live_not_a_real_key", base_url).models()
        except MsaniiError as e:
            assert e.status == 401 and e.code == "invalid_key", (e.status, e.code)
            return
        raise AssertionError("a bogus key was accepted")

    first = {}

    def run_terms(label):
        res = api.calculate(
            SMOKE_STATEMENT, contract_terms=SMOKE_TERMS, expenses=SMOKE_EXPENSES, idempotency_key="smoke-terms"
        )
        got = {p["song"]: round(p["amounts"]["payable"], 2) for p in res["payments"]}
        assert got == SMOKE_EXPECTED, f"expected {SMOKE_EXPECTED}, got {got}"
        assert res["summary"]["expense_review_required"] is True, res
        assert res["summary"]["total_payable"] == 600.0, res["summary"]
        print(f"   {label}: {res['summary']['payments']} payments {got} · {res['billing']['credits']} credits")
        return res

    def check_terms():
        first["result"] = run_terms("first run")

    def check_idempotent():
        # Same key + same inputs: the replay returns the same result and says
        # it was not charged again this billing period.
        before = first.get("result")
        assert before is not None, "the first run failed, so there is nothing to replay"
        res = run_terms("replay, same Idempotency-Key")
        # already_charged is advisory — a failed ledger read over-reports the
        # price, so this can FAIL on correct behaviour; rare, and worth seeing.
        assert res["billing"].get("replayed") is True and res["billing"]["credits"] == 0, res["billing"]
        assert _sans_billing(res) == _sans_billing(before), "replay returned a different result"

    ZOE_ASK = [{"role": "user", "content": "In one sentence, what is a mechanical royalty?"}]

    def check_zoe():
        answer = api.chat(ZOE_ASK, max_tokens=120)
        assert answer, "empty answer"
        print(f"   {answer[:140]}")

    def check_zoe_stream():
        text = "".join(api.chat_stream(ZOE_ASK, max_tokens=120))
        assert text, "empty stream"
        print(f"   {text[:140]}")

    def check_split_sheet():
        pdf = api.split_sheet(
            work_title="Blue Sky",
            date="6 September 2026",
            contributors=SMOKE_CONTRIBUTORS,
            idempotency_key="smoke-sheet",
        )
        assert pdf.startswith(b"%PDF"), "not a PDF"
        assert api.last_billing, "no billing headers on the split sheet"
        print(f"   {len(pdf):,} bytes · {api.last_billing['credits']} credits")

    def check_parse():
        res = api.parse_contract([contract], idempotency_key="smoke-parse")
        terms, splits = res["contract_terms"], res["splits"]
        assert terms["parties"], res
        print(f"   {len(terms['parties'])} parties, {len(terms['works'])} works, {len(terms['royalty_shares'])} shares")
        print(f"   main artist: {splits['main_artist'] or '(not named)'} · {res['billing']['credits']} credits")
        for p in splits["parties"]:
            print(f"   {p['name']}: master {p['master_pct']}% publishing {p['publishing_pct']}%")

    def check_files():
        res = api.calculate(statement, contracts=[contract], idempotency_key="smoke-files")
        assert res["summary"]["payments"] >= 1, res
        for p in res["payments"]:
            print(
                f"   {p['song']}: {p['payee']['name']} {p['share']['percentage']}% [{p['share']['basis']}]"
                f" -> {p['amounts']['payable']:.2f}"
            )
        print(f"   {res['billing']['credits']} credits")

    step("GET /zoe/v1/models (free key check)", check_models)
    step("GET /zoe/v1/models with a bad key -> 401 invalid_key", check_bad_key)
    step("POST /oneclick/v1/royalties — contract_terms + expenses", check_terms)
    step("POST /oneclick/v1/royalties — same request again returns the same result", check_idempotent)
    step("POST /zoe/v1/chat/completions (billed)", check_zoe)
    step("POST /zoe/v1/chat/completions, stream=true (billed)", check_zoe_stream)
    step("POST /splitsheet/v1/documents (billed)", check_split_sheet)
    if contract:
        step(f"POST /registry/v1/splits — contracts=[{Path(contract).name}] (AI parse, billed)", check_parse)
    else:
        print("\n== skipped contract parse (pass --contract to include it)")
    if statement and contract:
        step(f"POST /oneclick/v1/royalties — contracts=[{Path(contract).name}] (AI parse, billed)", check_files)
    else:
        print("\n== skipped PDF-parse calculation (pass --statement and --contract to include it)")

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Smoke-test a Msanii partner API key.")
    ap.add_argument("--url", default=os.environ.get("MSANII_API_URL"), help="base URL (env MSANII_API_URL)")
    ap.add_argument("--key", default=os.environ.get("MSANII_API_KEY"), help="mk_live_… (env MSANII_API_KEY)")
    ap.add_argument("--statement", help="CSV/XLSX statement for the PDF-parse calculation (needs --contract)")
    ap.add_argument("--contract", help="contract PDF for the parse and PDF-parse runs")
    args = ap.parse_args(argv)
    if not args.url or not args.key:
        ap.error("set MSANII_API_URL and MSANII_API_KEY (or pass --url / --key)")
    if args.statement and not args.contract:
        ap.error("--statement needs --contract")
    return smoke(args.url, args.key, args.statement, args.contract)


if __name__ == "__main__":
    sys.exit(main())
