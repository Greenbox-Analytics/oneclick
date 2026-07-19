"""402 walls + charge-on-success at the credit-gated endpoints.

Covers Task 9 wiring:
- Zoe / OneClick (stream + JSON) / Registry parse raise structured 402s when
  the credit check denies.
- Debits fire ONLY from terminal-success events (charge-on-success, spec §3):
  Zoe's `done`/`complete`, OneClick's fresh-computation completion, Registry's
  cache-miss parse. Error paths and cache hits never charge.
- create_work is gated by the max_works cap via gated_create.
- get_or_parse invokes on_miss exactly when a live LLM parse happens.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
import subscriptions.enforcement as enforcement
from subscriptions.models import Action, CheckResult, CreditCheckResult
from tests.conftest import TEST_USER_ID, MockQueryBuilder, _default_table_side_effect

PROJECT_ID = "proj-0000-0000-0000-0000-000000000001"
CONTRACT_ID = "cont-0000-0000-0000-0000-000000000001"
STATEMENT_FILE_ID = "file-0000-0000-0000-0000-000000000001"
CALCULATION_ID = "calc-0000-0000-0000-0000-000000000001"
ARTIST_ID = "aaaa0000-0000-0000-0000-000000000001"

SAMPLE_STATEMENT_FILE = {
    "id": STATEMENT_FILE_ID,
    "file_path": "project-files/statement.xlsx",
    "file_name": "royalty_statement.xlsx",
}

SAMPLE_PAYMENTS = [
    {
        "song_title": "Blue Sky",
        "party_name": "Jane Doe",
        "role": "Producer",
        "royalty_type": "master",
        "percentage": 50.0,
        "total_royalty": 1000.0,
        "amount_to_pay": 500.0,
        "terms": None,
    },
]


@pytest.fixture(autouse=True)
def _bypass_oneclick_ownership(monkeypatch):
    """These tests target the credit gate + debit wiring, not access control
    (ownership guard correctness is covered by test_oneclick_ownership.py).
    The JSON endpoint runs the access check BEFORE its gate, so it must be
    bypassed for the credit wall to be what fires."""
    monkeypatch.setattr(
        "main._assert_can_access_oneclick_inputs",
        AsyncMock(return_value=None),
    )


@pytest.fixture()
def broke_free_service(monkeypatch):
    """Every credit check denies with the free-tier wall; legacy can() allows."""
    svc = MagicMock()
    svc.check_credits.return_value = CreditCheckResult(
        allowed=False,
        price=3,
        reason="You've used this month's credits.",
        upgrade_required=True,
    )
    svc.can.return_value = CheckResult(allowed=True, reason=None, upgrade_required=False)
    monkeypatch.setattr(enforcement, "_service", lambda: svc)
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    return svc


@pytest.fixture()
def credit_service(monkeypatch):
    """Every credit check allows at a fixed price; legacy can() allows."""
    svc = MagicMock()
    svc.check_credits.return_value = CreditCheckResult(allowed=True, price=3)
    svc.can.return_value = CheckResult(allowed=True, reason=None, upgrade_required=False)
    monkeypatch.setattr(enforcement, "_service", lambda: svc)
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    return svc


@pytest.fixture()
def debit_spy(monkeypatch):
    """Spy on the EntitlementsService the debit call sites in main.py resolve."""
    ent = MagicMock()
    monkeypatch.setattr(main, "_get_entitlements_service", lambda: ent)
    return ent


def _make_chatbot(events):
    chatbot = MagicMock()
    chatbot.ask_stream.return_value = iter(events)
    return chatbot


# ---------------------------------------------------------------------------
# 402 walls
# ---------------------------------------------------------------------------


class TestCreditWalls:
    def test_zoe_402_when_broke(self, client, broke_free_service):
        resp = client.post("/zoe/ask-stream", json={"query": "what is my split?", "contract_ids": []})
        assert resp.status_code == 402
        detail = resp.json()["detail"]
        assert detail["upgradeRequired"] is True
        assert detail["reason"] == "You've used this month's credits."
        broke_free_service.check_credits.assert_called_once_with(TEST_USER_ID, "zoe_message", is_admin=False)

    def test_oneclick_json_402_when_broke(self, client, broke_free_service):
        resp = client.post(
            "/oneclick/calculate-royalties",
            json={
                "project_id": PROJECT_ID,
                "royalty_statement_file_id": STATEMENT_FILE_ID,
                "contract_ids": [CONTRACT_ID],
            },
        )
        assert resp.status_code == 402
        assert resp.json()["detail"]["upgradeRequired"] is True
        broke_free_service.check_credits.assert_called_once_with(TEST_USER_ID, "oneclick_run", is_admin=False)

    def test_oneclick_stream_402_when_broke(self, client, broke_free_service):
        resp = client.get(
            "/oneclick/calculate-royalties-stream",
            params={
                "project_id": PROJECT_ID,
                "royalty_statement_file_id": STATEMENT_FILE_ID,
                "contract_ids": [CONTRACT_ID],
            },
        )
        assert resp.status_code == 402
        assert resp.json()["detail"]["upgradeRequired"] is True

    def test_registry_parse_402_when_broke(self, client, broke_free_service):
        # gated_feature(USE_REGISTRY) passes (can() allows); the credit gate fires.
        resp = client.post("/registry/parse-contract-splits", data={"contract_file_id": CONTRACT_ID})
        assert resp.status_code == 402
        assert resp.json()["detail"]["upgradeRequired"] is True
        broke_free_service.check_credits.assert_called_once_with(TEST_USER_ID, "registry_parse", is_admin=False)


# ---------------------------------------------------------------------------
# max_works cap on create_work
# ---------------------------------------------------------------------------


class TestWorksCap:
    @staticmethod
    def _deny_create_work_service(monkeypatch):
        svc = MagicMock()

        def _can(user_id, action, **ctx):
            if action == Action.CREATE_WORK:
                return CheckResult(
                    allowed=False,
                    reason="You've reached your limit of 10 registered works.",
                    upgrade_required=True,
                )
            # USE_REGISTRY feature gate (checked first) must pass so the cap fires.
            return CheckResult(allowed=True, reason=None, upgrade_required=False)

        svc.can.side_effect = _can
        monkeypatch.setattr(enforcement, "_service", lambda: svc)
        return svc

    def test_work_creation_gated_by_max_works(self, client, monkeypatch):
        # max_works ships with the credits launch — the gate is live only when
        # CREDITS_ENABLED is on.
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        svc = self._deny_create_work_service(monkeypatch)

        resp = client.post(
            "/registry/works",
            json={"artist_id": ARTIST_ID, "project_id": PROJECT_ID, "title": "Song"},
        )
        assert resp.status_code == 402
        assert "limit of 10 registered works" in resp.json()["detail"]
        cap_calls = [c for c in svc.can.call_args_list if c.args[1] == Action.CREATE_WORK]
        assert len(cap_calls) == 1
        # MockQueryBuilder's count for works_registry is 0 by default.
        assert cap_calls[0].kwargs["current_count"] == 0

    def test_work_creation_not_gated_when_flag_off(self, client, mock_supabase, monkeypatch):
        # Flag OFF (conftest default): the cap must never be consulted, even
        # though can() would deny CREATE_WORK — the migration seeds the cap but
        # it must not bite before the flag flips.
        svc = self._deny_create_work_service(monkeypatch)

        sample_work = {
            "id": "aaaa0000-0000-0000-0000-0000000000ff",
            "user_id": TEST_USER_ID,
            "artist_id": ARTIST_ID,
            "project_id": PROJECT_ID,
            "title": "Song",
            "work_type": "single",
            "status": "draft",
        }
        artist_builder = MockQueryBuilder()
        artist_builder.execute.return_value = MagicMock(data={"id": ARTIST_ID})
        work_builder = MockQueryBuilder()
        work_builder.execute.return_value = MagicMock(data=[sample_work])

        def _side_effect(name):
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"):
                return _default_table_side_effect(name)
            if name == "artists":
                return artist_builder
            return work_builder

        mock_supabase.table.side_effect = _side_effect

        resp = client.post(
            "/registry/works",
            json={"artist_id": ARTIST_ID, "project_id": PROJECT_ID, "title": "Song"},
        )
        assert resp.status_code == 200
        cap_calls = [c for c in svc.can.call_args_list if c.args[1] == Action.CREATE_WORK]
        assert cap_calls == []


# ---------------------------------------------------------------------------
# Charge-on-success: Zoe
# ---------------------------------------------------------------------------


class TestZoeChargeOnSuccess:
    def _post(self, client, chatbot, query="What is the rate?"):
        with patch("main.get_zoe_chatbot", return_value=chatbot):
            return client.post("/zoe/ask-stream", json={"query": query})

    def test_debits_once_on_done_event(self, client, credit_service, debit_spy):
        chatbot = _make_chatbot(
            [
                'data: {"type": "token", "content": "Hello"}\n\n',
                'data: {"type": "done"}\n\n',
            ]
        )
        resp = self._post(client, chatbot)
        assert resp.status_code == 200
        debit_spy.debit_for_action.assert_called_once()
        uid, grant = debit_spy.debit_for_action.call_args.args
        assert uid == TEST_USER_ID
        assert grant.action == "zoe_message" and grant.enabled and grant.price == 3

    def test_debits_on_complete_event(self, client, credit_service, debit_spy):
        # `complete` is the instant non-streamed terminal event (tiers 1/2).
        chatbot = _make_chatbot(['data: {"type": "complete", "answer": "50%"}\n\n'])
        resp = self._post(client, chatbot)
        assert resp.status_code == 200
        debit_spy.debit_for_action.assert_called_once()

    def test_no_debit_on_error_event(self, client, credit_service, debit_spy):
        # The chatbot catches its own failures and yields a terminal `error`
        # event without raising — that stream must not charge.
        chatbot = _make_chatbot(['data: {"type": "error", "message": "boom"}\n\n'])
        resp = self._post(client, chatbot)
        assert resp.status_code == 200
        debit_spy.debit_for_action.assert_not_called()

    def test_conversational_reply_not_charged(self, client, credit_service, debit_spy):
        # Tier-1 conversational fast path ("hi"/"thanks") is a canned string
        # with zero LLM calls — FREE (spec: meter only what costs money).
        # Payload shape mirrors _handle_conversational_query (contract_chatbot).
        chatbot = _make_chatbot(
            ['data: {"type": "complete", "answer": "Hey there!", "confidence": "conversational", "sources": []}\n\n']
        )
        resp = self._post(client, chatbot, query="hi")
        assert resp.status_code == 200
        debit_spy.debit_for_action.assert_not_called()

    def test_context_cleared_notice_not_charged(self, client, credit_service, debit_spy):
        # The context-reset notice involves no LLM call — FREE. Payload shape
        # mirrors the pending-suggestion reset branch in ask_stream.
        chatbot = _make_chatbot(
            [
                'data: {"type": "complete", "answer": "It looks like the conversation context was reset.", '
                '"confidence": "context_cleared", "context_cleared": true, "sources": []}\n\n'
            ]
        )
        resp = self._post(client, chatbot)
        assert resp.status_code == 200
        debit_spy.debit_for_action.assert_not_called()

    def test_double_terminal_events_debit_same_grant(self, client, credit_service, debit_spy):
        # Invariant pin: if a stream ever carries BOTH terminal-success events,
        # the debit fires twice with the SAME grant object — correctness rests
        # on the debit RPC deduping by the grant's request_id, by design.
        chatbot = _make_chatbot(
            [
                'data: {"type": "complete", "answer": "50%"}\n\n',
                'data: {"type": "done"}\n\n',
            ]
        )
        resp = self._post(client, chatbot)
        assert resp.status_code == 200
        assert debit_spy.debit_for_action.call_count == 2
        first_grant = debit_spy.debit_for_action.call_args_list[0].args[1]
        second_grant = debit_spy.debit_for_action.call_args_list[1].args[1]
        assert first_grant is second_grant

    def test_no_debit_when_stream_dies_midway(self, client, credit_service, debit_spy):
        def _explodes():
            yield 'data: {"type": "token", "content": "par"}\n\n'
            raise RuntimeError("OpenAI timeout")

        chatbot = MagicMock()
        chatbot.ask_stream.return_value = _explodes()
        resp = self._post(client, chatbot)
        assert resp.status_code == 200  # SSE stream carries the error event
        debit_spy.debit_for_action.assert_not_called()


# ---------------------------------------------------------------------------
# Charge-on-success: OneClick streaming
# ---------------------------------------------------------------------------


class TestOneClickStreamCharge:
    def _params(self):
        return {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_ids": [CONTRACT_ID],
        }

    def test_fresh_run_debits_once(self, client, mock_supabase, credit_service, debit_spy):
        mock_supabase.table.side_effect = lambda name: (
            _default_table_side_effect(name)
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles")
            else self._builder([SAMPLE_STATEMENT_FILE] if name == "project_files" else [])
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)):
            resp = client.get("/oneclick/calculate-royalties-stream", params=self._params())

        assert resp.status_code == 200
        assert '"type": "complete"' in resp.text
        debit_spy.debit_for_action.assert_called_once()
        uid, grant = debit_spy.debit_for_action.call_args.args
        assert uid == TEST_USER_ID
        assert grant.action == "oneclick_run" and grant.enabled

    def test_cached_run_is_free(self, client, mock_supabase, credit_service, debit_spy):
        cached_result = {
            "status": "success",
            "total_payments": 1,
            "payments": [SAMPLE_PAYMENTS[0]],
            "message": "1 payment",
        }
        sequences = [
            [{"id": CALCULATION_ID, "results": cached_result}],  # royalty_calculations
            [{"calculation_id": CALCULATION_ID, "contract_id": CONTRACT_ID}],  # junction
        ]
        call_idx = [0]

        def _side_effect(name):
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"):
                return _default_table_side_effect(name)
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return self._builder(data)

        mock_supabase.table.side_effect = _side_effect

        with patch("main.calculate_royalty_payments") as mock_calc:
            resp = client.get("/oneclick/calculate-royalties-stream", params=self._params())
            mock_calc.assert_not_called()

        assert resp.status_code == 200
        assert '"is_cached": true' in resp.text
        # Cached re-runs are FREE (spec §3) — no ledger debit.
        debit_spy.debit_for_action.assert_not_called()

    def test_failed_run_no_debit(self, client, mock_supabase, credit_service, debit_spy):
        mock_supabase.table.side_effect = lambda name: (
            _default_table_side_effect(name)
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles")
            else self._builder([SAMPLE_STATEMENT_FILE] if name == "project_files" else [])
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch("main.calculate_royalty_payments", side_effect=RuntimeError("parser exploded")):
            resp = client.get("/oneclick/calculate-royalties-stream", params=self._params())

        assert resp.status_code == 200  # SSE stream carries the error event
        assert '"type": "error"' in resp.text
        debit_spy.debit_for_action.assert_not_called()

    @staticmethod
    def _builder(data):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=data, count=len(data))
        return b


# ---------------------------------------------------------------------------
# Charge-on-success: OneClick JSON endpoint
# ---------------------------------------------------------------------------


class TestOneClickJsonCharge:
    def _post(self, client):
        return client.post(
            "/oneclick/calculate-royalties",
            json={
                "project_id": PROJECT_ID,
                "royalty_statement_file_id": STATEMENT_FILE_ID,
                "contract_ids": [CONTRACT_ID],
            },
        )

    def _wire_statement(self, mock_supabase):
        def _side_effect(name):
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"):
                return _default_table_side_effect(name)
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(
                data=[SAMPLE_STATEMENT_FILE] if name == "project_files" else [],
                count=0,
            )
            return b

        mock_supabase.table.side_effect = _side_effect
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

    def test_success_debits_once(self, client, mock_supabase, credit_service, debit_spy):
        self._wire_statement(mock_supabase)
        with patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)):
            resp = self._post(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        debit_spy.debit_for_action.assert_called_once()
        uid, grant = debit_spy.debit_for_action.call_args.args
        assert uid == TEST_USER_ID
        assert grant.action == "oneclick_run" and grant.enabled

    def test_failure_no_debit(self, client, mock_supabase, credit_service, debit_spy):
        self._wire_statement(mock_supabase)
        with patch("main.calculate_royalty_payments", side_effect=RuntimeError("parser exploded")):
            resp = self._post(client)
        assert resp.status_code == 500
        debit_spy.debit_for_action.assert_not_called()


# ---------------------------------------------------------------------------
# Charge-on-success: Registry parse (cache-aware)
# ---------------------------------------------------------------------------


class TestRegistryParseCharge:
    def _wire(self, client, mock_supabase, monkeypatch, *, miss: bool):
        """Drive POST /registry/parse-contract-splits with get_or_parse faked to
        either report a cache miss (live parse) or a silent cache hit."""
        ent = MagicMock()
        monkeypatch.setattr("subscriptions.deps._get_entitlements_service", lambda: ent)

        def _side_effect(name):
            if name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"):
                return _default_table_side_effect(name)
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(
                data={"file_path": "c/a.pdf", "file_name": "a.pdf", "contract_markdown": "# Contract"}
            )
            return b

        mock_supabase.table.side_effect = _side_effect

        def _fake_get_or_parse(db, load_text, *, on_miss=None, **kwargs):
            if miss and on_miss is not None:
                on_miss()
            return MagicMock()  # ContractData stand-in; pivot is patched below

        with (
            patch("main.verify_user_owns_contract", return_value=True),
            patch("utils.contract_parsing.cache.get_or_parse", side_effect=_fake_get_or_parse),
            patch(
                "registry.contract_splits.parse_royalty_splits",
                return_value={"parties": [], "main_artist_found": False},
            ),
        ):
            resp = client.post("/registry/parse-contract-splits", data={"contract_file_id": CONTRACT_ID})
        return resp, ent

    def test_cache_miss_debits_once(self, client, mock_supabase, credit_service, monkeypatch):
        resp, ent = self._wire(client, mock_supabase, monkeypatch, miss=True)
        assert resp.status_code == 200
        ent.debit_for_action.assert_called_once()
        uid, grant = ent.debit_for_action.call_args.args
        assert uid == TEST_USER_ID
        assert grant.action == "registry_parse" and grant.enabled

    def test_cache_hit_is_free(self, client, mock_supabase, credit_service, monkeypatch):
        resp, ent = self._wire(client, mock_supabase, monkeypatch, miss=False)
        assert resp.status_code == 200
        # Cache hits never charge (spec §3).
        ent.debit_for_action.assert_not_called()


# ---------------------------------------------------------------------------
# get_or_parse on_miss semantics (unit)
# ---------------------------------------------------------------------------


class TestGetOrParseOnMiss:
    @staticmethod
    def _parser(result=None):
        parser = MagicMock()
        parser.parse_contract.return_value = result or MagicMock()
        return parser

    def test_fires_on_live_parse_without_db(self):
        from utils.contract_parsing.cache import get_or_parse

        fired = []
        parser = self._parser()
        get_or_parse(None, lambda: "contract text", parser=parser, on_miss=lambda: fired.append(1))
        assert fired == [1]
        parser.parse_contract.assert_called_once()

    def test_fires_on_bypass_even_with_cached_entry(self):
        from utils.contract_parsing.cache import get_or_parse

        fired = []
        db = MagicMock()
        get_or_parse(db, lambda: "contract text", parser=self._parser(), bypass=True, on_miss=lambda: fired.append(1))
        assert fired == [1]

    def test_not_fired_on_cache_hit(self):
        from utils.contract_parsing.cache import get_or_parse

        cached_payload = {
            "parties": [],
            "works": [],
            "royalty_shares": [],
            "contract_summary": None,
            "default_basis": None,
        }
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data={"parsed": cached_payload})
        db = MagicMock()
        db.table.side_effect = lambda name: b

        fired = []
        parser = self._parser()
        get_or_parse(db, lambda: "contract text", parser=parser, on_miss=lambda: fired.append(1))
        assert fired == []
        parser.parse_contract.assert_not_called()

    def test_observer_exception_never_breaks_the_parse(self):
        from utils.contract_parsing.cache import get_or_parse

        sentinel = MagicMock()
        parser = self._parser(result=sentinel)

        def _explodes():
            raise RuntimeError("observer bug")

        out = get_or_parse(None, lambda: "contract text", parser=parser, on_miss=_explodes)
        assert out is sentinel
