"""Pending tester claim inside POST /me/bootstrap-tester (admin credits &
testers spec). Own file on purpose: test_bootstrap_tester.py shadows
conftest's client fixture (own auth overrides, hardcoded email, no supabase
mock) — here conftest's client + mock_supabase wiring applies."""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_EMAIL, TEST_USER_ID


class TestPendingClaim:
    """Pre-signup designation claim (admin credits & testers spec)."""

    def _wire_pending(self, mock_supabase, pending_row, claim_updates_row=True, claimed_by=None):
        def table_side_effect(name):
            t = MagicMock()
            if name == "pending_tester_grants":
                t.select.return_value.eq.return_value.is_.return_value.execute.return_value.data = (
                    [pending_row] if pending_row else []
                )
                t.update.return_value.eq.return_value.is_.return_value.execute.return_value.data = (
                    [pending_row] if claim_updates_row else []
                )
                # Race-loser re-read (claim_pending_tester_grant): no `.is_()`
                # in this chain, so it's a DIFFERENT stub than the pending-row
                # select above — left unstubbed, a bare MagicMock's truthy
                # `.data` would misreport a claim that never happened.
                t.select.return_value.eq.return_value.execute.return_value.data = (
                    [{"claimed_user_id": claimed_by}] if claimed_by else []
                )
            elif name == "tier_overrides":
                t.select.return_value.eq.return_value.like.return_value.execute.return_value.data = []
            elif name == "credit_wallets":
                t.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{"id": "w-1"}]
            return t

        mock_supabase.table.side_effect = table_side_effect

    def _confirmed_user(self, mock_supabase, confirmed=True):
        user = MagicMock()
        user.email_confirmed_at = "2026-08-08T00:00:00Z" if confirmed else None
        wrapper = MagicMock()
        wrapper.user = user
        mock_supabase.auth.admin.get_user_by_id.return_value = wrapper

    def test_claims_pending_row_and_reports_source(self, client, mock_supabase):
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": 30,
                "credits": 500,
                "claimed_at": None,
            },
        )
        self._confirmed_user(mock_supabase)
        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        assert resp.json() == {"granted": True, "source": "pending"}

    def test_unverified_email_does_not_claim(self, client, mock_supabase, monkeypatch):
        monkeypatch.delenv("TESTER_EMAILS", raising=False)  # deterministic fall-through
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": None,
                "credits": None,
                "claimed_at": None,
            },
        )
        self._confirmed_user(mock_supabase, confirmed=False)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json()["granted"] is False  # falls through to env path

    def test_no_pending_row_uses_env_path(self, client, mock_supabase, monkeypatch):
        monkeypatch.delenv("TESTER_EMAILS", raising=False)  # deterministic fall-through
        self._wire_pending(mock_supabase, None)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json() == {"granted": False, "reason": "not_in_allowlist"}

    def test_race_loser_same_user_reports_granted(self, client, mock_supabase):
        """Two concurrent sign-ins for the SAME user: this call's conditional
        UPDATE matches zero rows (the other sign-in already claimed it), but
        the re-read shows claimed_user_id == this user — report success so
        AuthContext still refreshes."""
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": 30,
                "credits": 500,
                "claimed_at": None,
            },
            claim_updates_row=False,
            claimed_by=TEST_USER_ID,
        )
        self._confirmed_user(mock_supabase)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json() == {"granted": True, "source": "pending"}

    def test_race_loser_claimed_by_other_falls_through(self, client, mock_supabase, monkeypatch):
        """The conditional UPDATE matched zero rows and the re-read shows a
        DIFFERENT user won the claim — this call must not report success,
        and falls through to the (deterministically empty) env allowlist
        check."""
        monkeypatch.delenv("TESTER_EMAILS", raising=False)  # deterministic fall-through
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": 30,
                "credits": 500,
                "claimed_at": None,
            },
            claim_updates_row=False,
            claimed_by="someone-else",
        )
        self._confirmed_user(mock_supabase)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json()["granted"] is False

    def test_credits_off_claim_still_completes(self, client, mock_supabase):
        """Parity with the existing admin-grant path: with CREDITS_ENABLED off
        (the conftest autouse default — no monkeypatch needed),
        grant_initial_tester_credits skips with a warn but the claim itself
        must still land (tester status is flag-independent)."""
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": None,
                "credits": 500,
                "claimed_at": None,
            },
        )
        self._confirmed_user(mock_supabase)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json() == {"granted": True, "source": "pending"}
        mock_supabase.rpc.assert_not_called()

    def test_credits_on_claim_grants_initial_credits(self, client, mock_supabase, monkeypatch):
        """The credits-ON path — conftest's autouse fixture disables credits for
        every test, so without this setenv NO claim test would exercise the
        grant RPC at all. Asserts the once-ever tester-init key and the pending
        row's credit override."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        self._wire_pending(
            mock_supabase,
            {
                "email": TEST_USER_EMAIL.lower(),
                "reason": "tester",
                "grant_duration_days": 30,
                "credits": 500,
                "claimed_at": None,
            },
        )
        self._confirmed_user(mock_supabase)
        resp = client.post("/me/bootstrap-tester")
        assert resp.json() == {"granted": True, "source": "pending"}
        rpc_args = mock_supabase.rpc.call_args
        assert rpc_args.args[0] == "grant_credits"
        assert rpc_args.args[1]["p_request_id"] == f"tester-init:{TEST_USER_ID}"
        assert rpc_args.args[1]["p_amount"] == 500

    def test_credits_rpc_failure_does_not_block_claim(self, client, mock_supabase, monkeypatch):
        """Regression: grant_initial_tester_credits' grant_credits RPC blowing up
        must not block the claim. The tier_overrides upsert already landed by
        that point, so an unswallowed exception would skip the claim update,
        misreport granted=False, and desync the pending row (it stays
        unclaimed forever even though the user is now a tester)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        pending_row = {
            "email": TEST_USER_EMAIL.lower(),
            "reason": "tester",
            "grant_duration_days": 30,
            "credits": 500,
            "claimed_at": None,
        }
        # Own table double (not _wire_pending): pending_tester_grants must be the
        # SAME mock across the select + update calls so `.update` is assertable —
        # table_side_effect normally mints a fresh MagicMock per call.
        pending_table = MagicMock()
        pending_table.select.return_value.eq.return_value.is_.return_value.execute.return_value.data = [pending_row]
        pending_table.update.return_value.eq.return_value.is_.return_value.execute.return_value.data = [pending_row]

        def table_side_effect(name):
            if name == "pending_tester_grants":
                return pending_table
            t = MagicMock()
            if name == "tier_overrides":
                t.select.return_value.eq.return_value.like.return_value.execute.return_value.data = []
            elif name == "credit_wallets":
                t.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{"id": "w-1"}]
            return t

        mock_supabase.table.side_effect = table_side_effect
        mock_supabase.rpc.return_value.execute.side_effect = Exception("rpc down")
        self._confirmed_user(mock_supabase)

        resp = client.post("/me/bootstrap-tester")
        assert resp.json() == {"granted": True, "source": "pending"}
        pending_table.update.assert_called_once()
