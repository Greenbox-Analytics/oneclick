"""Partner API abuse controls: per-key rate limit, the request log, and the
"whose key is this?" lookup.

These sit on the auth chokepoint every partner route shares, so the load-bearing
test is the DEPENDENCY, not the helpers underneath it — a route added tomorrow
inherits all three or none.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from partner_api import router as prouter
from partner_api import service as psvc
from tests.conftest import MockQueryBuilder

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
KEY_ID = "00000000-0000-0000-0000-0000000000bb"
SECRET = "mk_live_" + "z" * 43

ACTIVE_ORG = {
    "id": ORG_ID,
    "name": "Acme Records",
    "status": "active",
    "archived_at": None,
    "partner_api_enabled": True,
}
ACTIVE_KEY = {"id": KEY_ID, "org_id": ORG_ID, "status": "active", "expires_at": None}


def _request(xff=None, ua="acme-sdk/1.0", path="/oneclick/v1/royalties", peer="10.0.0.1"):
    headers = {"user-agent": ua}
    if xff:
        headers["x-forwarded-for"] = xff
    return SimpleNamespace(
        headers=headers,
        url=SimpleNamespace(path=path),
        client=SimpleNamespace(host=peer),
    )


def _sb(*, key_rows=(ACTIVE_KEY,), org_row=ACTIVE_ORG, request_count=0, request_rows=()):
    keys = MockQueryBuilder()
    keys.execute = MagicMock(return_value=MagicMock(data=list(key_rows)))
    orgs = MockQueryBuilder()
    orgs.execute = MagicMock(return_value=MagicMock(data=[org_row] if org_row else []))
    reqs = MockQueryBuilder()
    reqs.execute = MagicMock(return_value=MagicMock(data=list(request_rows), count=request_count))
    sb = MagicMock()
    sb.table.side_effect = lambda name: {
        "partner_api_keys": keys,
        "organizations": orgs,
        "partner_api_requests": reqs,
    }[name]
    return sb, reqs


def _inserted(reqs):
    return reqs.insert.call_args[0][0]


# ---- the chokepoint ---------------------------------------------------------


def test_a_key_under_its_ceiling_passes_and_is_logged():
    sb, reqs = _sb(request_count=59)
    with patch.object(prouter, "_get_supabase", return_value=sb):
        ctx = prouter.get_partner_context(_request(), f"Bearer {SECRET}")
    assert ctx == psvc.PartnerContext(org_id=ORG_ID, key_id=KEY_ID)
    row = _inserted(reqs)
    assert row["outcome"] == "ok"
    assert row["key_id"] == KEY_ID and row["org_id"] == ORG_ID


def test_a_key_at_its_ceiling_gets_429_with_retry_after():
    """The whole point: a leaked key was previously bounded only by the org's
    credit pool, which it can drain in an afternoon."""
    sb, reqs = _sb(request_count=60)
    with patch.object(prouter, "_get_supabase", return_value=sb), pytest.raises(HTTPException) as exc:
        prouter.get_partner_context(_request(), f"Bearer {SECRET}")
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "rate_limited"
    assert exc.value.headers["Retry-After"] == "60"
    # Refusals are logged and COUNTED, so a flood stays refused rather than
    # oscillating either side of the limit.
    assert _inserted(reqs)["outcome"] == "rate_limited"


def test_a_failed_auth_is_logged_and_never_records_the_secret():
    """Nothing else records a 401, so a burst of these is the only signal that
    a revoked or guessed key is being hammered."""
    sb, reqs = _sb(key_rows=())
    with patch.object(prouter, "_get_supabase", return_value=sb), pytest.raises(HTTPException) as exc:
        prouter.get_partner_context(_request(), f"Bearer {SECRET}")
    assert exc.value.status_code == 401
    row = _inserted(reqs)
    assert row["outcome"] == "invalid_key" and row["key_id"] is None
    assert row["key_prefix"] == SECRET[:12]
    assert SECRET not in str(row), "the presented secret must never be stored"


def test_a_foreign_secret_pasted_into_the_header_leaves_no_fragment():
    """A caller mis-pasting an OpenAI key must not deposit 12 characters of it
    in our database."""
    assert psvc.presented_prefix("sk-proj-abcdefghijklmnop") is None
    assert psvc.presented_prefix(None) is None
    assert psvc.presented_prefix(SECRET) == SECRET[:12]


def test_the_logged_ip_is_the_hop_the_caller_cannot_forge():
    """Cloud Run APPENDS the real client to any X-Forwarded-For the caller sent,
    so the LAST entry is trustworthy and everything before it is attacker-
    supplied. Reading the first would make the whole signal forgeable."""
    assert psvc.client_ip(_request(xff="1.2.3.4, 203.0.113.9")) == "203.0.113.9"
    assert psvc.client_ip(_request(xff=None)) == "10.0.0.1"


def test_the_user_agent_is_truncated_before_storage():
    sb, reqs = _sb()
    psvc.log_request(sb, _request(ua="A" * 5000), outcome="ok", key_id=KEY_ID)
    assert len(_inserted(reqs)["user_agent"]) == psvc.MAX_USER_AGENT


# ---- the limiter itself -----------------------------------------------------


def test_the_limiter_can_be_switched_off_without_a_deploy():
    sb, _ = _sb(request_count=10_000)
    with patch.dict("os.environ", {"PARTNER_RATE_LIMIT_PER_MIN": "0"}):
        assert psvc.rate_limited(sb, KEY_ID) is False


def test_a_broken_counter_fails_open():
    """A limiter outage must not become an API outage for every partner — the
    pool balance is still a hard ceiling underneath."""
    sb, reqs = _sb()
    reqs.execute = MagicMock(side_effect=RuntimeError("pg down"))
    assert psvc.rate_limited(sb, KEY_ID) is False


def test_the_window_is_the_last_sixty_seconds():
    sb, reqs = _sb(request_count=0)
    reqs.gte = MagicMock(return_value=reqs)
    psvc.rate_limited(sb, KEY_ID)
    column, since = reqs.gte.call_args[0]
    assert column == "created_at"
    age = datetime.now(UTC) - datetime.fromisoformat(since)
    assert timedelta(seconds=55) < age < timedelta(seconds=65)


def test_an_audit_write_failure_never_fails_the_request():
    sb, reqs = _sb()
    reqs.insert = MagicMock(side_effect=RuntimeError("pg down"))
    psvc.log_request(sb, _request(), outcome="ok", key_id=KEY_ID)  # must not raise


# ---- the lookup -------------------------------------------------------------


def test_lookup_names_the_org_and_the_ips_a_leak_would_show():
    """The operator path: found a key, need the org to revoke it in — and the
    source IPs, which are what separates a busy partner from a leak."""
    revoked = {
        "id": KEY_ID,
        "org_id": ORG_ID,
        "label": "Production",
        "key_prefix": SECRET[:12],
        "status": "revoked",
        "expires_at": None,
    }
    seen = [
        {"client_ip": "203.0.113.9", "created_at": "2026-09-07T10:00:00+00:00"},
        {"client_ip": "198.51.100.4", "created_at": "2026-09-07T09:00:00+00:00"},
        {"client_ip": "203.0.113.9", "created_at": "2026-09-06T09:00:00+00:00"},
    ]
    sb, _ = _sb(key_rows=(revoked,), request_rows=seen)

    # Pasting the WHOLE key works — only the prefix is ever used.
    (hit,) = psvc.lookup_by_prefix(sb, SECRET)
    assert hit["org_id"] == ORG_ID and hit["org_name"] == "Acme Records"
    # Included on purpose: a key leaked a year ago and already revoked is
    # exactly what someone looks up.
    assert hit["status"] == "revoked"
    assert [ip["ip"] for ip in hit["recent_ips"]] == ["203.0.113.9", "198.51.100.4"]
    assert hit["recent_ips"][0]["requests"] == 2
    # Newest-first rows, so the first sighting is the last_seen.
    assert hit["recent_ips"][0]["last_seen"] == "2026-09-07T10:00:00+00:00"


def test_lookup_refuses_anything_that_is_not_one_of_our_keys():
    sb, _ = _sb()
    with pytest.raises(ValueError):
        psvc.lookup_by_prefix(sb, "sk-proj-notours")


def test_lookup_returns_a_list_because_the_prefix_can_collide():
    """After mk_live_ the prefix is 4 characters — unlikely to collide, not
    impossible, and silently returning one of two keys would revoke the wrong
    partner."""
    twin = {"id": "other", "org_id": ORG_ID, "key_prefix": SECRET[:12], "status": "active", "expires_at": None}
    sb, _ = _sb(key_rows=({**twin, "id": KEY_ID}, twin))
    assert len(psvc.lookup_by_prefix(sb, SECRET[:12])) == 2


# ---- retention --------------------------------------------------------------


def test_the_purge_drops_only_rows_past_the_retention_window():
    sb, reqs = _sb()
    reqs.delete.return_value.lt.return_value.execute.return_value = MagicMock(data=[{"id": 1}, {"id": 2}])
    with patch.dict("os.environ", {"PARTNER_LOG_RETENTION_DAYS": "30"}):
        assert psvc.purge_request_log(sb) == 2
    column, cutoff = reqs.delete.return_value.lt.call_args[0]
    assert column == "created_at"
    age = datetime.now(UTC) - datetime.fromisoformat(cutoff)
    assert timedelta(days=29) < age < timedelta(days=31)
