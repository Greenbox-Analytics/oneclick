import hashlib
import hmac
import time
from unittest.mock import patch


def _sign(secret: str, ts: str, body: bytes) -> str:
    base = f"v0:{ts}:".encode() + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_webhook_rejects_bad_signature(client, monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "shhh")
    with patch("integrations.slack.router._process_app_mention") as proc:
        resp = client.post(
            "/integrations/slack/webhook",
            content=b'{"type":"event_callback"}',
            headers={"X-Slack-Signature": "v0=bad", "X-Slack-Request-Timestamp": str(int(time.time()))},
        )
    assert resp.status_code == 403
    proc.assert_not_called()


def test_webhook_accepts_valid_signature(client, monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "shhh")
    ts = str(int(time.time()))
    body = b'{"type":"url_verification","challenge":"abc"}'
    sig = _sign("shhh", ts, body)
    resp = client.post(
        "/integrations/slack/webhook",
        content=body,
        headers={"X-Slack-Signature": sig, "X-Slack-Request-Timestamp": ts},
    )
    assert resp.status_code == 200
    assert resp.json()["challenge"] == "abc"


def test_webhook_rejects_stale_timestamp(client, monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "shhh")
    old_ts = str(int(time.time()) - 60 * 10)  # 10 minutes old
    body = b'{"type":"url_verification","challenge":"abc"}'
    sig = _sign("shhh", old_ts, body)
    resp = client.post(
        "/integrations/slack/webhook",
        content=body,
        headers={"X-Slack-Signature": sig, "X-Slack-Request-Timestamp": old_ts},
    )
    assert resp.status_code == 403
