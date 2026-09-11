"""Invite email deliverability guards (orgs/emails.py, projects/emails.py).

Cold-recipient invites landed in spam when they were HTML-only, had no reply
path, and carried a user-controlled inviter name in a quoted subject. These
tests pin the payload shape that fixed it: a plain-text part, a reply-to,
List-Unsubscribe headers, a neutral subject, and no raw URL dump under the
button. They patch `resend.Emails.send` and never touch the network.
"""

import resend

TOKEN = "tok-123"
FRONTEND = "https://app.msanii.test"


def _capture_send(monkeypatch):
    sent = {}

    def fake_send(payload):
        sent.update(payload)
        return {"id": "email-1"}

    monkeypatch.setattr(resend.Emails, "send", staticmethod(fake_send))
    return sent


def _configure(monkeypatch, reply_to=None):
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Msanii <noreply@msanii.test>")
    monkeypatch.setenv("VITE_FRONTEND_URL", FRONTEND)
    if reply_to is None:
        monkeypatch.delenv("RESEND_REPLY_TO", raising=False)
    else:
        monkeypatch.setenv("RESEND_REPLY_TO", reply_to)


def _send_org_invite(existing_user=False, inviter="Alice", recipient="new@acme.com"):
    from orgs import emails

    return emails.send_org_invite_email(
        recipient_email=recipient,
        org_name="Acme",
        inviter_name=inviter,
        role="member",
        token=TOKEN,
        existing_user=existing_user,
    )


def _send_project_invite(inviter="Alice"):
    from projects import emails

    return emails.send_project_invite_email(
        recipient_email="new@acme.com",
        project_name="Debut EP",
        inviter_name=inviter,
        role="editor",
    )


# ---------------------------------------------------------------------------
# Org invite
# ---------------------------------------------------------------------------


def test_org_invite_subject_names_the_team_not_the_inviter(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite()
    assert sent["subject"] == "You're invited to join Acme on Msanii"
    assert "Alice" not in sent["subject"]


def test_org_invite_carries_text_part_reply_to_and_unsubscribe(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite()
    assert f"{FRONTEND}/orgs/invite/{TOKEN}" in sent["text"]
    assert f"{FRONTEND}/orgs/invite/{TOKEN}" in sent["html"]
    assert sent["reply_to"] == "Msanii <noreply@msanii.test>"
    assert sent["headers"]["List-Unsubscribe"] == f"<{FRONTEND}/profile>"
    assert sent["headers"]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"


def test_org_invite_reply_to_prefers_the_monitored_inbox(monkeypatch):
    _configure(monkeypatch, reply_to="hello@msanii.test")
    sent = _capture_send(monkeypatch)
    _send_org_invite()
    assert sent["reply_to"] == "hello@msanii.test"


def test_org_invite_link_carries_urlencoded_email_and_signup_hint(monkeypatch):
    """The claim link is the only thing that carries the invitee's address to
    the client (the public preview deliberately withholds it), so /auth can
    prefill it. A new invitee also gets a `signup=1` hint so /auth opens on
    the Sign Up tab. The HTML attribute must escape the `&`."""
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite(existing_user=False)
    assert f"{FRONTEND}/orgs/invite/{TOKEN}?email=new%40acme.com&signup=1" in sent["text"]
    assert f'href="{FRONTEND}/orgs/invite/{TOKEN}?email=new%40acme.com&amp;signup=1"' in sent["html"]
    assert '&signup=1"' not in sent["html"]


def test_org_invite_link_omits_signup_hint_for_existing_users(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite(existing_user=True)
    assert "email=new%40acme.com" in sent["text"]
    assert "signup=" not in sent["text"]
    assert "signup=" not in sent["html"]


def test_org_invite_link_encodes_plus_in_address(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite(recipient="new+team@acme.com")
    assert "email=new%2Bteam%40acme.com" in sent["text"]
    assert "email=new%2Bteam%40acme.com" in sent["html"]


def test_org_invite_has_no_raw_link_dump(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    for existing_user in (True, False):
        sent.clear()
        _send_org_invite(existing_user=existing_user)
        assert "copy this link" not in sent["html"].lower()


def test_org_invite_escapes_inviter_in_html_but_not_text(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_org_invite(inviter="<b>Mallory</b>")
    assert "<b>Mallory</b>" not in sent["html"]
    assert "&lt;b&gt;Mallory&lt;/b&gt;" in sent["html"]
    # The plain-text part is not HTML; it must not carry entities.
    assert "&lt;" not in sent["text"]


def test_org_invite_skips_send_when_resend_unconfigured(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Msanii <noreply@msanii.test>")
    sent = _capture_send(monkeypatch)
    assert _send_org_invite() is None
    assert sent == {}


# ---------------------------------------------------------------------------
# Project invite
# ---------------------------------------------------------------------------


def test_project_invite_subject_names_the_project_not_the_inviter(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_project_invite()
    assert sent["subject"] == "You're invited to join Debut EP on Msanii"
    assert "Alice" not in sent["subject"]


def test_project_invite_carries_text_part_reply_to_and_unsubscribe(monkeypatch):
    _configure(monkeypatch)
    sent = _capture_send(monkeypatch)
    _send_project_invite()
    assert f"{FRONTEND}/auth" in sent["text"]
    assert f"{FRONTEND}/auth" in sent["html"]
    assert sent["reply_to"] == "Msanii <noreply@msanii.test>"
    assert sent["headers"]["List-Unsubscribe"] == f"<{FRONTEND}/profile>"
    assert "copy this link" not in sent["html"].lower()


def test_project_invite_reply_to_prefers_the_monitored_inbox(monkeypatch):
    _configure(monkeypatch, reply_to="hello@msanii.test")
    sent = _capture_send(monkeypatch)
    _send_project_invite()
    assert sent["reply_to"] == "hello@msanii.test"


def test_project_invite_skips_send_when_resend_unconfigured(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.delenv("RESEND_FROM_EMAIL", raising=False)
    sent = _capture_send(monkeypatch)
    assert _send_project_invite() is None
    assert sent == {}
