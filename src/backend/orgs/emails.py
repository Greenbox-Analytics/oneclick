"""Resend org emails. Mirrors teams/emails.py with org wording. Every sender
is copy + `_layout(...)` + `_send(...)` — the shared HTML shell (Msanii
header, CTA button, optional footer) lives in `_layout` only."""

import html
import os

import resend


def _send(subject: str, html_body: str, recipients: list[str]):
    """Shared Resend dispatch + env guard. Returns None (no send) when
    RESEND_API_KEY / RESEND_FROM_EMAIL are unset."""
    api_key = os.getenv("RESEND_API_KEY")
    from_address = os.getenv("RESEND_FROM_EMAIL")
    if not api_key or not from_address:
        print("Warning: RESEND not configured — skipping email")
        return None

    resend.api_key = api_key
    return resend.Emails.send(
        {
            "from": from_address,
            "to": recipients,
            "subject": subject,
            "html": html_body,
        }
    )


def _frontend_url() -> str:
    return os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")


def _layout(headline: str, detail_html: str, cta_href: str, cta_label: str, footer: str = "") -> str:
    """The one HTML shell every org email shares. `detail_html` is
    pre-escaped/pre-built HTML (paragraphs, optional note); `footer` is
    plain text, wrapped here when present."""
    footer_html = f'<p style="font-size: 13px; color: #999; text-align: center;">{footer}</p>' if footer else ""
    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">{headline}</p>
      {detail_html}
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          {cta_label}
        </a>
      </div>
      {footer_html}
    </div>
    """


def _note_html(note: str | None) -> str:
    return (
        f'<p style="font-size: 14px; color: #555; font-style: italic;">&ldquo;{html.escape(note)}&rdquo;</p>'
        if note
        else ""
    )


def send_org_invite_email(
    recipient_email: str, org_name: str, inviter_name: str, role: str, token: str, existing_user: bool = False
):
    safe_org = html.escape(org_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    # Both branches link to the token'd claim page (OrgInviteClaim.tsx), which is
    # a PUBLIC route that self-gates: signed out it offers "Sign in / Create
    # account" with ?redirect back to itself, signed in it Accepts/Declines and
    # owns the expired / wrong-email / not-found copy.
    #
    # It previously pointed at /notifications (existing user) and /auth (new
    # user). Both dead-ended: nothing writes an in-app row at invite time, the
    # org's pending-invites list is admin-only so the INVITEE can never see it,
    # and there is no signup trigger converting a pending_org_invites row (the
    # board-teams module that had one was removed 2026-08-16), so "you'll be added
    # automatically" was never true. The token is the only thing that carries
    # the invite; the link has to hold it.
    cta_href = f"{_frontend_url()}/orgs/invite/{token}"
    if existing_user:
        cta_label = "Review invitation"
        footer = "Accept or Decline this invite from the link above."
    else:
        cta_label = "Sign Up to Join"
        footer = "Create your account with this email, then accept the invitation to join."

    detail = f"""<p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has invited you as a <strong>{safe_role}</strong>
        on the organization <strong>&ldquo;{safe_org}&rdquo;</strong>.
      </p>"""

    return _send(
        subject=f'{safe_inviter} invited you to "{safe_org}" on Msanii',
        html_body=_layout("You've been invited to an organization!", detail, cta_href, cta_label, footer),
        recipients=[recipient_email],
    )


def send_credit_request_email(
    recipient_emails: list[str],
    org_name: str,
    requester_name: str,
    requested_cap: int | None,
    note: str | None = None,
):
    """Notify every ACTIVE org admin that a member has asked for a higher
    monthly credit limit. `recipient_emails` is pre-resolved by the caller
    (orgs/router.py's background task, via the auth admin API — org_members
    only carries user_id, not email)."""
    if not recipient_emails:
        print("Warning: no admin recipients resolved — skipping credit request email")
        return None

    safe_org = html.escape(org_name)
    safe_requester = html.escape(requester_name)
    amount_label = (
        f"a limit of {requested_cap:,} credits / month" if requested_cap else "a higher limit (amount up to you)"
    )
    safe_amount = html.escape(amount_label)

    detail = f"""<p style="font-size: 15px; color: #555;">
        <strong>{safe_requester}</strong> has requested <strong>{safe_amount}</strong>
        on the organization <strong>&ldquo;{safe_org}&rdquo;</strong>. Raising a limit
        doesn&rsquo;t move any credits — they simply draw more from the shared pool.
      </p>
      {_note_html(note)}"""

    return _send(
        subject=f'{safe_requester} requested a higher credit limit on "{safe_org}"',
        html_body=_layout(
            "A member has asked for a higher credit limit.",
            detail,
            f"{_frontend_url()}/teams",
            "Review request",
            "Approve or deny this request from your organization's admin console.",
        ),
        recipients=recipient_emails,
    )


def send_billing_reverted_email(recipient_email: str, org_name: str):
    """Tell a member their org seat no longer covers them (suspended, removed,
    or the org was archived) — their Msanii usage now bills to their personal
    plan."""
    safe_org = html.escape(org_name)

    detail = f"""<p style="font-size: 15px; color: #555;">
        Your seat on <strong>&ldquo;{safe_org}&rdquo;</strong> is no longer active, so your
        Msanii usage now bills to your <strong>personal plan</strong>. Your artists, projects,
        and files stay exactly as they are.
      </p>"""

    return _send(
        subject=f'Your Msanii usage now bills to your personal plan ("{safe_org}")',
        html_body=_layout(
            "Your billing has moved to your personal plan.",
            detail,
            f"{_frontend_url()}/profile",
            "Review your plan",
            "If this is unexpected, contact your organization's admin.",
        ),
        recipients=[recipient_email],
    )


def send_standing_email(recipient_emails: list[str], org_name: str, headline: str, detail: str):
    """Best-effort email to every ACTIVE admin on a team-standing transition
    (grace/lapsed/reactivated — orgs/standing.py's evaluate_standing, the
    daily sweep). The caller (standing._notify_standing) builds the
    headline/detail copy once and shares it with the in-app notification;
    `detail` is plain text and escaped here."""
    if not recipient_emails:
        print("Warning: no admin recipients resolved — skipping standing email")
        return None

    detail_html = f'<p style="font-size: 15px; color: #555;">{html.escape(detail)}</p>'

    return _send(
        subject=f'{headline} ("{html.escape(org_name)}")',
        html_body=_layout(headline, detail_html, f"{_frontend_url()}/teams", "Open your organization"),
        recipients=recipient_emails,
    )


def send_credit_request_resolved_email(
    recipient_email: str,
    org_name: str,
    approved: bool,
    new_cap: int | None = None,
    note: str | None = None,
):
    """Tell the requesting member how their cap-raise request was resolved.
    `note` is the admin's optional deny note."""
    safe_org = html.escape(org_name)

    if approved:
        headline = "Your credit limit request was approved."
        detail = (
            f"Your monthly limit on <strong>&ldquo;{safe_org}&rdquo;</strong> is now "
            f"<strong>{new_cap:,} credits / month</strong>."
            if new_cap is not None
            else f"Your monthly limit on <strong>&ldquo;{safe_org}&rdquo;</strong> has been raised."
        )
        subject = f'Your credit limit on "{safe_org}" was raised'
    else:
        headline = "Your credit limit request was declined."
        detail = (
            f"An admin of <strong>&ldquo;{safe_org}&rdquo;</strong> declined your request "
            "for a higher monthly credit limit."
        )
        subject = f'Your credit limit request on "{safe_org}" was declined'

    detail_html = f"""<p style="font-size: 15px; color: #555;">{detail}</p>
      {_note_html(note)}"""

    return _send(
        subject=subject,
        html_body=_layout(headline, detail_html, f"{_frontend_url()}/teams", "Open your organization"),
        recipients=[recipient_email],
    )
