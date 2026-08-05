"""Resend org-invite email. Mirrors teams/emails.py with org wording."""

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


def send_org_invite_email(
    recipient_email: str, org_name: str, inviter_name: str, role: str, existing_user: bool = False
):
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    safe_org = html.escape(org_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    # NOTE: /notifications is a ProtectedRoute — a logged-OUT recipient bounces to /auth
    # (and only returns here if redirect-back is wired). Low impact: the invite is ALSO in-app
    # via the org's pending-invites list, so once they log in they can find it there regardless.
    if existing_user:
        cta_href = f"{frontend_url}/notifications"
        cta_label = "Open Msanii to accept"
        footer = "This invite is waiting in your Msanii notifications — Accept or Decline it there."
    else:
        cta_href = f"{frontend_url}/auth"
        cta_label = "Sign Up to Join"
        footer = "Once you create your account with this email, you'll be added to the organization automatically."

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">You've been invited to an organization!</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has invited you as a <strong>{safe_role}</strong>
        on the organization <strong>&ldquo;{safe_org}&rdquo;</strong>.
      </p>
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          {cta_label}
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">{footer}</p>
    </div>
    """

    return _send(
        subject=f'{safe_inviter} invited you to "{safe_org}" on Msanii',
        html_body=html_body,
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
    monthly credit limit. Clones send_org_invite_email's
    shape/env guards; `recipient_emails` is pre-resolved by the caller
    (orgs/router.py's background task, via the auth admin API — org_members
    only carries user_id, not email)."""
    if not recipient_emails:
        print("Warning: no admin recipients resolved — skipping credit request email")
        return None

    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    safe_org = html.escape(org_name)
    safe_requester = html.escape(requester_name)
    amount_label = (
        f"a limit of {requested_cap:,} credits / month" if requested_cap else "a higher limit (amount up to you)"
    )
    safe_amount = html.escape(amount_label)
    note_html = (
        f'<p style="font-size: 14px; color: #555; font-style: italic;">&ldquo;{html.escape(note)}&rdquo;</p>'
        if note
        else ""
    )
    cta_href = f"{frontend_url}/organization"

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">A member has asked for a higher credit limit.</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_requester}</strong> has requested <strong>{safe_amount}</strong>
        on the organization <strong>&ldquo;{safe_org}&rdquo;</strong>. Raising a limit
        doesn&rsquo;t move any credits — they simply draw more from the shared pool.
      </p>
      {note_html}
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Review request
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        Approve or deny this request from your organization's admin console.
      </p>
    </div>
    """

    return _send(
        subject=f'{safe_requester} requested a higher credit limit on "{safe_org}"',
        html_body=html_body,
        recipients=recipient_emails,
    )


def send_billing_reverted_email(recipient_email: str, org_name: str):
    """Tell a member their org seat no longer covers them (suspended, removed,
    or the org was archived) — their Msanii usage now bills to their personal
    plan. Clones send_org_invite_email's shape/env guards."""
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    safe_org = html.escape(org_name)
    cta_href = f"{frontend_url}/profile"

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">Your billing has moved to your personal plan.</p>
      <p style="font-size: 15px; color: #555;">
        Your seat on <strong>&ldquo;{safe_org}&rdquo;</strong> is no longer active, so your
        Msanii usage now bills to your <strong>personal plan</strong>. Your artists, projects,
        and files stay exactly as they are.
      </p>
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Review your plan
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        If this is unexpected, contact your organization's admin.
      </p>
    </div>
    """

    return _send(
        subject=f'Your Msanii usage now bills to your personal plan ("{safe_org}")',
        html_body=html_body,
        recipients=[recipient_email],
    )


def send_credit_request_resolved_email(
    recipient_email: str,
    org_name: str,
    approved: bool,
    new_cap: int | None = None,
    note: str | None = None,
):
    """Tell the requesting member how their cap-raise request was resolved.
    Clones send_credit_request_email's shape/env guards; `note` is the admin's
    optional deny note."""
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    safe_org = html.escape(org_name)
    cta_href = f"{frontend_url}/organization"

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

    note_html = (
        f'<p style="font-size: 14px; color: #555; font-style: italic;">&ldquo;{html.escape(note)}&rdquo;</p>'
        if note
        else ""
    )

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">{headline}</p>
      <p style="font-size: 15px; color: #555;">{detail}</p>
      {note_html}
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Open your organization
        </a>
      </div>
    </div>
    """

    return _send(subject=subject, html_body=html_body, recipients=[recipient_email])
