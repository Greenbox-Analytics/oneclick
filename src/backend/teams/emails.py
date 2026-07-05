"""Resend team-invite email. Mirrors projects/emails.py."""

import html
import os

import resend


def send_team_invite_email(
    recipient_email: str, team_name: str, inviter_name: str, role: str, existing_user: bool = False
):
    api_key = os.getenv("RESEND_API_KEY")
    from_address = os.getenv("RESEND_FROM_EMAIL")
    if not api_key or not from_address:
        print("Warning: RESEND not configured — skipping team invite email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    safe_team = html.escape(team_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    # NOTE: /notifications is a ProtectedRoute — a logged-OUT recipient bounces to /auth
    # (and only returns here if redirect-back is wired). Low impact: the invite is ALSO in-app,
    # so once they log in the bell badge shows it regardless. Keeping /notifications is fine for
    # the common case (existing users are usually already logged in). Switch to /dashboard only
    # if you'd rather never risk the /auth bounce.
    if existing_user:
        cta_href = f"{frontend_url}/notifications"
        cta_label = "Open Msanii to accept"
        footer = "This invite is waiting in your Msanii notifications — Accept or Decline it there."
    else:
        cta_href = f"{frontend_url}/auth"
        cta_label = "Sign Up to Join"
        footer = "Once you create your account with this email, you'll be added to the team automatically."

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">You've been invited to a team!</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has invited you as a <strong>{safe_role}</strong>
        on the team <strong>&ldquo;{safe_team}&rdquo;</strong>.
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

    return resend.Emails.send(
        {
            "from": from_address,
            "to": [recipient_email],
            "subject": f'{safe_inviter} invited you to "{safe_team}" on Msanii',
            "html": html_body,
        }
    )
