"""Resend team-invite email. Mirrors projects/emails.py."""

import html
import os

import resend


def send_team_invite_email(recipient_email: str, team_name: str, inviter_name: str, role: str):
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
        <a href="{frontend_url}/auth"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Sign Up to Join
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        Once you create your account with this email, you'll be added to the team automatically.
        Existing users: check your in-app notifications to accept.
      </p>
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
