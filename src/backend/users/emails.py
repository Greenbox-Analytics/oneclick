"""Welcome email sent immediately after first Google sign-up."""

import html
import os

import resend


def send_welcome_email(recipient_email: str, first_name: str | None) -> dict | None:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping welcome email")
        return None

    from_address = os.getenv("RESEND_FROM_EMAIL")
    if not from_address:
        print("Warning: RESEND_FROM_EMAIL not set — skipping welcome email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")

    greeting_name = html.escape(first_name) if first_name else "there"

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">Hi {greeting_name},</p>
      <p style="font-size: 15px; color: #555; line-height: 1.5;">
        Welcome to Msanii &mdash; we're glad to have you. Sign in to start managing your
        music projects, rights, and collaborations.
      </p>
      <div style="text-align: center; margin: 32px 0;">
        <a href="{frontend_url}/dashboard"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Go to Dashboard
        </a>
      </div>
      <p style="font-size: 12px; color: #999; text-align: center;">Sent from Msanii.</p>
    </div>
    """

    try:
        return resend.Emails.send(
            {
                "from": from_address,
                "to": [recipient_email],
                "subject": "Welcome to Msanii",
                "html": html_body,
            }
        )
    except Exception as e:
        print(f"Warning: Failed to send welcome email: {e}")
        return None
