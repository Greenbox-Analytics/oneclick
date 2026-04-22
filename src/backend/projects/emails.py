import html
import os

import resend


def send_project_invite_email(
    recipient_email: str,
    project_name: str,
    inviter_name: str,
    role: str,
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping project invite email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    safe_project = html.escape(project_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">You've been invited to a project!</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has invited you as a <strong>{safe_role}</strong>
        on the project <strong>&ldquo;{safe_project}&rdquo;</strong>.
      </p>
      <div style="text-align: center; margin: 32px 0;">
        <a href="{frontend_url}/auth"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Sign Up to Get Started
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        Once you create your account, you'll automatically be added to the project.
      </p>
    </div>
    """

    response = resend.Emails.send(
        {
            "from": from_address,
            "to": [recipient_email],
            "subject": f'{safe_inviter} invited you to "{safe_project}" on Msanii',
            "html": html_body,
        }
    )
    return response
