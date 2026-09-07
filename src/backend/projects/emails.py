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

    from_address = os.getenv("RESEND_FROM_EMAIL")
    if not from_address:
        print("Warning: RESEND_FROM_EMAIL not set — skipping project invite email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")

    safe_project = html.escape(project_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    cta_href = f"{frontend_url}/auth"
    cta_label = "Join the project"

    # Fuller copy on purpose: the recipient is cold, the inviter's name is
    # user-controlled, and a one-line "X invited you" with a single button reads
    # as phishing to spam filters. Inviter stays out of the subject; plain-text
    # part, reply-to and List-Unsubscribe headers go on the payload.
    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>
      <p style="font-size: 16px; color: #333;">Hi, you've been invited to a project on Msanii.</p>
      <p style="font-size: 15px; color: #555; line-height: 1.5;">
        {safe_inviter} has invited you to join the project <strong>{safe_project}</strong>
        as a <strong>{safe_role}</strong>.
      </p>
      <p style="font-size: 15px; color: #555; line-height: 1.5;">
        Msanii is where artists, managers, and collaborators keep their music projects,
        ownership splits, contracts, and royalties in one place. Joining gives you access
        to the project's works and files.
      </p>
      <div style="text-align: center; margin: 32px 0;">
        <a href="{cta_href}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          {cta_label}
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        Create your Msanii account with this email address and you'll be added to the project.
        If you weren't expecting this invitation, you can ignore this email.
      </p>
    </div>
    """

    text_body = (
        "Hi,\n\n"
        f"{inviter_name} has invited you to join the project {project_name} on Msanii as a {role}.\n\n"
        "Msanii is where artists, managers, and collaborators keep their music projects, "
        "ownership splits, contracts, and royalties in one place. Joining gives you access "
        "to the project's works and files.\n\n"
        f"{cta_label}: {cta_href}\n\n"
        "Create your Msanii account with this email address and you'll be added to the project. "
        "If you weren't expecting this invitation, you can ignore this email."
    )

    response = resend.Emails.send(
        {
            "from": from_address,
            "to": [recipient_email],
            "subject": f"You're invited to join {project_name} on Msanii",
            "html": html_body,
            "text": text_body,
            "reply_to": os.getenv("RESEND_REPLY_TO") or from_address,
            "headers": {
                "List-Unsubscribe": f"<{frontend_url}/profile>",
                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
            },
        }
    )
    return response
