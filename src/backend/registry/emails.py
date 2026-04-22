"""Invitation and verification emails for the Rights Registry."""

import html
import os

import resend


def send_invitation_email(
    recipient_email: str,
    recipient_name: str,
    inviter_name: str,
    work_title: str,
    role: str,
    invite_token: str,
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping invitation email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    claim_url = f"{frontend_url}/tools/registry/invite/{invite_token}"
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    # Escape all user-supplied values to prevent HTML injection
    safe_name = html.escape(recipient_name)
    safe_inviter = html.escape(inviter_name)
    safe_title = html.escape(work_title)
    safe_role = html.escape(role)

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii Rights Registry</h1>
      </div>

      <p style="font-size: 16px; color: #333;">Hi {safe_name},</p>

      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has listed you as a <strong>{safe_role}</strong> on the work
        <strong>&ldquo;{safe_title}&rdquo;</strong> and is requesting you confirm your ownership stake.
      </p>

      <div style="text-align: center; margin: 32px 0;">
        <a href="{claim_url}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Review &amp; Confirm Your Stake
        </a>
      </div>

      <p style="font-size: 13px; color: #888;">
        You'll be asked to sign in (or create an account) to view the full ownership details
        and confirm or dispute your stake. This invitation expires in 48 hours.
      </p>

      <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />

      <p style="font-size: 12px; color: #aaa; text-align: center;">
        Sent via Msanii Rights &amp; Ownership Registry
      </p>
    </div>
    """

    try:
        response = resend.Emails.send(
            {
                "from": from_address,
                "to": [recipient_email],
                "subject": f'{safe_inviter} needs you to confirm your stake on "{safe_title}"',
                "html": html_body,
            }
        )
        return response
    except Exception as e:
        print(f"Warning: Failed to send invitation email: {e}")
        return None


def send_rich_invitation_email(
    recipient_email: str,
    recipient_name: str,
    inviter_name: str,
    work_title: str,
    project_name: str,
    artist_name: str,
    role: str,
    stakes: list,
    notes: str = None,
    invite_token: str = "",
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping invitation email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    claim_url = f"{frontend_url}/tools/registry/invite/{invite_token}"
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    safe_name = html.escape(recipient_name)
    safe_inviter = html.escape(inviter_name)
    safe_title = html.escape(work_title)
    safe_project = html.escape(project_name)
    safe_artist = html.escape(artist_name)
    safe_role = html.escape(role)

    stakes_html = ""
    for s in stakes or []:
        stake_type = html.escape(s.stake_type if hasattr(s, "stake_type") else s.get("stake_type", ""))
        pct = s.percentage if hasattr(s, "percentage") else s.get("percentage", 0)
        stakes_html += f'<div style="margin: 4px 0;"><strong>{stake_type.title()}:</strong> {pct}%</div>'

    notes_html = ""
    if notes:
        safe_notes = html.escape(notes)
        notes_html = f'<div style="margin-top: 12px; padding: 10px; background: #f5f5f5; border-radius: 6px; font-size: 14px; color: #555;"><strong>Notes:</strong> {safe_notes}</div>'

    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii Rights Registry</h1>
      </div>
      <p style="font-size: 16px; color: #333;">Hi {safe_name},</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has listed you as a <strong>{safe_role}</strong> on the work
        <strong>&ldquo;{safe_title}&rdquo;</strong>.
      </p>
      <div style="background: #f9f9f9; border-radius: 8px; padding: 16px; margin: 16px 0;">
        <div style="font-size: 13px; color: #888; margin-bottom: 8px;">Details:</div>
        <div style="font-size: 14px; color: #333;">
          <div><strong>Artist:</strong> {safe_artist}</div>
          <div><strong>Project:</strong> {safe_project}</div>
          <div><strong>Role:</strong> {safe_role}</div>
          {stakes_html}
        </div>
      </div>
      {notes_html}
      <div style="text-align: center; margin: 32px 0;">
        <a href="{claim_url}" style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                   border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Review &amp; Respond
        </a>
      </div>
      <p style="font-size: 13px; color: #999; text-align: center;">
        This invitation expires in 48 hours. You can also respond from your
        <a href="{frontend_url}/tools/registry" style="color: #1a3a2a;">Registry Dashboard</a>.
      </p>
    </div>
    """

    try:
        response = resend.Emails.send(
            {
                "from": from_address,
                "to": [recipient_email],
                "subject": f'{safe_inviter} needs you to confirm your stake on "{safe_title}"',
                "html": html_body,
            }
        )
        return response
    except Exception as e:
        print(f"Warning: Failed to send invitation email: {e}")
        return None
