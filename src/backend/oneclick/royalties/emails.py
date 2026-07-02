"""Payout receipt email sent to payees via Resend after a PayPal payment.

Follows the registry/emails.py conventions: missing config or any send
failure logs a warning and returns None — a failed email must never fail the
payment capture that triggered it.
"""

import base64
import html
import os

import resend


def send_payout_receipt_email(
    recipient_email: str,
    payee_name: str,
    payer_name: str,
    amount_str: str,
    paid_at: str | None,
    paypal_capture_id: str | None,
    receipt_pdf_bytes: bytes,
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping payout receipt email")
        return None

    from_address = os.getenv("RESEND_FROM_EMAIL")
    if not from_address:
        print("Warning: RESEND_FROM_EMAIL not set — skipping payout receipt email")
        return None

    resend.api_key = api_key

    safe_payee = html.escape(payee_name or "there")
    safe_payer = html.escape(payer_name or "A Msanii user")
    safe_amount = html.escape(amount_str)
    paid_date = html.escape(str(paid_at)[:10]) if paid_at else "today"
    capture_line = (
        f'<p style="font-size: 13px; color: #888;">PayPal transaction: {html.escape(paypal_capture_id)}</p>'
        if paypal_capture_id
        else ""
    )

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii</h1>
      </div>

      <p style="font-size: 16px; color: #333;">Hi {safe_payee},</p>

      <p style="font-size: 15px; color: #555;">
        <strong>{safe_payer}</strong> has sent you a royalty payment of
        <strong>{safe_amount}</strong> via PayPal on {paid_date}.
      </p>

      <p style="font-size: 15px; color: #555;">
        Your receipt is attached to this email. The money will appear in the PayPal
        account linked to this email address.
      </p>

      {capture_line}

      <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />

      <p style="font-size: 12px; color: #aaa; text-align: center;">
        Sent via Msanii OneClick royalty tracking
      </p>
    </div>
    """

    try:
        response = resend.Emails.send(
            {
                "from": from_address,
                "to": [recipient_email],
                "subject": f"You've received a royalty payment of {amount_str}",
                "html": html_body,
                "attachments": [
                    {
                        "filename": "Msanii_Payment_Receipt.pdf",
                        "content": base64.b64encode(receipt_pdf_bytes).decode("ascii"),
                        "content_type": "application/pdf",
                    }
                ],
            }
        )
        return response
    except Exception as e:
        print(f"Warning: Failed to send payout receipt email: {e}")
        return None
