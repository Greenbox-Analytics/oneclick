"""Shared AWS SES client for transactional emails.

Two helpers:
- send_html(): plain HTML emails via SES SendEmail.
- send_with_attachments(): HTML + binary attachments via SES SendRawEmail.

Auth: boto3 reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env automatically.
Region: AWS_REGION (defaults to us-east-1).
From address: EMAIL_FROM (defaults to a placeholder; must be a verified SES identity).
"""

import logging
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

DEFAULT_FROM = "Msanii <noreply@msanii.com>"


@lru_cache(maxsize=1)
def _client():
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("ses", region_name=region)


def _from_address(override: str | None = None) -> str:
    return override or os.getenv("EMAIL_FROM", DEFAULT_FROM)


def _credentials_present() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def send_html(
    to: str | list[str],
    subject: str,
    html_body: str,
    from_address: str | None = None,
) -> dict | None:
    """Send an HTML email. Returns None (and logs a warning) if AWS creds are missing."""
    if not _credentials_present():
        logger.warning("AWS SES credentials not set — skipping email to %s", to)
        return None

    recipients = [to] if isinstance(to, str) else list(to)
    try:
        return _client().send_email(
            Source=_from_address(from_address),
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
            },
        )
    except (BotoCoreError, ClientError) as e:
        logger.warning("SES send_email failed: %s", e)
        return None


def send_with_attachments(
    to: str,
    subject: str,
    html_body: str,
    attachments: list[dict],
    from_address: str | None = None,
) -> dict:
    """Send an HTML email with binary attachments via SES SendRawEmail.

    attachments: list of {"filename": str, "content": bytes}.
    Raises RuntimeError if creds missing; lets boto3 errors propagate so callers
    can map them to HTTP responses (existing share_via_email returns 502).
    """
    if not _credentials_present():
        raise RuntimeError("AWS SES credentials not configured")

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = _from_address(from_address)
    msg["To"] = to

    body = MIMEMultipart("alternative")
    body.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(body)

    for att in attachments:
        part = MIMEApplication(att["content"])
        part.add_header("Content-Disposition", "attachment", filename=att["filename"])
        msg.attach(part)

    return _client().send_raw_email(
        Source=_from_address(from_address),
        Destinations=[to],
        RawMessage={"Data": msg.as_string()},
    )
