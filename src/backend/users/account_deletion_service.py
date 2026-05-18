"""Account deletion orchestration. Order matters: last-admin guard, Stripe cancel,
storage cleanup, then auth.users delete (which cascades the rest via FK)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from analytics import capture as analytics_capture
from subscriptions.admin_auth import env_admin_emails, is_user_admin
from subscriptions.stripe_client import get_stripe

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger(__name__)


def list_user_storage_paths(supabase: Client, user_id: str) -> list[tuple[str, str]]:
    """Return [(bucket, path), ...] for every file owned by this user.

    Walks artists → projects → project_files + audio_files. Storage rows do
    not have FK cascades, so we must enumerate them explicitly before
    deleting the auth user.
    """
    artists_res = supabase.table("artists").select("id").eq("user_id", user_id).execute()
    artist_ids = [a["id"] for a in (artists_res.data or [])]
    if not artist_ids:
        return []

    projects_res = supabase.table("projects").select("id").in_("artist_id", artist_ids).execute()
    project_ids = [p["id"] for p in (projects_res.data or [])]
    if not project_ids:
        return []

    paths: list[tuple[str, str]] = []

    pf_res = supabase.table("project_files").select("file_path").in_("project_id", project_ids).execute()
    for row in pf_res.data or []:
        if row.get("file_path"):
            paths.append(("project-files", row["file_path"]))

    af_res = supabase.table("audio_files").select("file_path").in_("project_id", project_ids).execute()
    for row in af_res.data or []:
        if row.get("file_path"):
            paths.append(("audio-files", row["file_path"]))

    return paths


def cancel_user_stripe(supabase: Client, user_id: str) -> None:
    """Cancel subscription + delete customer. Idempotent.

    Reads stripe_subscription_id and stripe_customer_id from the subscriptions
    row. Cancels the subscription (if any) first, then deletes the customer
    (if any) — the latter wipes email/name/last4 from Stripe to satisfy
    right-to-erasure. "Already canceled / no such X" errors are swallowed;
    any other Stripe error re-raises so the orchestrator can abort before
    destroying local data.
    """
    res = (
        supabase.table("subscriptions")
        .select("stripe_subscription_id, stripe_customer_id")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    row = (res.data if res else None) or {}
    sub_id = row.get("stripe_subscription_id")
    customer_id = row.get("stripe_customer_id")
    if not sub_id and not customer_id:
        return

    stripe = get_stripe()
    InvalidRequestError = getattr(stripe, "InvalidRequestError", None) or stripe.error.InvalidRequestError

    if sub_id:
        try:
            stripe.Subscription.delete(sub_id)
        except InvalidRequestError as exc:
            logger.info("Stripe subscription %s already canceled or missing: %s", sub_id, exc)

    if customer_id:
        try:
            stripe.Customer.delete(customer_id)
        except InvalidRequestError as exc:
            logger.info("Stripe customer %s already deleted or missing: %s", customer_id, exc)


def would_be_last_admin(supabase: Client, user_id: str, user_email: str | None) -> bool:
    """True if deleting this user removes the last admin.

    Combines env-admin emails (ADMIN_EMAILS) and db-admins (profiles.is_admin = true),
    dedupes by email/id, and checks whether removing this user leaves >= 1 other admin.
    """
    if not is_user_admin(supabase, user_email, user_id):
        return False

    env_set = {e.lower() for e in env_admin_emails()}
    other_env_admins = env_set - {(user_email or "").lower()}

    db_res = supabase.table("profiles").select("id").eq("is_admin", True).execute()
    db_admin_ids = {r["id"] for r in (db_res.data or []) if r.get("id") != user_id}

    return len(other_env_admins) == 0 and len(db_admin_ids) == 0


class LastAdminError(Exception):
    """Raised when the user is the only remaining admin."""


_STORAGE_BATCH = 1000


def _chunk(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _emit(user_id: str, event: str, props: dict | None = None) -> None:
    """Best-effort analytics emit — log on failure, never raise."""
    try:
        analytics_capture(user_id, event, props or {})
    except Exception as exc:
        logger.warning("%s analytics emit failed: %s", event, exc)


def delete_user_account(supabase: Client, user_id: str, user_email: str | None) -> None:
    """Run the full deletion. Order matters — Stripe before storage before auth.

    Storage `.remove()` failure aborts before `auth.admin.delete_user` — we
    cannot silently leave storage objects orphaned (their DB rows would
    cascade away with the auth user, leaving objects with no record they
    ever existed). The caller can retry; `.remove()` is idempotent on
    already-deleted paths.
    """
    if would_be_last_admin(supabase, user_id, user_email):
        _emit(user_id, "account_delete_blocked", {"reason": "last_admin", "email": user_email})
        raise LastAdminError("Cannot delete the only admin. Promote another admin first.")

    _emit(user_id, "account_delete_started", {"email": user_email})

    cancel_user_stripe(supabase, user_id)

    paths_by_bucket: dict[str, list[str]] = {}
    for bucket, path in list_user_storage_paths(supabase, user_id):
        paths_by_bucket.setdefault(bucket, []).append(path)

    for bucket, paths in paths_by_bucket.items():
        client = supabase.storage.from_(bucket)
        for batch in _chunk(paths, _STORAGE_BATCH):
            client.remove(batch)

    supabase.auth.admin.delete_user(user_id)

    _emit(user_id, "account_deleted", {"email": user_email})
