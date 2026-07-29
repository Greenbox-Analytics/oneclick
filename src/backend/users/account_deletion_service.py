"""Account deletion orchestration. Order matters: last-admin guard, Stripe cancel,
storage cleanup, org seat reclaim/teardown, then auth.users delete (which
cascades the rest via FK)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from analytics import capture as analytics_capture
from orgs.service import _now_iso
from subscriptions.admin_auth import env_admin_emails, is_user_admin
from subscriptions.stripe_client import get_stripe

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger(__name__)


def list_user_storage_paths(supabase: Client, user_id: str) -> list[tuple[str, str]]:
    """Return [(bucket, path), ...] for every file owned by this user.

    project_files are project-scoped (project_files.project_id → projects).
    audio_files are artist-scoped via folders (audio_files.folder_id →
    audio_folders.artist_id → artists), so the audio walk is independent
    of whether the user has any projects. Storage rows do not have FK
    cascades, so we must enumerate them explicitly before deleting the
    auth user.
    """
    artists_res = supabase.table("artists").select("id").eq("user_id", user_id).execute()
    artist_ids = [a["id"] for a in (artists_res.data or [])]
    if not artist_ids:
        return []

    paths: list[tuple[str, str]] = []

    projects_res = supabase.table("projects").select("id").in_("artist_id", artist_ids).execute()
    project_ids = [p["id"] for p in (projects_res.data or [])]
    if project_ids:
        pf_res = supabase.table("project_files").select("file_path").in_("project_id", project_ids).execute()
        for row in pf_res.data or []:
            if row.get("file_path"):
                paths.append(("project-files", row["file_path"]))

    folders_res = supabase.table("audio_folders").select("id").in_("artist_id", artist_ids).execute()
    folder_ids = [f["id"] for f in (folders_res.data or [])]
    if folder_ids:
        af_res = supabase.table("audio_files").select("file_path").in_("folder_id", folder_ids).execute()
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


# ---------------------------------------------------------------------------
# Licensing Phase B (Task 10): org seat reclaim + sole-admin org teardown.
#
# Reused idioms from orgs.service._offboard (money-first ordering, the
# offboard:{member_id}:{epoch(revoked_at)} request_id grammar) but
# reimplemented locally rather than calling _offboard/remove_member/
# suspend_member directly: those are gated on authz.require_admin(acting_
# user, org_id), which doesn't apply here — account deletion acts on the
# deleting user's OWN seats (they need not be an org admin at all) and, in
# the sole-admin case, on every OTHER member's seat during a system-
# initiated teardown (the deleting user IS the admin, but this is not an
# in-app admin action). orgs/service.py and orgs/wallets.py are imported
# for their epoch/timestamp/wallet-lookup helpers but never modified.
# ---------------------------------------------------------------------------


def _remove_own_memberships(supabase: Client, user_id: str, own_rows: list[dict], archived_org_ids: set[str]) -> None:
    """Soft-remove every org_members row this user holds, EXCEPT orgs already
    handled by `_archive_sole_admin_orgs` (see the skip below), BEFORE the user
    row is deleted. Nothing is reclaimed — a member never held credits, only a
    monthly cap on the org pool. Best-effort per membership: a failure is logged
    and never blocks deletion; the
    status transition below is deliberately not rolled back either —
    money-first, same stance as `_offboard` (a retry re-derives the
    identical request_id and converges).

    `own_rows` and `archived_org_ids` are supplied by the caller
    (`delete_user_account`), which fetches org_members ONCE and runs
    `_archive_sole_admin_orgs` FIRST — see that call site's comment for why
    the ordering matters (Phase B review finding 1).
    """
    for row in own_rows:
        org_id = row["org_id"]
        member_id = row["id"]
        if org_id in archived_org_ids:
            # `_archive_sole_admin_orgs` already archived this org and
            # reclaimed EVERY member's seat balance — including this user's
            # own — WITHOUT writing to org_members at all (see that
            # function's docstring). Attempting the status flip below
            # ('active' -> 'removed') on the sole ACTIVE admin's own row
            # would trip `org_members_admin_guard`
            # (supabase/migrations/20260721000001_licensing_core.sql):
            # is_cascade is FALSE here (pg_trigger_depth() == 1 — this is a
            # direct UPDATE from the service-role client, not a FK cascade,
            # and the user row is still in auth.users mid-flow), and with no
            # other active admins the guard RAISEs instead of
            # auto-archiving. That RAISE would be swallowed by the broad
            # except below and logged as a scary ERROR-level traceback on
            # every sole-admin deletion, even though the org is already
            # correctly torn down. Skip outright — nothing left to do here.
            continue
        if row.get("status") == "removed" and row.get("revoked_at"):
            continue  # already removed by a prior attempt — nothing left to do
        try:
            supabase.table("org_members").update({"status": "removed", "revoked_at": _now_iso()}).eq(
                "id", member_id
            ).execute()
        except Exception:
            logger.exception("account deletion: membership removal failed member=%s org=%s", member_id, org_id)


def _teardown_archived_org_grants(supabase: Client, org_id: str) -> None:
    """Licensing Phase C, Task 4 (rule 12): mirrors
    `orgs.service._teardown_archived_org_grants` exactly — `archived_at` is
    an UPDATE, so the `org_project_links.org_id` ON DELETE CASCADE never
    fires, and without this an archived org can strand a live
    `project_members` grant or a tombstone link that blocks re-linking under
    rule 8's `UNIQUE(project_id)`. Reimplemented locally rather than calling
    the `orgs.service` helper directly, for the same reason every other
    teardown helper in this module is local (see the module-section
    docstring above `_seat_wallet_balance`): this runs during a
    system-initiated account deletion, not an in-app admin action, and
    `orgs/service.py` is out of scope for this task's touch list.

    Reuses `orgs.projects.revoke_org_granted_memberships` (Task 2's single
    implementation of rule 3) rather than re-deriving the delete filter —
    imported lazily to avoid a needless module-level cross-package import at
    process start. Never raises — logs and returns, matching this module's
    other best-effort teardown helper (`_reclaim_seat_to_pool`); a cleanup
    failure here must never block account deletion.
    """
    try:
        from orgs.projects import revoke_org_granted_memberships

        revoke_org_granted_memberships(supabase, org_id)
    except Exception:
        logger.exception("account deletion: org-granted membership revocation failed org=%s", org_id)
    try:
        supabase.table("org_project_links").delete().eq("org_id", org_id).execute()
    except Exception:
        logger.exception("account deletion: org_project_links cleanup failed org=%s", org_id)


def _archive_sole_admin_orgs(supabase: Client, user_id: str, own_rows: list[dict]) -> set[str]:
    """For every org where this user is the LAST ACTIVE admin, archive the
    org and reclaim EVERY member's seat balance to the pool — INCLUDING the
    deleting admin's own — BEFORE the user is deleted (review round 4 /
    spec §4 lifecycle; Phase B review finding 1). The last-admin guard
    trigger's own cascade-archive branch is only a backstop for non-service
    deletion paths — relying on it alone would strand every other member's
    credits in seat wallets of an org whose admin endpoints just died with
    it.

    Deliberately runs BEFORE `_remove_own_memberships` (see the call site in
    `delete_user_account`) and deliberately NEVER writes to org_members: the
    archived org confers nothing once `archived_at` is set, so there is no
    need to soft-remove any membership row — and critically, a direct
    status flip on the sole active admin's OWN row here would trip
    `org_members_admin_guard` and RAISE (see the skip comment in
    `_remove_own_memberships`). Other members' org_members rows are likewise left
    untouched (status, revoked_at unchanged) — archived_at alone already
    zeroes their entitlement resolution, per spec.

    Also tears down (Task 4, rule 12) every `project_members` grant and
    `org_project_links` row this org holds, via `_teardown_archived_org_grants`
    — same best-effort, never-blocks-deletion posture as the seat reclaim
    above, and run for BOTH a fresh archive and a retry-detected
    already-archived org (a prior attempt may have archived the org but
    crashed before this cleanup ran; re-running it is idempotent).

    Returns the set of org_ids archived (whether archived just now or on a
    prior retry) so `_remove_own_memberships` knows which orgs to skip.
    """
    admin_org_ids = {r["org_id"] for r in own_rows if r.get("role") == "admin" and r.get("status") == "active"}
    archived_org_ids: set[str] = set()
    for org_id in admin_org_ids:
        try:
            other_admins = (
                supabase.table("org_members")
                .select("id")
                .eq("org_id", org_id)
                .eq("role", "admin")
                .eq("status", "active")
                .neq("user_id", user_id)
                .execute()
                .data
                or []
            )
            if other_admins:
                continue  # not the sole admin — no teardown

            org_res = supabase.table("organizations").select("archived_at").eq("id", org_id).maybe_single().execute()
            if not ((org_res.data or {}).get("archived_at") if org_res else None):
                supabase.table("organizations").update({"archived_at": _now_iso()}).eq("id", org_id).execute()

            archived_org_ids.add(org_id)
            # No per-member reclaim: members hold no credits. Whatever the POOL
            # still holds stays there for support to dispose of (the admin
            # clawback endpoint), exactly as with an in-app archive.
            _teardown_archived_org_grants(supabase, org_id)
        except Exception:
            logger.exception("account deletion: sole-admin org teardown failed org=%s", org_id)
    return archived_org_ids


def delete_user_account(supabase: Client, user_id: str, user_email: str | None) -> None:
    """Run the full deletion. Order matters — Stripe before storage before org
    seat reclaim before auth.

    Storage `.remove()` failure aborts before `auth.admin.delete_user` — we
    cannot silently leave storage objects orphaned (their DB rows would
    cascade away with the auth user, leaving objects with no record they
    ever existed). The caller can retry; `.remove()` is idempotent on
    already-deleted paths.

    Org seat reclaim/teardown (licensing Phase B, Task 10) runs last, still
    BEFORE the auth user (and its org_members CASCADE) disappear, but is
    wrapped so that ANY failure here — not just a per-seat reclaim failure —
    is logged and never blocks deletion: a personal account deletion must
    never fail over an org role (privacy implications), and the org_members
    CASCADE is a safe fallback for whatever this step didn't reach.

    Within that step, `_archive_sole_admin_orgs` MUST run before
    `_remove_own_memberships` (Phase B review finding 1): archiving reclaims
    every member's seat balance — including this user's own — without ever
    writing to org_members, so those orgs become a pure skip for
    `_remove_own_memberships`. Reversed, `_remove_own_memberships` would flip this
    user's own row to 'removed' BEFORE the org is archived; for a sole
    ACTIVE admin that direct status flip trips `org_members_admin_guard`
    (supabase/migrations/20260721000001_licensing_core.sql) — is_cascade is
    false (trigger depth 1, user still in auth.users, no other active
    admins) — so the guard RAISEs, the broad except below swallows it, and
    the reclaim for that seat never runs.
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

    try:
        own_org_rows = supabase.table("org_members").select("*").eq("user_id", user_id).execute().data or []
        archived_org_ids = _archive_sole_admin_orgs(supabase, user_id, own_org_rows)
        _remove_own_memberships(supabase, user_id, own_org_rows, archived_org_ids)
    except Exception:
        logger.exception("account deletion: org seat reclaim/teardown failed user=%s", user_id)

    supabase.auth.admin.delete_user(user_id)

    _emit(user_id, "account_deleted", {"email": user_email})
