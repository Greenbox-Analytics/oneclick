"""Self-serve team standing: slots, dials, coverage (spec 2026-08-15 §2/§3).

One module so the predicate has ONE home. The dials/ranking reads (TeamDials,
team_dials_for_user, count_covered_orgs) are read-side; evaluate_standing —
the sweep's ONE call in — WRITES organization rows, inserts notifications,
and sends best-effort email. Service-role client — callers gate authz.
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class NoSlotError(Exception):
    """Caller's tier has no free team slot (router maps to 402)."""


@dataclass(frozen=True)
class TeamDials:
    max_teams: int = 0
    max_team_members: int = 0  # seats EXCLUDING the covering owner
    team_storage_bytes: int = 0  # per-OWNER pool across all owned teams


def grace_days() -> int:
    """Configurable grace window (owner decision; env like its siblings)."""
    return int(os.getenv("ORG_GRACE_DAYS", "14"))


def _first(res):
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else None


def team_dials_for_user(db, user_id: str) -> TeamDials:
    """The user's team dials. Msanii admins resolve as Pro (consistent with
    admin-implicit-Pro caps); everyone else reads their subscription tier's
    row. A MISSING row (no profile, no subscription, no tier_entitlements
    match) fails CLOSED (zeros) via the explicit branches below — a user with
    no subscription row must never mint slots. A read that RAISES is a
    different failure (an outage, not "this user has zero dials") and is
    deliberately left to propagate as a 500 rather than being swallowed into
    a misleading upsell 402."""
    prof = _first(db.table("profiles").select("is_admin").eq("id", user_id).execute())
    if prof and prof.get("is_admin") is True:
        tier = "pro"
    else:
        sub = _first(db.table("subscriptions").select("tier").eq("user_id", user_id).execute())
        tier = (sub or {}).get("tier")
    if not tier:
        return TeamDials()
    row = _first(
        db.table("tier_entitlements")
        .select("max_teams, max_team_members, team_storage_bytes")
        .eq("tier", tier)
        .execute()
    )
    if not row:
        return TeamDials()
    return TeamDials(
        max_teams=row.get("max_teams") or 0,
        max_team_members=row.get("max_team_members") or 0,
        team_storage_bytes=row.get("team_storage_bytes") or 0,
    )


def count_covered_orgs(db, user_id: str) -> int:
    """Non-archived, non-dissolved self-serve orgs this user covers — keyed
    on `covered_by`, so a RELEASED org (covered_by still set, covered_at
    cleared — see orgs.service.release_coverage) still holds the slot until
    someone actually reclaims or the sweep reassigns it. Archived teams do
    NOT hold a slot (spec §3) — their bytes still count (Task 12)."""
    res = (
        db.table("organizations")
        .select("id", count="exact")
        .eq("kind", "self_serve")
        .eq("covered_by", user_id)
        .is_("archived_at", "null")
        .is_("dissolved_at", "null")
        .execute()
    )
    return res.count if res.count is not None else len(res.data or [])


def require_free_slot(db, user_id: str) -> None:
    """Raises NoSlotError when the caller has no free team slot. TOCTOU note:
    this is count-then-insert, not a locked reservation — two concurrent
    calls can both read a free slot and both insert, landing one team over
    the limit. Accepted (not fixed here): the sweep's daily ranking
    (evaluate_standing) is the actual enforcement backstop, and a one-team
    overshoot self-heals there rather than needing a DB-level reservation."""
    dials = team_dials_for_user(db, user_id)
    if count_covered_orgs(db, user_id) >= dials.max_teams:
        raise NoSlotError(
            "You have no free team slot on your plan. Upgrade, or archive a team you're covering, to create another."
        )


def self_serve_enabled() -> bool:
    """Both flags (spec §0): the sweep early-returns without credits, so a
    licensing-only deploy must not create self-serve rows it can't govern."""
    from subscriptions.service import credits_enabled, licensing_enabled

    return credits_enabled() and licensing_enabled()


# ============================================================================
# Daily standing sweep (Task 6, spec §3): grace -> lapsed -> recovery for
# every live self-serve org. Read-side ranking lives here alongside the dials
# it depends on; the write path (evaluate_standing) is the sweep's ONE call in.
# ============================================================================

# v1 single-page cap on the org scan below, mirroring subscriptions/sweep.py's
# _capped norm (that module's scans use the same cap + log-and-press-on
# shape). Not imported from there to avoid coupling this module's one query
# to that one's helper signature — if the cap is ever hit we log an error and
# press on rather than silently dropping the tail.
ROW_CAP = 10000


def _holds_active_admin_seat(db, user_id: str, org_id: str) -> bool:
    """True if user_id holds an ACTIVE admin seat in org_id — coverage needs
    both the ranking slot AND the seat that earned it; a demoted/offboarded
    coverer keeps neither. A plain org_members read (not authz.is_org_admin's
    RPC) — this runs once per ranked org in a batch sweep, not per request."""
    res = (
        db.table("org_members")
        .select("id")
        .eq("org_id", org_id)
        .eq("user_id", user_id)
        .eq("role", "admin")
        .eq("status", "active")
        .execute()
    )
    return bool(res.data)


def _notify_standing(db, org: dict, kind: str) -> None:
    """In-app + best-effort email to every ACTIVE admin of `org` for a
    standing transition, plus analytics to the covering admin. Copies
    orgs.service._notify_invite_expired's insert shape (status_change /
    entity_type 'org', no action buttons — this is a report, not something to
    action). Genuinely never raises: the notification-insert / email-send /
    analytics-capture side effects are each wrapped individually so one
    failing doesn't swallow the others for the same transition, AND the whole
    body sits in one more outer try/except so even the copy[kind] lookup and
    the admin-list read can't escape. A failure here is a lost notification,
    not a retried one — the daily sweep re-derives the transition from DB
    state next run, but nothing re-sends THIS message specifically.
    """
    try:
        org_name = org.get("name") or "your organization"
        copy = {
            "grace_started": (
                "Your team is losing access soon",
                f"Your team {org_name} loses access in {grace_days()} days — reactivate by "
                "keeping a Basic/Pro subscription or claiming coverage.",
            ),
            "lapsed": (
                "Your team has been paused",
                f"Your team {org_name} is paused — nothing is deleted, but active work stops "
                "until it's covered again. If a credit top-up subscription is still running for "
                "this team, cancel it from billing settings to stop further charges (credits "
                "already purchased stay in the pool, spendable once you reactivate). Reactivate "
                "anytime by keeping a Basic/Pro subscription or claiming coverage.",
            ),
            "reactivated": (
                "Your team is active again",
                f"Your team {org_name} is active again.",
            ),
        }
        title, message = copy[kind]

        admins = (
            db.table("org_members")
            .select("id, user_id, email")
            .eq("org_id", org["id"])
            .eq("role", "admin")
            .eq("status", "active")
            .execute()
            .data
            or []
        )
        if admins:
            try:
                db.table("notifications").insert(
                    [
                        {
                            "user_id": a["user_id"],
                            "type": "status_change",
                            "title": title,
                            "message": message,
                            "entity_type": "org",
                            "entity_id": org["id"],
                            "metadata": {"org_id": org["id"], "reason": f"standing_{kind}"},
                        }
                        for a in admins
                    ]
                ).execute()
            except Exception:
                logger.exception("_notify_standing: notification insert failed org=%s kind=%s", org["id"], kind)

            try:
                from orgs.emails import send_standing_email
                from orgs.service import _member_email

                recipients = [e for e in (_member_email(db, a) for a in admins) if e]
                if recipients:
                    send_standing_email(recipients, org_name, kind)
            except Exception:
                logger.exception("_notify_standing: email send failed org=%s kind=%s", org["id"], kind)

        covering_admin = org.get("covered_by")
        if covering_admin:
            try:
                from analytics import capture as analytics_capture

                analytics_capture(covering_admin, f"team_{kind}", {"org_id": org["id"]})
            except Exception:
                logger.exception("_notify_standing: analytics capture failed org=%s kind=%s", org["id"], kind)
    except Exception:
        logger.exception("_notify_standing: unexpected failure org=%s kind=%s", org.get("id"), kind)


def evaluate_standing(db) -> dict:
    """Daily standing predicate (spec §3). Returns counters for the sweep.
    Never raises — each org isolated; a bad row logs and moves on."""
    from subscriptions.service import _parse_iso

    now = datetime.now(UTC)
    orgs = (
        db.table("organizations")
        .select("id, name, status, covered_by, covered_at, grace_started_at")
        .eq("kind", "self_serve")
        .is_("archived_at", "null")
        .is_("dissolved_at", "null")
        .in_("status", ["active", "lapsed"])  # the pending/suspended guard, in the QUERY
        .limit(ROW_CAP)
        .execute()
    ).data or []
    if len(orgs) >= ROW_CAP:
        logger.error("evaluate_standing: org scan hit the %d-row cap — some orgs skipped this run", ROW_CAP)
    if not orgs:
        return {"evaluated": 0, "lapsed": 0, "grace_started": 0}

    # Rank each coverer's orgs once: covered_at DESC, released (None) last.
    by_user: dict[str, list[dict]] = {}
    for o in orgs:
        if o.get("covered_by"):
            by_user.setdefault(o["covered_by"], []).append(o)
    covered_ids: set[str] = set()
    skip_ids: set[str] = set()
    for user_id, owned in by_user.items():
        try:
            dials = team_dials_for_user(db, user_id)
            ranked = sorted(
                (o for o in owned if o.get("covered_at")),
                key=lambda o: _parse_iso(o["covered_at"]),
                reverse=True,  # never raw-string sort
            )
            for o in ranked[: dials.max_teams]:
                if _holds_active_admin_seat(db, user_id, o["id"]):
                    covered_ids.add(o["id"])
        except Exception:
            # A bad ranking input (a corrupt covered_at, a dials read blowing
            # up) must not abort the whole scan. Deliberately NOT "treat as
            # uncovered" — that would grace-stamp/lapse this owner's orgs over
            # a DATA problem, not a real coverage lapse. Skip their orgs
            # entirely this run; the next sweep re-ranks them from scratch.
            logger.exception("standing ranking failed user=%s — skipping their orgs this run", user_id)
            skip_ids.update(o["id"] for o in owned)

    lapsed = 0
    grace_started = 0
    for o in orgs:
        if o["id"] in skip_ids:
            continue
        try:
            if o["id"] in covered_ids:
                if o.get("grace_started_at") or o["status"] == "lapsed":
                    patch = {"grace_started_at": None}
                    was_lapsed = o["status"] == "lapsed"
                    if was_lapsed:
                        from orgs.storage_guard import reactivation_allowed

                        if not reactivation_allowed(db, o["covered_by"]):
                            continue  # covered but parked over-pool: stays lapsed
                        patch["status"] = "active"
                    # Write THEN notify — never tell an admin "active again"
                    # ahead of the update that makes it true.
                    db.table("organizations").update(patch).eq("id", o["id"]).execute()
                    if was_lapsed:
                        _notify_standing(db, o, "reactivated")
                continue
            # Uncovered:
            if not o.get("grace_started_at"):
                db.table("organizations").update({"grace_started_at": now.isoformat()}).eq("id", o["id"]).execute()
                _notify_standing(db, o, "grace_started")
                grace_started += 1
            elif o["status"] != "lapsed" and (now - _parse_iso(o["grace_started_at"])).days >= grace_days():
                db.table("organizations").update({"status": "lapsed"}).eq("id", o["id"]).execute()
                _notify_standing(db, o, "lapsed")
                lapsed += 1
        except Exception:
            logger.exception("standing evaluation failed org=%s", o.get("id"))
    return {"evaluated": len(orgs), "lapsed": lapsed, "grace_started": grace_started}
