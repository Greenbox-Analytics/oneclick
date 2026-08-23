"""Artist visibility — the backend's single mirror of the SQL can_access_artist().

The backend runs on the service-role client, which BYPASSES RLS, so these
helpers ARE the authorization for every artist-scoped read in the app. They must
answer what can_access_artist() answers for RLS
(supabase/migrations/20260803000001_team_owned_artists.sql,
20260816000002_self_serve_orgs.sql):

    personal: team_id IS NULL AND user_id = me
    team:     team_id = an org where I hold an ACTIVE seat, org not archived
              and not 'lapsed'

`artists.user_id` is the CREATOR and keeps pointing at them after a transfer, so
`AND team_id IS NULL` on the personal branch is load-bearing in BOTH directions:
without it a team artist is invisible to the colleagues who should see it, and
still visible to an offboarded creator who should not. That is the same
re-scoping 20260803000002 applied to the 21 creator-keyed RLS policies — this
module is the service-role side of it.

Lives at top level rather than in main.py because registry/ and boards/ need it
too and cannot import main (cycle).

ponytail: mirrors can_access_artist in Python instead of calling the SQL
function per row, because the listing paths need a SET (one query, still
paginatable) rather than a per-artist predicate. ONE implementation, so there is
exactly one place to change if the SQL's definition of access changes.

Two function families live here, and the whole workspace-scoping design rests
on not merging them:

    visible_artists / can_access_artist  -> AUTHORIZATION, always the union
    scoped_artists  / scoped_artist_ids  -> PRESENTATION, the union narrowed

Scoping decides which workspace a LISTING shows (personal vs one org — see
resolve_scope). It is applied ON TOP of visible_artists, never instead of it,
so a forged or stale Scope can only ever return less. Narrowing the
authorization family instead would break transfer, invite-accept, deep links
and the org console.
"""

import os
from dataclasses import dataclass

from fastapi import HTTPException
from supabase import Client


def live_org_ids(db: Client, user_id: str) -> list[str]:
    """Orgs whose team-owned artists `user_id` may reach: an ACTIVE seat in a
    NON-ARCHIVED, non-`'lapsed'` org. All three are load-bearing — a
    suspended/removed seat confers nothing, can_access_artist denies on
    `archived_at` (which is exactly why archiving an org can leave
    `artists.team_id` attached), and 20260816000002_self_serve_orgs.sql adds
    the same denial for a self-serve org whose grace period ran out:
    `'lapsed'` goes inert for EVERY member, admins included.
    `'pending'`/`'suspended'` are deliberately still unchecked here
    (pre-existing, documented in 20260803000001)."""
    seats = db.table("org_members").select("org_id").eq("user_id", user_id).eq("status", "active").execute()
    org_ids = [s["org_id"] for s in (seats.data or []) if s.get("org_id")]
    if not org_ids:
        return []
    live = (
        db.table("organizations")
        .select("id")
        .in_("id", org_ids)
        .is_("archived_at", "null")
        .neq("status", "lapsed")  # mirror of the SQL clause added in 20260816000002_self_serve_orgs.sql
        .execute()
    )
    return [o["id"] for o in (live.data or []) if o.get("id")]


def visible_artists(db: Client, user_id: str, query):
    """Constrain an `artists` query to what `user_id` may see.

    Returns the query so callers can keep chaining (`.order()`, `.limit()`,
    pagination) — the filter is applied, nothing is executed here.
    """
    org_ids = live_org_ids(db, user_id)
    if not org_ids:
        return query.is_("team_id", "null").eq("user_id", user_id)
    return query.or_(f"and(team_id.is.null,user_id.eq.{user_id}),team_id.in.({','.join(org_ids)})")


def can_access_artist(db: Client, user_id: str, artist_id: str) -> bool:
    """True when `user_id` may reach this artist, personally or via their org."""
    if not artist_id:
        return False
    query = db.table("artists").select("id").eq("id", artist_id)
    return bool(visible_artists(db, user_id, query).execute().data)


def accessible_artist_ids(db: Client, user_id: str) -> list[str]:
    """Every artist id the user may reach, personal and team-owned alike."""
    res = visible_artists(db, user_id, db.table("artists").select("id")).execute()
    return [a["id"] for a in (res.data or []) if a.get("id")]


# ---------------------------------------------------------------------------
# Workspace scoping — presentation, never authorization.
# ---------------------------------------------------------------------------


def _flag(name: str) -> bool:
    # Local env reads, NOT subscriptions.service.licensing_enabled():
    # subscriptions.service imports this module for the entitlements
    # workspaceScope payload, so importing back would be a cycle. This module
    # must stay a leaf.
    return os.getenv(name, "").strip().lower() == "true"


def scoping_enabled() -> bool:
    """Both flags, deliberately. LICENSING_ENABLED off is the 'frozen but live'
    state: orgs may exist and RLS still grants their artists, but no switcher
    renders — narrowing to personal there would hide every transferred artist
    behind a control the user cannot see."""
    return _flag("WORKSPACE_SCOPING_ENABLED") and _flag("LICENSING_ENABLED")


@dataclass(frozen=True)
class Scope:
    """The active workspace for a listing. Construct via the classmethods.

    active=False is the rollback path: filters nothing, byte-identical to
    pre-scoping behaviour. active=True with org_id None is Personal.
    """

    org_id: str | None = None
    active: bool = False

    @classmethod
    def unscoped(cls) -> "Scope":
        return cls()

    @classmethod
    def personal(cls) -> "Scope":
        return cls(org_id=None, active=True)

    @classmethod
    def org(cls, org_id: str) -> "Scope":
        return cls(org_id=org_id, active=True)


def resolve_scope(db: Client, user_id: str, requested: str | None) -> Scope:
    """Turn a `?scope=` param ('personal' | <org_id> | None) into a Scope.

    Falls closed, with two deliberately different failure modes:
    - an EXPLICIT foreign/nonexistent org 404s with the same body either way
      (?scope= must not become an existence oracle for org ids);
    - a stale STORED context (offboarded member, archived org) quietly degrades
      to personal — 404ing there would break the caller's whole app on the
      next request after offboarding. No lazy-clear here either: the billing
      path owns that column's hygiene.

    The stored context is validated with the ACCESS predicate (live_org_ids),
    not the billing one (_resolve_context): a PENDING org bills personal but is
    still the caller's workspace, and hiding its roster would make every
    transferred artist vanish.
    """
    if not scoping_enabled():
        return Scope.unscoped()
    if requested == "personal":
        return Scope.personal()
    if requested:
        member = db.rpc("is_org_member", {"p_user_id": user_id, "p_org_id": requested}).execute()
        if not getattr(member, "data", None):
            raise HTTPException(status_code=404, detail="Organization not found")
        return Scope.org(requested)

    prof = db.table("profiles").select("billing_context_org_id").eq("id", user_id).execute()
    prof_row = (prof.data or [None])[0]
    org_id = prof_row.get("billing_context_org_id") if prof_row else None
    if not org_id:
        return Scope.personal()
    if org_id in live_org_ids(db, user_id):
        return Scope.org(org_id)
    return Scope.personal()


def scoped_artists(db: Client, user_id: str, query, scope: Scope | None):
    """visible_artists ∩ scope — narrowing by CONSTRUCTION.

    The union filter is applied first and unconditionally, so the scope can
    only ever remove rows. A bare `.eq("team_id", X)` without it would WIDEN
    access (archived/lapsed orgs, suspended seats), making every construction
    site of a Scope an authorization decision. An org scope the union doesn't
    cover therefore ANDs down to the empty set, with no sentinel.
    """
    query = visible_artists(db, user_id, query)
    if scope is None or not scope.active:
        return query
    if scope.org_id is None:
        return query.is_("team_id", "null").eq("user_id", user_id)
    return query.eq("team_id", scope.org_id)


def scoped_artist_ids(db: Client, user_id: str, scope: Scope | None) -> list[str]:
    """The artist ids the active workspace shows. Presentation only — use
    accessible_artist_ids for authorization questions."""
    res = scoped_artists(db, user_id, db.table("artists").select("id"), scope).execute()
    return [a["id"] for a in (res.data or []) if a.get("id")]
