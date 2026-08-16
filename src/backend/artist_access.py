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
"""

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
