"""Expense Tracker rollup, narrowed to the workspace.

`get_expenses_summary` used to build its project set from `project_members`
alone, which was wrong in both directions once workspaces existed: switching
the workspace changed nothing (memberships span every workspace), and an org
member with no explicit membership row saw nothing for projects the org owns.

The One Rule applies — a project's workspace is the owner of its artist:

    any scope:      projects of the workspace's artists, by OWNERSHIP
    Personal only:  plus membership-grant projects, EXCLUDING those whose
                    artist one of the caller's own live orgs owns
    unscoped:       memberships only — byte-identical to pre-scoping

As in test_workspace_scope.py, mocks can't evaluate a PostgREST filter, so
these pin the SHAPE of the queries: which project ids reach the
`project_expenses` filter, and which never do.
"""

import asyncio
from unittest.mock import MagicMock

from artist_access import Scope
from projects import service
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ORG_A = "20000000-0000-0000-0000-00000000000a"
MY_ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
ORG_ARTIST = "aaaaaaaa-0000-0000-0000-000000000002"
OTHER_ARTIST = "aaaaaaaa-0000-0000-0000-000000000009"
P_MINE = "bbbbbbbb-0000-0000-0000-000000000001"  # my personal artist's project
P_ORG = "bbbbbbbb-0000-0000-0000-000000000002"  # my org's artist's project
P_GRANT = "bbbbbbbb-0000-0000-0000-000000000003"  # membership grant, foreign artist
P_ORG_GRANT = "bbbbbbbb-0000-0000-0000-000000000004"  # membership grant on MY org's artist


def _expense(expense_id, project_id):
    return {
        "id": expense_id,
        "project_id": project_id,
        "description": "Studio time",
        "amount": 100.0,
        "currency": "USD",  # USD → _attach_amount_usd stays off the fx tables
        "category": "studio",
        "incurred_on": "2026-06-01",
    }


class _Recorder:
    """Routes table() by (name, select-columns) so the same table can serve
    different rows to the distinct queries get_expenses_summary issues —
    a name-keyed mock can't, because `projects` is read three different ways."""

    def __init__(self, mapping):
        self.mapping = mapping  # {(table, select_sig): rows}
        self.requested: list = []
        self.expense_filters: list = []
        self.artist_filters: list = []

    def __call__(self, name):
        rec = self

        class _B(MockQueryBuilder):
            def select(self, *args, **kwargs):
                sig = args[0] if args else "*"
                rec.requested.append((name, sig))
                rows = rec.mapping.get((name, sig), [])
                self.execute.return_value = MagicMock(data=rows, count=len(rows))
                return self

        b = _B()
        if name in ("artists", "project_expenses"):
            sink = rec.artist_filters if name == "artists" else rec.expense_filters
            for method in ("eq", "in_", "is_", "or_"):
                original = getattr(b, method)

                def _capture(*args, _o=original, _m=method, _s=sink, **kwargs):
                    _s.append((_m, args))
                    return _o(*args, **kwargs)

                setattr(b, method, _capture)
        return b


def _db(rec):
    db = MagicMock()
    db.table.side_effect = rec
    return db


def _run(rec, **kwargs):
    return asyncio.run(service.get_expenses_summary(_db(rec), TEST_USER_ID, **kwargs))


def _expense_project_set(rec):
    """The project-id set the final expenses query was filtered on."""
    for method, args in rec.expense_filters:
        if method == "in_" and args[0] == "project_id":
            return set(args[1])
    return None


# ---------------------------------------------------------------------------
# The rollback path
# ---------------------------------------------------------------------------


def test_unscoped_is_memberships_only():
    rec = _Recorder(
        {
            ("project_members", "project_id"): [{"project_id": P_GRANT}],
            ("projects", "id, name, artist_id"): [{"id": P_GRANT, "name": "Album", "artist_id": OTHER_ARTIST}],
            ("artists", "id, name"): [{"id": OTHER_ARTIST, "name": "Someone"}],
            ("project_expenses", "*"): [_expense("e1", P_GRANT)],
        }
    )
    rows = _run(rec, scope=Scope.unscoped())

    assert _expense_project_set(rec) == {P_GRANT}
    assert [r["id"] for r in rows] == ["e1"]
    # No ownership derivation ran — the artist roster was never consulted.
    assert ("artists", "id") not in rec.requested


def test_no_scope_argument_behaves_as_unscoped():
    rec = _Recorder(
        {
            ("project_members", "project_id"): [{"project_id": P_GRANT}],
            ("projects", "id, name, artist_id"): [{"id": P_GRANT, "name": "Album", "artist_id": OTHER_ARTIST}],
            ("project_expenses", "*"): [_expense("e1", P_GRANT)],
        }
    )
    _run(rec)
    assert _expense_project_set(rec) == {P_GRANT}


# ---------------------------------------------------------------------------
# Org workspace — ownership only, grants dropped
# ---------------------------------------------------------------------------


def test_org_scope_lists_by_artist_ownership_not_membership():
    """The fix's whole point: an org member sees org-project expenses with NO
    project_members row, and their cross-workspace grants disappear."""
    rec = _Recorder(
        {
            ("project_members", "project_id"): [{"project_id": P_GRANT}],  # must be ignored
            ("org_members", "org_id"): [{"org_id": ORG_A}],
            ("organizations", "id"): [{"id": ORG_A}],
            ("artists", "id"): [{"id": ORG_ARTIST}],
            ("projects", "id"): [{"id": P_ORG}],
            ("projects", "id, name, artist_id"): [{"id": P_ORG, "name": "Org LP", "artist_id": ORG_ARTIST}],
            ("artists", "id, name"): [{"id": ORG_ARTIST, "name": "Nova"}],
            ("project_expenses", "*"): [_expense("e1", P_ORG)],
        }
    )
    rows = _run(rec, scope=Scope.org(ORG_A))

    assert _expense_project_set(rec) == {P_ORG}
    assert [r["id"] for r in rows] == ["e1"]
    # The narrowing was applied to the artist roster (not trusted from data).
    assert ("eq", ("team_id", ORG_A)) in rec.artist_filters
    # The grant path never ran in an org workspace.
    assert ("projects", "id, artist_id") not in rec.requested


def test_empty_org_workspace_short_circuits_without_querying_expenses():
    rec = _Recorder(
        {
            ("project_members", "project_id"): [{"project_id": P_GRANT}],
            ("org_members", "org_id"): [],
            ("organizations", "id"): [],
            ("artists", "id"): [],
            ("project_expenses", "*"): [_expense("leak", P_GRANT)],
        }
    )
    assert _run(rec, scope=Scope.org(ORG_A)) == []
    assert rec.expense_filters == []


# ---------------------------------------------------------------------------
# Personal workspace — ownership ∪ grants, minus my orgs' artists
# ---------------------------------------------------------------------------


def test_personal_scope_unions_owned_and_kept_grants():
    """Grants survive in Personal — except on artists MY live orgs own (those
    list in that org's workspace, same rule as the registry's Shared with me).
    A grant on a stranger's artist stays: grant-only access belongs to Personal."""
    rec = _Recorder(
        {
            ("project_members", "project_id"): [{"project_id": P_GRANT}, {"project_id": P_ORG_GRANT}],
            ("org_members", "org_id"): [{"org_id": ORG_A}],
            ("organizations", "id"): [{"id": ORG_A}],
            ("artists", "id"): [{"id": MY_ARTIST}],
            ("projects", "id"): [{"id": P_MINE}],
            ("projects", "id, artist_id"): [
                {"id": P_GRANT, "artist_id": OTHER_ARTIST},
                {"id": P_ORG_GRANT, "artist_id": ORG_ARTIST},
            ],
            ("artists", "id, team_id"): [
                {"id": OTHER_ARTIST, "team_id": None},
                {"id": ORG_ARTIST, "team_id": ORG_A},
            ],
            ("projects", "id, name, artist_id"): [
                {"id": P_MINE, "name": "Mine", "artist_id": MY_ARTIST},
                {"id": P_GRANT, "name": "Granted", "artist_id": OTHER_ARTIST},
            ],
            ("artists", "id, name"): [{"id": MY_ARTIST, "name": "Me"}],
            ("project_expenses", "*"): [_expense("e1", P_MINE), _expense("e2", P_GRANT)],
        }
    )
    rows = _run(rec, scope=Scope.personal())

    assert _expense_project_set(rec) == {P_MINE, P_GRANT}
    assert {r["id"] for r in rows} == {"e1", "e2"}
    # The ownership half was narrowed to personal artists.
    assert ("is_", ("team_id", "null")) in rec.artist_filters


def test_personal_scope_with_nothing_at_all_returns_empty():
    rec = _Recorder(
        {
            ("project_members", "project_id"): [],
            ("artists", "id"): [],
            ("project_expenses", "*"): [_expense("leak", P_GRANT)],
        }
    )
    assert _run(rec, scope=Scope.personal()) == []
    assert rec.expense_filters == []
