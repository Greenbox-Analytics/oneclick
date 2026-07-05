"""Merge-on-write link writes + label snapshot (spec 2026-07-04 §3, §3.1)."""

from unittest.mock import AsyncMock, MagicMock

from boards import service

USER = "u-1"
TASK = "task-1"
BOARD = "board-1"
P1 = "proj-accessible"
P2 = "proj-foreign"
COL = "col-1"


class _RecordingTable:
    """Records .insert()/.delete() payloads; .select()/.eq()/.in_() chain; .execute() returns preset data."""

    def __init__(self, select_data):
        self._select_data = select_data
        self.inserted = []
        self.deleted_filters = []
        self._pending = None
        self._del_eqs = {}

    def select(self, *a, **k):
        self._pending = ("select", None)
        return self

    def insert(self, rows):
        self.inserted.append(rows)
        self._pending = ("insert", rows)
        return self

    def delete(self):
        self._pending = ("delete", None)
        self._del_eqs = {}
        return self

    def eq(self, col, val):
        self._del_eqs[col] = val
        return self

    def in_(self, col, vals):
        self._del_eqs[col] = list(vals)
        return self

    def execute(self):
        if self._pending and self._pending[0] == "delete":
            self.deleted_filters.append(dict(self._del_eqs))
        return MagicMock(data=self._select_data if self._pending and self._pending[0] == "select" else [])


def _db(existing_ids):
    """existing board_task_artists rows for the task."""
    tbl = _RecordingTable([{"artist_id": i} for i in existing_ids])
    db = MagicMock()
    db.table.side_effect = lambda name: tbl
    db._tbl = tbl
    return db


def test_merge_preserves_inaccessible_rows(monkeypatch):
    # existing links: X (co-owner, editor can't access) + A (editor owns)
    db = _db(["X", "A"])
    # editor can access only A (of existing) and submits [A, Y]
    monkeypatch.setattr(service, "_owned_artist_ids", lambda d, u, ids: [i for i in ids if i in ("A", "Y")])
    service._merge_junction(
        db,
        TASK,
        "board_task_artists",
        "artist_id",
        submitted_ids=["A", "Y"],
        labels={"A": "Art A", "Y": "Art Y"},
        access_fn=service._owned_artist_ids,
        user_id=USER,
    )
    # scoped delete only targeted A (accessible existing) — never X
    del_filter = db._tbl.deleted_filters[0]
    assert del_filter["task_id"] == TASK
    assert set(del_filter["artist_id"]) == {"A"}  # X preserved
    # inserted the editor's accessible submitted set with labels
    inserted_ids = {r["artist_id"] for r in db._tbl.inserted[0]}
    assert inserted_ids == {"A", "Y"}
    assert {r["artist_id"]: r["label"] for r in db._tbl.inserted[0]}["Y"] == "Art Y"


def test_merge_noop_when_nothing_accessible_existing(monkeypatch):
    # editor owns nothing existing; submits their own new Y — must NOT delete anything
    db = _db(["X"])  # X is a co-owner's link
    monkeypatch.setattr(service, "_owned_artist_ids", lambda d, u, ids: [i for i in ids if i == "Y"])
    service._merge_junction(
        db,
        TASK,
        "board_task_artists",
        "artist_id",
        submitted_ids=["Y"],
        labels={"Y": "Art Y"},
        access_fn=service._owned_artist_ids,
        user_id=USER,
    )
    assert db._tbl.deleted_filters == []  # X untouched — no delete at all
    assert {r["artist_id"] for r in db._tbl.inserted[0]} == {"Y"}


def test_merge_empty_submitted_clears_only_accessible(monkeypatch):
    # existing: X (co-owner, inaccessible) + A, B (editor owns); editor submits [] → "clear my links"
    db = _db(["X", "A", "B"])
    monkeypatch.setattr(service, "_owned_artist_ids", lambda d, u, ids: [i for i in ids if i in ("A", "B")])
    service._merge_junction(
        db,
        TASK,
        "board_task_artists",
        "artist_id",
        submitted_ids=[],
        labels={},
        access_fn=service._owned_artist_ids,
        user_id=USER,
    )
    # deletes only the editor's accessible existing rows (A, B) — never the co-owner's X
    del_filter = db._tbl.deleted_filters[0]
    assert del_filter["task_id"] == TASK
    assert set(del_filter["artist_id"]) == {"A", "B"}
    # nothing inserted (empty submitted set)
    assert db._tbl.inserted == []


def _chain(data):
    """A minimal chainable supabase-table mock: every builder method returns self;
    .execute() yields the preset data (used for both select and insert on the same table)."""
    m = MagicMock()
    for method in ("select", "insert", "delete", "eq", "in_", "limit", "order", "single"):
        getattr(m, method).return_value = m
    m.execute.return_value = MagicMock(data=data)
    return m


async def test_create_task_emit_payload_carries_accessible_project_ids(monkeypatch):
    """Slack-routing contract: create_task re-sets task['project_ids'] to the ACCESSIBLE
    project ids (foreign ones dropped) on the emitted TASK_CREATED payload, before emit —
    the Slack bridge routes a new task via task['project_ids'][0]."""

    def _table(name):
        return {
            "board_columns": _chain([{"board_id": BOARD}]),  # _column_board_id resolution
            "board_tasks": _chain([{"id": TASK}]),  # insert result
        }.get(name, _chain([]))  # junction selects → no existing rows

    db = MagicMock()
    db.table.side_effect = _table

    monkeypatch.setattr(service.authz, "require_board_edit", lambda *a, **k: None)
    monkeypatch.setattr(service, "_owned_artist_ids", lambda d, u, ids: [])
    # P1 is accessible, P2 (foreign) is dropped
    monkeypatch.setattr(service, "_accessible_project_ids", lambda d, u, ids: [i for i in ids if i == P1])
    monkeypatch.setattr(service, "_accessible_contract_ids", lambda d, u, ids: [])
    emit = AsyncMock()
    monkeypatch.setattr(service.events, "emit", emit)

    await service.create_task(
        db,
        USER,
        {"title": "T", "column_id": COL, "project_ids": [P1, P2], "project_labels": {P1: "Proj 1"}},
    )

    emit.assert_awaited_once()
    (_event, payload), _kwargs = emit.call_args
    assert payload["task"]["project_ids"] == [P1]  # foreign P2 dropped, set before emit


def test_enrich_uses_label_and_stamps_can_open(monkeypatch):
    """_enrich_tasks emits [{id, name, can_open}] per kind: name from the junction label
    (fallback to entity-table lookup on NULL label), can_open via the access helper, plus
    a batched creator profile."""
    tasks = [{"id": TASK, "user_id": USER, "parent_task_id": None}]

    # junction+label rows: artist A has a label; artist B has a NULL label (fallback path)
    def _table(name):
        m = MagicMock()
        data = {
            "board_task_artists": [
                {"task_id": TASK, "artist_id": "A", "label": "Snap A"},
                {"task_id": TASK, "artist_id": "B", "label": None},
            ],
            "board_task_projects": [],
            "board_task_contracts": [],
            "board_task_assignees": [],
            "artists": [{"id": "B", "name": "Live B"}],  # fallback lookup for the NULL-label one
            "profiles": [{"id": USER, "full_name": "Owner", "avatar_url": None}],
        }.get(name, [])
        m.select.return_value = m
        m.in_.return_value = m
        m.eq.return_value = m
        m.execute.return_value = MagicMock(data=data)
        return m

    db = MagicMock()
    db.table.side_effect = _table
    monkeypatch.setattr(service, "_owned_artist_ids", lambda d, u, ids: ["A"])  # owns A, not B

    out = service._enrich_tasks(db, tasks, USER)
    artists = {a["id"]: a for a in out[0]["artists"]}
    assert artists["A"]["name"] == "Snap A"  # label wins
    assert artists["B"]["name"] == "Live B"  # fallback lookup for NULL label
    assert artists["A"]["can_open"] is True
    assert artists["B"]["can_open"] is False
    assert out[0]["creator"]["full_name"] == "Owner"
