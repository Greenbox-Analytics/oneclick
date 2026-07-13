"""Tests for the Boards (Workspace) endpoints.

Acceptance criteria:
1. Columns: list, create, update, delete, create defaults
2. Tasks: list, create, get detail, update, delete, reorder
3. Parent tasks: list, create
4. Calendar: list tasks in date range
5. Comments: add, delete
"""

from unittest.mock import MagicMock, patch

from tests.conftest import _DEFAULT_USAGE_ROW, _PRO_SUB_ROW, _PRO_TIER_ROW, _SUBSCRIPTION_TABLES, TEST_USER_ID


def _pro_sub_builder(name):
    """Return a builder with Pro subscription data for subscription/entitlements tables.

    Includes 'profiles' (with is_admin=False default) because
    EntitlementsService.get_for_user now checks is_db_admin via the profiles
    table — without this branch the lookup would fall through to the test's
    domain builders and either crash or skew the sequence index.
    """
    b = BoardMockBuilder([])
    if name == "subscriptions":
        b._data = [_PRO_SUB_ROW]
    elif name == "tier_entitlements":
        b._data = [_PRO_TIER_ROW]
    elif name == "tier_overrides":
        b._data = []
    elif name == "usage_counters":
        b._data = [_DEFAULT_USAGE_ROW]
    elif name == "profiles":
        b._data = []  # is_db_admin returns False on empty result
    b._count = len(b._data)
    return b


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

COLUMN_ID = "col-00000000-0000-0000-0000-000000000001"
TASK_ID = "task-0000-0000-0000-0000-000000000001"
COMMENT_ID = "cmnt-0000-0000-0000-0000-000000000001"
PARENT_ID = "prnt-0000-0000-0000-0000-000000000001"
ARTIST_ID = "art-00000000-0000-0000-0000-000000000001"
BOARD_ID = "b0000000-0000-0000-0000-000000000001"

SAMPLE_COLUMN = {
    "id": COLUMN_ID,
    "user_id": TEST_USER_ID,
    "board_id": BOARD_ID,
    "title": "To Do",
    "color": "#6366f1",
    "position": 0,
}

SAMPLE_TASK = {
    "id": TASK_ID,
    "user_id": TEST_USER_ID,
    "board_id": BOARD_ID,
    "column_id": COLUMN_ID,
    "title": "My Task",
    "description": "Task description",
    "priority": "medium",
    "start_date": "2026-04-01",
    "due_date": "2026-04-30",
    "position": 0,
    "is_parent": False,
    "parent_task_id": None,
}

SAMPLE_COMMENT = {
    "id": COMMENT_ID,
    "task_id": TASK_ID,
    "user_id": TEST_USER_ID,
    "content": "Great progress!",
    "created_at": "2026-04-10T00:00:00Z",
}

SAMPLE_PARENT = {
    "id": PARENT_ID,
    "user_id": TEST_USER_ID,
    "board_id": BOARD_ID,
    "title": "Parent Task",
    "is_parent": True,
    "column_id": None,
    "position": 0,
}


# ---------------------------------------------------------------------------
# Extended MockQueryBuilder with or_ / filter and single() awareness
# ---------------------------------------------------------------------------


class BoardMockBuilder:
    """Chainable mock builder that also supports or_(), filter(), and single().

    When .single() is called before .execute(), the execute return_value is
    automatically coerced from a list to its first element (a dict), matching
    how the real supabase-py client behaves.
    """

    def __init__(self, data: list, count: int | None = None):
        self._data = data
        self._count = count if count is not None else len(data)
        self._is_single = False

    def _make_execute(self):
        if self._is_single:
            single_data = self._data[0] if self._data else None
            return MagicMock(return_value=MagicMock(data=single_data, count=self._count))
        return MagicMock(return_value=MagicMock(data=self._data, count=self._count))

    @property
    def execute(self):
        return self._make_execute()

    # --- chainable methods ---------------------------------------------------

    def select(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
        return self

    def upsert(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def neq(self, *args, **kwargs):
        return self

    def in_(self, *args, **kwargs):
        return self

    def gt(self, *args, **kwargs):
        return self

    def gte(self, *args, **kwargs):
        return self

    def lt(self, *args, **kwargs):
        return self

    def lte(self, *args, **kwargs):
        return self

    def like(self, *args, **kwargs):
        return self

    def ilike(self, *args, **kwargs):
        return self

    def is_(self, *args, **kwargs):
        return self

    def or_(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def range(self, *args, **kwargs):
        return self

    def single(self, *args, **kwargs):
        self._is_single = True
        return self

    def maybe_single(self, *args, **kwargs):
        self._is_single = True
        return self


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _builder(data: list, count: int | None = None) -> BoardMockBuilder:
    """Return a BoardMockBuilder pre-loaded with the given data list."""
    return BoardMockBuilder(data, count)


def _single_builder(row: dict | None) -> BoardMockBuilder:
    """Return a BoardMockBuilder pre-configured to return a single row dict."""
    b = BoardMockBuilder([row] if row else [])
    b._is_single = True
    return b


def _authz_board_builder(name):
    """Return a well-formed personal board (owned by TEST_USER_ID) for the `boards`
    table, and an empty result for `team_members`, so future `teams/authz.py`
    lookups (get_board / is_team_member / require_board_access/edit) resolve to
    an authorized personal board instead of crashing or 403ing.

    Returns None for any other table name so callers can fall through to their
    own domain-specific mock data.
    """
    if name == "boards":
        # NOTE: `authz.get_board` / `ensure_personal_board` use `.limit(1).execute()`
        # (no `.single()`) and index `.data[0]` — so this must be a LIST-shaped
        # builder (`_builder`), not `_single_builder` (which always collapses
        # `.data` to a bare dict and breaks that `[0]` indexing).
        return _builder([{"id": BOARD_ID, "team_id": None, "owner_id": TEST_USER_ID, "archived": False}])
    if name == "team_members":
        return _builder([])
    return None


def _sequence_side_effect(sequences: list[list]) -> callable:
    """Return a side_effect function that returns builders from a pre-defined sequence.

    Each element of `sequences` is a data list for one table() call.
    Subscription/entitlements tables are handled transparently with Pro data so the
    subscription gate always passes in tests that aren't testing the gate.
    Once exhausted, returns empty builders.
    """
    idx = [0]

    def _side_effect(name):
        if name in _SUBSCRIPTION_TABLES:
            return _pro_sub_builder(name)
        _b = _authz_board_builder(name)
        if _b is not None:
            return _b
        data = sequences[idx[0]] if idx[0] < len(sequences) else []
        idx[0] += 1
        return _builder(data)

    return _side_effect


def _sequence_side_effect_mixed(steps: list) -> callable:
    """Like _sequence_side_effect but accepts (data, is_single) tuples or plain lists.

    Steps can be:
      - a list  -> BoardMockBuilder(data)
      - a dict  -> single-row builder (BoardMockBuilder where _is_single=True)
      - None    -> single-row builder returning None

    Subscription/entitlements tables are handled transparently with Pro data.
    """
    idx = [0]

    def _side_effect(name):
        if name in _SUBSCRIPTION_TABLES:
            return _pro_sub_builder(name)
        _b = _authz_board_builder(name)
        if _b is not None:
            return _b
        step = steps[idx[0]] if idx[0] < len(steps) else []
        idx[0] += 1
        if isinstance(step, dict):
            return _single_builder(step)
        if step is None:
            return _single_builder(None)
        return _builder(step)

    return _side_effect


# ===========================================================================
# Columns
# ===========================================================================


class TestListColumns:
    def test_list_columns_returns_200_and_columns_key(self, client, mock_supabase):
        """GET /boards/columns returns 200 with a columns list."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_COLUMN])

        response = client.get("/boards/columns")

        assert response.status_code == 200
        body = response.json()
        assert "columns" in body
        assert len(body["columns"]) == 1
        assert body["columns"][0]["id"] == COLUMN_ID

    def test_list_columns_empty_returns_empty_list(self, client, mock_supabase):
        """GET /boards/columns with no data returns an empty list."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.get("/boards/columns")

        assert response.status_code == 200
        assert response.json()["columns"] == []

    def test_list_columns_with_artist_id_filter(self, client, mock_supabase):
        """GET /boards/columns?artist_id=... passes artist_id filter."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_COLUMN])

        response = client.get(f"/boards/columns?artist_id={ARTIST_ID}")

        assert response.status_code == 200
        assert "columns" in response.json()


class TestCreateColumn:
    def test_create_column_returns_200_with_column(self, client, mock_supabase):
        """POST /boards/columns returns created column."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_COLUMN])

        response = client.post(
            "/boards/columns",
            json={"title": "To Do", "color": "#6366f1", "position": 0},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == COLUMN_ID
        assert body["title"] == "To Do"

    def test_create_column_returns_500_when_insert_fails(self, client, mock_supabase):
        """POST /boards/columns returns 500 when Supabase insert returns empty data."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.post("/boards/columns", json={"title": "Broken"})

        assert response.status_code == 500

    def test_create_column_with_only_title(self, client, mock_supabase):
        """POST /boards/columns works with just a title (other fields optional)."""
        col = {**SAMPLE_COLUMN, "title": "Minimal"}
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([col])

        response = client.post("/boards/columns", json={"title": "Minimal"})

        assert response.status_code == 200
        assert response.json()["title"] == "Minimal"


class TestUpdateColumn:
    def test_update_column_returns_updated_data(self, client, mock_supabase):
        """PUT /boards/columns/{column_id} returns updated column."""
        updated = {**SAMPLE_COLUMN, "title": "In Progress"}
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([updated])

        response = client.put(f"/boards/columns/{COLUMN_ID}", json={"title": "In Progress"})

        assert response.status_code == 200
        assert response.json()["title"] == "In Progress"

    def test_update_column_returns_404_when_not_found(self, client, mock_supabase):
        """PUT /boards/columns/{column_id} returns 404 when column absent."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.put(f"/boards/columns/{COLUMN_ID}", json={"title": "Ghost"})

        assert response.status_code == 404


class TestDeleteColumn:
    def test_delete_column_returns_success_true(self, client, mock_supabase):
        """DELETE /boards/columns/{column_id} returns success=True."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_COLUMN])

        response = client.delete(f"/boards/columns/{COLUMN_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_column_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/columns/{column_id} returns 404 when column absent."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.delete(f"/boards/columns/{COLUMN_ID}")

        assert response.status_code == 404


class TestCreateDefaultColumns:
    def test_create_defaults_returns_five_columns(self, client, mock_supabase):
        """POST /boards/columns/defaults creates and returns five default columns."""
        default_columns = [
            {**SAMPLE_COLUMN, "id": f"col-{i}", "title": title, "position": i}
            for i, title in enumerate(["Backlog", "To Do", "In Progress", "Review", "Done"])
        ]

        call_count = [0]

        def _side_effect(name):
            _b = _authz_board_builder(name)
            if _b is not None:
                return _b
            idx = call_count[0] % len(default_columns)
            data = [default_columns[idx]]
            call_count[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = client.post("/boards/columns/defaults")

        assert response.status_code == 200
        body = response.json()
        assert "columns" in body
        assert len(body["columns"]) == 5

    def test_create_defaults_with_artist_id(self, client, mock_supabase):
        """POST /boards/columns/defaults?artist_id=... passes artist_id to each column."""
        default_columns = [
            {**SAMPLE_COLUMN, "id": f"col-{i}", "title": t, "position": i, "artist_id": ARTIST_ID}
            for i, t in enumerate(["Backlog", "To Do", "In Progress", "Review", "Done"])
        ]

        call_count = [0]

        def _side_effect(name):
            _b = _authz_board_builder(name)
            if _b is not None:
                return _b
            idx = call_count[0] % len(default_columns)
            data = [default_columns[idx]]
            call_count[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = client.post(f"/boards/columns/defaults?artist_id={ARTIST_ID}")

        assert response.status_code == 200
        assert "columns" in response.json()

    def test_create_defaults_with_board_id_targets_that_board(self, client, mock_supabase):
        """POST /boards/columns/defaults?board_id=... seeds the given board directly and does
        NOT fall back to the personal board — ensure_personal_board must be bypassed."""
        default_columns = [
            {**SAMPLE_COLUMN, "id": f"col-{i}", "title": t, "position": i, "board_id": BOARD_ID}
            for i, t in enumerate(["Backlog", "To Do", "In Progress", "Review", "Done"])
        ]

        call_count = [0]

        def _side_effect(name):
            _b = _authz_board_builder(name)
            if _b is not None:
                return _b
            idx = call_count[0] % len(default_columns)
            data = [default_columns[idx]]
            call_count[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        with patch(
            "boards.service.ensure_personal_board",
            side_effect=AssertionError("ensure_personal_board must not run when board_id is provided"),
        ):
            response = client.post(f"/boards/columns/defaults?board_id={BOARD_ID}")

        assert response.status_code == 200
        assert len(response.json()["columns"]) == 5


# ===========================================================================
# Tasks
# ===========================================================================


class TestListTasks:
    def test_list_tasks_returns_200_with_tasks_key(self, client, mock_supabase):
        """GET /boards/tasks returns 200 with tasks list."""
        # Sequence: board_tasks (primary), then junction tables (board_task_artists,
        # board_task_projects, board_task_contracts), then artists for name resolution.
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], [], []])

        response = client.get("/boards/tasks")

        assert response.status_code == 200
        body = response.json()
        assert "tasks" in body

    def test_list_tasks_with_column_filter(self, client, mock_supabase):
        """GET /boards/tasks?column_id=... filters by column."""
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], [], []])

        response = client.get(f"/boards/tasks?column_id={COLUMN_ID}")

        assert response.status_code == 200
        assert "tasks" in response.json()

    def test_list_tasks_empty_returns_empty_list(self, client, mock_supabase):
        """GET /boards/tasks with no tasks returns empty list."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.get("/boards/tasks")

        assert response.status_code == 200
        assert response.json()["tasks"] == []

    def test_list_tasks_with_pagination(self, client, mock_supabase):
        """GET /boards/tasks?page=1&page_size=10 returns paginated envelope."""
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], [], []])

        response = client.get("/boards/tasks?page=1&page_size=10")

        assert response.status_code == 200
        body = response.json()
        # Paginated response returns data/total/page/page_size
        assert "data" in body or "tasks" in body


class TestCreateTask:
    def test_create_task_returns_task_with_id(self, client, mock_supabase):
        """POST /boards/tasks creates a task and returns it."""
        # Sequence: 1) board_tasks count (gate) → empty/count=0,
        # 2) board_columns (_column_board_id resolution), 3) board_tasks insert
        mock_supabase.table.side_effect = _sequence_side_effect([[], [SAMPLE_COLUMN], [SAMPLE_TASK]])

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "My Task", "column_id": COLUMN_ID, "priority": "medium"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == TASK_ID
        assert body["title"] == "My Task"

    def test_create_task_returns_500_when_insert_fails(self, client, mock_supabase):
        """POST /boards/tasks returns 500 when insert returns no data."""
        # Sequence: 1) board_tasks count (gate) → empty,
        # 2) board_columns (_column_board_id resolution), 3) board_tasks insert → empty → 500
        mock_supabase.table.side_effect = _sequence_side_effect([[], [SAMPLE_COLUMN], []])

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "Broken Task", "column_id": COLUMN_ID},
            )

        assert response.status_code == 500

    def test_create_task_with_due_date(self, client, mock_supabase):
        """POST /boards/tasks with due_date converts date to string."""
        task_with_date = {**SAMPLE_TASK, "due_date": "2026-04-30"}
        # Sequence: 1) board_tasks count (gate), 2) board_columns (_column_board_id), 3) board_tasks insert
        mock_supabase.table.side_effect = _sequence_side_effect([[], [SAMPLE_COLUMN], [task_with_date]])

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "Due Date Task", "column_id": COLUMN_ID, "due_date": "2026-04-30"},
            )

        assert response.status_code == 200

    def test_create_task_with_artist_ids_calls_junction_table(self, client, mock_supabase):
        """POST /boards/tasks with artist_ids sets junction rows (merge-on-write)."""
        # Sequence: 1) board_tasks count (gate), 2) artists (owned-artist filter for board
        # resolution), 3) board_columns (_column_board_id), 4) board_tasks insert,
        # then _merge_junction(artists): 5) board_task_artists select (existing links → none),
        # 6) artists (_owned_artist_ids ownership filter for the submitted set),
        # 7) board_task_artists insert; then _merge_junction(projects): 8) board_task_projects
        # select (existing → none); _merge_junction(contracts): 9) board_task_contracts select.
        mock_supabase.table.side_effect = _sequence_side_effect(
            [
                [],
                [{"id": ARTIST_ID}],
                [SAMPLE_COLUMN],
                [SAMPLE_TASK],
                [],
                [{"id": ARTIST_ID}],
                [],
                [],
                [],
            ]
        )

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "Tagged Task", "column_id": COLUMN_ID, "artist_ids": [ARTIST_ID]},
            )

        assert response.status_code == 200


class TestGetTaskDetail:
    def test_get_task_detail_returns_task_with_enriched_fields(self, client, mock_supabase):
        """GET /boards/tasks/{task_id}/detail returns task with comments, children, parent."""
        detail_task = {**SAMPLE_TASK, "is_parent": False, "parent_task_id": None}

        # get_task_detail sequence:
        # 1. board_tasks -> list (main task lookup via .limit(1), gated by require_board_access)
        # 2. board_task_artists -> list
        # 3. board_task_projects -> list
        # 4. board_task_contracts -> list
        # 5. board_task_comments -> list
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
                [detail_task],  # .limit(1) lookup for main task (list-shaped, not .single())
                [],  # board_task_artists
                [],  # board_task_projects
                [],  # board_task_contracts
                [SAMPLE_COMMENT],  # board_task_comments
            ]
        )

        response = client.get(f"/boards/tasks/{TASK_ID}/detail")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == TASK_ID
        assert "comments" in body
        assert "children" in body
        assert "parent" in body

    def test_get_task_detail_returns_404_when_not_found(self, client, mock_supabase):
        """GET /boards/tasks/{task_id}/detail returns 404 when task absent."""
        mock_supabase.table.side_effect = _sequence_side_effect_mixed([None])

        response = client.get(f"/boards/tasks/{TASK_ID}/detail")

        assert response.status_code == 404


class TestUpdateTask:
    def test_update_task_returns_updated_task(self, client, mock_supabase):
        """PUT /boards/tasks/{task_id} returns updated task."""
        updated_task = {**SAMPLE_TASK, "title": "Updated Title"}

        # update_task sequence:
        # 1. board_tasks -> single (router pre-read of existing column_id for analytics)
        # 2. board_tasks -> list (service task_row fetch: id/board_id/parent_task_id, gates require_board_edit)
        # 3. board_columns -> list (service _column_board_id: §7.3 dst-board resolution; same board, no move)
        # 4. board_columns -> single (for Done-column title check in service)
        # 5. board_tasks -> list (update result)
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
                {"column_id": COLUMN_ID},  # router pre-read of existing task
                [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
                [{"board_id": BOARD_ID}],  # service _column_board_id (dst == src, no cross-board move)
                {"id": COLUMN_ID, "title": "In Progress"},  # single for board_columns
                [updated_task],  # board_tasks update result
            ]
        )

        with patch("boards.service.events.emit"):
            response = client.put(
                f"/boards/tasks/{TASK_ID}",
                json={"title": "Updated Title", "column_id": COLUMN_ID},
            )

        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"

    def test_update_task_returns_404_when_not_found(self, client, mock_supabase):
        """PUT /boards/tasks/{task_id} returns 404 when task absent."""
        # 1. board_tasks router pre-read (existing column_id for analytics)
        # 2. board_tasks service task_row fetch (gates require_board_edit)
        # 3. board_columns _column_board_id (§7.3 dst-board resolution; same board, no move)
        # 4. board_columns returns the column, 5. board_tasks update returns empty
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
                {"column_id": COLUMN_ID},  # router pre-read
                [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
                [{"board_id": BOARD_ID}],  # service _column_board_id (dst == src, no cross-board move)
                {"id": COLUMN_ID, "title": "To Do"},  # service done-check lookup
                [],  # update returns empty -> task = {}
            ]
        )

        with patch("boards.service.events.emit"):
            response = client.put(
                f"/boards/tasks/{TASK_ID}",
                json={"title": "Ghost", "column_id": COLUMN_ID},
            )

        assert response.status_code == 404

    def test_update_task_without_column_id_skips_done_check(self, client, mock_supabase):
        """PUT /boards/tasks/{task_id} without column_id skips the Done-column check."""
        updated_task = {**SAMPLE_TASK, "title": "No Column Update"}
        # No board_columns lookup, just board_tasks update
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([updated_task])

        with patch("boards.service.events.emit"):
            response = client.put(f"/boards/tasks/{TASK_ID}", json={"title": "No Column Update"})

        assert response.status_code == 200


class TestDeleteTask:
    def test_delete_task_returns_success_true(self, client, mock_supabase):
        """DELETE /boards/tasks/{task_id} returns success=True."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_TASK])

        response = client.delete(f"/boards/tasks/{TASK_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_task_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/tasks/{task_id} returns 404 when task absent."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.delete(f"/boards/tasks/{TASK_ID}")

        assert response.status_code == 404


class TestReorderTasks:
    def test_reorder_tasks_returns_success_true(self, client, mock_supabase):
        """PUT /boards/tasks/reorder accepts batch reorder payload."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_TASK])

        response = client.put(
            "/boards/tasks/reorder",
            json={"reorders": [{"task_id": TASK_ID, "target_column_id": COLUMN_ID, "position": 1}]},
        )

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_reorder_tasks_with_multiple_items(self, client, mock_supabase):
        """PUT /boards/tasks/reorder handles multiple reorder items."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_TASK])
        task_id_2 = "task-0000-0000-0000-0000-000000000002"

        response = client.put(
            "/boards/tasks/reorder",
            json={
                "reorders": [
                    {"task_id": TASK_ID, "target_column_id": COLUMN_ID, "position": 0},
                    {"task_id": task_id_2, "target_column_id": COLUMN_ID, "position": 1},
                ]
            },
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_reorder_tasks_empty_list(self, client, mock_supabase):
        """PUT /boards/tasks/reorder with empty reorders still returns success."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.put("/boards/tasks/reorder", json={"reorders": []})

        assert response.status_code == 200
        assert response.json()["success"] is True


# ===========================================================================
# Parent Tasks
# ===========================================================================


class TestListParents:
    def test_list_parents_returns_parents_and_ungrouped(self, client, mock_supabase):
        """GET /boards/parents returns parents and ungrouped children."""
        # get_all_parents_with_children sequence:
        # 1. board_tasks (parent query, is_parent=True)
        # 2. board_tasks (children query, is_parent=false/null)
        # 3-5. _enrich_tasks junction tables: board_task_artists, board_task_projects,
        #      board_task_contracts (called once on parents + children combined)
        # 6. board_columns (for column_title mapping)
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_PARENT], [], [], [], [], []])

        response = client.get("/boards/parents")

        assert response.status_code == 200
        body = response.json()
        assert "parents" in body
        assert "ungrouped" in body

    def test_list_parents_with_search_filter(self, client, mock_supabase):
        """GET /boards/parents?search=... applies search filtering."""
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_PARENT], [], [], [], [], []])

        response = client.get("/boards/parents?search=Parent")

        assert response.status_code == 200
        body = response.json()
        assert "parents" in body

    def test_list_parents_with_artist_id_filter(self, client, mock_supabase):
        """GET /boards/parents?artist_id=... applies artist_id filtering."""
        parent_with_artist = {**SAMPLE_PARENT, "artist_ids": [ARTIST_ID]}
        mock_supabase.table.side_effect = _sequence_side_effect([[parent_with_artist], [], [], [], [], []])

        response = client.get(f"/boards/parents?artist_id={ARTIST_ID}")

        assert response.status_code == 200
        assert "parents" in response.json()


class TestCreateParent:
    def test_create_parent_task_returns_task(self, client, mock_supabase):
        """POST /boards/parents creates a parent task and returns it."""
        # _merge_junction selects existing junction rows first — return empty for those tables
        # (the generic SAMPLE_PARENT row lacks the fk columns the merge reads).
        mock_supabase.table.side_effect = lambda name: (
            _builder([])
            if name in ("board_task_artists", "board_task_projects", "board_task_contracts")
            else _authz_board_builder(name) or _builder([SAMPLE_PARENT])
        )

        response = client.post(
            "/boards/parents",
            json={"title": "Parent Task", "priority": "high"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == PARENT_ID
        assert body["title"] == "Parent Task"

    def test_create_parent_task_returns_500_when_insert_fails(self, client, mock_supabase):
        """POST /boards/parents returns 500 when insert fails."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.post("/boards/parents", json={"title": "Broken Parent"})

        assert response.status_code == 500

    def test_create_parent_task_with_due_date(self, client, mock_supabase):
        """POST /boards/parents with due_date converts date to string."""
        parent_with_date = {**SAMPLE_PARENT, "due_date": "2026-05-01"}
        # _merge_junction selects existing junction rows first — return empty for those tables.
        mock_supabase.table.side_effect = lambda name: (
            _builder([])
            if name in ("board_task_artists", "board_task_projects", "board_task_contracts")
            else _authz_board_builder(name) or _builder([parent_with_date])
        )

        response = client.post(
            "/boards/parents",
            json={"title": "Timed Parent", "due_date": "2026-05-01"},
        )

        assert response.status_code == 200


# ===========================================================================
# Calendar
# ===========================================================================


class TestCalendar:
    def test_calendar_tasks_returns_tasks_in_date_range(self, client, mock_supabase):
        """GET /boards/calendar returns tasks within the given date range."""
        # get_tasks_by_date_range sequence:
        # 1. board_tasks (due_date query)
        # 2-4. _enrich_tasks: board_task_artists, board_task_projects, board_task_contracts
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], []])

        response = client.get("/boards/calendar?start=2026-04-01&end=2026-04-30")

        assert response.status_code == 200
        body = response.json()
        assert "tasks" in body

    def test_calendar_missing_start_returns_422(self, client, mock_supabase):
        """GET /boards/calendar without start param returns 422."""
        response = client.get("/boards/calendar?end=2026-04-30")

        assert response.status_code == 422

    def test_calendar_missing_end_returns_422(self, client, mock_supabase):
        """GET /boards/calendar without end param returns 422."""
        response = client.get("/boards/calendar?start=2026-04-01")

        assert response.status_code == 422

    def test_calendar_tasks_query_uses_due_date_only(self, client, mock_supabase):
        """GET /boards/calendar keys off due_date only — start_date does not place a task
        on the calendar, so the range is fetched with a single board_tasks query (no
        separate start_date query, no dedup)."""
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], []])

        response = client.get("/boards/calendar?start=2026-04-01&end=2026-04-30")

        assert response.status_code == 200
        assert len(response.json()["tasks"]) == 1
        # Only one board_tasks range query should run (due_date). _resolve_read_board_ids
        # reads `boards`/`team_members`, and _enrich_tasks reads the junction tables — none
        # of those is `board_tasks`, and this task has no parent, so board_tasks is queried
        # exactly once. Before this change it was queried twice (due_date + start_date).
        board_task_queries = [c for c in mock_supabase.table.call_args_list if c.args and c.args[0] == "board_tasks"]
        assert len(board_task_queries) == 1


# ===========================================================================
# Comments
# ===========================================================================


class TestAddComment:
    def test_add_comment_returns_comment(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments returns created comment."""

        # create_comment now resolves the task's board (_task_board_id reads board_tasks)
        # and gates via require_board_edit before inserting the comment.
        def _router(name):
            _b = _authz_board_builder(name)
            if _b is not None:
                return _b
            if name == "board_tasks":
                return _builder([SAMPLE_TASK])
            return _builder([SAMPLE_COMMENT])

        mock_supabase.table.side_effect = _router

        response = client.post(
            f"/boards/tasks/{TASK_ID}/comments",
            json={"content": "Great progress!"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == COMMENT_ID
        assert body["content"] == "Great progress!"
        assert body["task_id"] == TASK_ID

    def test_add_comment_returns_500_when_insert_fails(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments returns 500 when insert fails."""
        # Sequence: 1) board_tasks (ownership check passes — task owned by caller),
        # 2) board_task_comments insert → empty → 500
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], []])

        response = client.post(
            f"/boards/tasks/{TASK_ID}/comments",
            json={"content": "Lost comment"},
        )

        assert response.status_code == 500

    def test_add_comment_requires_content_field(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments returns 422 when content missing."""
        response = client.post(f"/boards/tasks/{TASK_ID}/comments", json={})

        assert response.status_code == 422


class TestDeleteComment:
    def test_delete_comment_returns_success_true(self, client, mock_supabase):
        """DELETE /boards/comments/{comment_id} returns success=True."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_COMMENT])

        response = client.delete(f"/boards/comments/{COMMENT_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_comment_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/comments/{comment_id} returns 404 when comment absent."""
        mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

        response = client.delete(f"/boards/comments/{COMMENT_ID}")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Subscription gate tests
# ---------------------------------------------------------------------------


class TestTaskCreateGated:
    """POST /boards/tasks with a Free user at cap returns 402."""

    def test_create_task_at_cap_full_count_path(self, client, mock_supabase):
        """Full count-path test for task gate.

        Catches count-query regressions (wrong table/filter/etc.) that the monkeypatch
        test would miss. Free tier max_tasks=50; user has 50 tasks → 402.
        """
        FREE_TIER = {
            "tier": "free",
            "max_artists": 3,
            "max_projects": 3,
            "max_boards": -1,
            "max_tasks": 50,
            "max_storage_bytes": 1073741824,
            "max_split_sheets_per_month": 5,
            "max_oneclick_runs_per_month": 1,
            "zoe_enabled": False,
            "oneclick_enabled": True,
            "registry_enabled": False,
            "integrations_allowed": ["google_drive"],
            "updated_at": "2026-05-09T00:00:00+00:00",
        }
        FREE_SUB = {
            "id": "s1",
            "user_id": TEST_USER_ID,
            "tier": "free",
            "status": "active",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "stripe_price_id": None,
            "current_period_start": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "canceled_at": None,
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
        }
        ZERO_USAGE = {
            "user_id": TEST_USER_ID,
            "total_storage_bytes": 0,
            "split_sheets_this_period": 0,
            "zoe_queries_this_period": 0,
            "oneclick_runs_this_period": 0,
            "period_start": "2026-05-09T00:00:00+00:00",
            "period_end": "2099-05-09T00:00:00+00:00",
            "updated_at": "2026-05-09T00:00:00+00:00",
        }

        # Reset the EntitlementsService singleton so it is rebuilt with free-tier data
        import subscriptions.deps as _sub_deps

        _sub_deps._entitlements_service = None

        def _table(name):
            _b = _authz_board_builder(name)
            if _b is not None:
                return _b
            b = BoardMockBuilder([])
            if name == "board_tasks":
                b._data = [{"id": f"t{i}"} for i in range(50)]
                b._count = 50
            elif name == "subscriptions":
                b._data = [FREE_SUB]
                b._count = 1
            elif name == "tier_entitlements":
                b._data = [FREE_TIER]
                b._count = 1
            elif name == "tier_overrides":
                b._data = []
                b._count = 0
            elif name == "usage_counters":
                b._data = [ZERO_USAGE]
                b._count = 1
            return b

        mock_supabase.table.side_effect = _table

        resp = client.post("/boards/tasks", json={"title": "Task 51", "column_id": "c1"})
        assert resp.status_code == 402
        assert "task" in resp.json()["detail"].lower()

    def test_create_task_at_cap_returns_402(self, client, mock_supabase, monkeypatch):
        """Free user at 50/50 tasks → POST /boards/tasks returns 402."""
        from unittest.mock import MagicMock

        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        # Patch enforcement._service to return a service that denies CREATE_TASK
        deny_result = CheckResult(
            allowed=False,
            reason="You've reached your limit of 50 tasks.",
            upgrade_required=True,
        )
        svc = MagicMock()
        svc.can.return_value = deny_result
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        # board_tasks table: count=50 at cap
        def _table(name):
            if name == "board_tasks":
                b = BoardMockBuilder(data=[{"id": f"t{i}"} for i in range(50)], count=50)
                return b
            return _authz_board_builder(name) or _builder([])

        mock_supabase.table.side_effect = _table

        resp = client.post(
            "/boards/tasks",
            json={"title": "Task 51"},
        )
        assert resp.status_code == 402
        assert "task" in resp.json()["detail"].lower()
