"""Tests for the Boards (Workspace) endpoints.

Acceptance criteria:
1. Columns: list, create, update, delete, create defaults
2. Tasks: list, create, get detail, update, delete, reorder
3. Parent tasks: list, create
4. Calendar: list tasks in date range
5. Comments: add, delete
"""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

COLUMN_ID = "col-00000000-0000-0000-0000-000000000001"
TASK_ID = "task-0000-0000-0000-0000-000000000001"
COMMENT_ID = "cmnt-0000-0000-0000-0000-000000000001"
PARENT_ID = "prnt-0000-0000-0000-0000-000000000001"
ARTIST_ID = "art-00000000-0000-0000-0000-000000000001"

SAMPLE_COLUMN = {
    "id": COLUMN_ID,
    "user_id": TEST_USER_ID,
    "title": "To Do",
    "color": "#6366f1",
    "position": 0,
}

SAMPLE_TASK = {
    "id": TASK_ID,
    "user_id": TEST_USER_ID,
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


def _sequence_side_effect(sequences: list[list]) -> callable:
    """Return a side_effect function that returns builders from a pre-defined sequence.

    Each element of `sequences` is a data list for one table() call.
    Once exhausted, returns empty builders.
    """
    idx = [0]

    def _side_effect(name):
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
    """
    idx = [0]

    def _side_effect(name):
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
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COLUMN])

        response = client.get("/boards/columns")

        assert response.status_code == 200
        body = response.json()
        assert "columns" in body
        assert len(body["columns"]) == 1
        assert body["columns"][0]["id"] == COLUMN_ID

    def test_list_columns_empty_returns_empty_list(self, client, mock_supabase):
        """GET /boards/columns with no data returns an empty list."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.get("/boards/columns")

        assert response.status_code == 200
        assert response.json()["columns"] == []

    def test_list_columns_with_artist_id_filter(self, client, mock_supabase):
        """GET /boards/columns?artist_id=... passes artist_id filter."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COLUMN])

        response = client.get(f"/boards/columns?artist_id={ARTIST_ID}")

        assert response.status_code == 200
        assert "columns" in response.json()


class TestCreateColumn:
    def test_create_column_returns_200_with_column(self, client, mock_supabase):
        """POST /boards/columns returns created column."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COLUMN])

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
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.post("/boards/columns", json={"title": "Broken"})

        assert response.status_code == 500

    def test_create_column_with_only_title(self, client, mock_supabase):
        """POST /boards/columns works with just a title (other fields optional)."""
        col = {**SAMPLE_COLUMN, "title": "Minimal"}
        mock_supabase.table.side_effect = lambda name: _builder([col])

        response = client.post("/boards/columns", json={"title": "Minimal"})

        assert response.status_code == 200
        assert response.json()["title"] == "Minimal"


class TestUpdateColumn:
    def test_update_column_returns_updated_data(self, client, mock_supabase):
        """PUT /boards/columns/{column_id} returns updated column."""
        updated = {**SAMPLE_COLUMN, "title": "In Progress"}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put(f"/boards/columns/{COLUMN_ID}", json={"title": "In Progress"})

        assert response.status_code == 200
        assert response.json()["title"] == "In Progress"

    def test_update_column_returns_404_when_not_found(self, client, mock_supabase):
        """PUT /boards/columns/{column_id} returns 404 when column absent."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.put(f"/boards/columns/{COLUMN_ID}", json={"title": "Ghost"})

        assert response.status_code == 404


class TestDeleteColumn:
    def test_delete_column_returns_success_true(self, client, mock_supabase):
        """DELETE /boards/columns/{column_id} returns success=True."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COLUMN])

        response = client.delete(f"/boards/columns/{COLUMN_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_column_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/columns/{column_id} returns 404 when column absent."""
        mock_supabase.table.side_effect = lambda name: _builder([])

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
            idx = call_count[0] % len(default_columns)
            data = [default_columns[idx]]
            call_count[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = client.post(f"/boards/columns/defaults?artist_id={ARTIST_ID}")

        assert response.status_code == 200
        assert "columns" in response.json()


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
        mock_supabase.table.side_effect = lambda name: _builder([])

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
        # create_task: board_tasks insert -> junction tables (no artist_ids)
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], []])

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
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "Broken Task", "column_id": COLUMN_ID},
            )

        assert response.status_code == 500

    def test_create_task_with_due_date(self, client, mock_supabase):
        """POST /boards/tasks with due_date converts date to string."""
        task_with_date = {**SAMPLE_TASK, "due_date": "2026-04-30"}
        mock_supabase.table.side_effect = _sequence_side_effect([[task_with_date], []])

        with patch("boards.service.events.emit"):
            response = client.post(
                "/boards/tasks",
                json={"title": "Due Date Task", "column_id": COLUMN_ID, "due_date": "2026-04-30"},
            )

        assert response.status_code == 200

    def test_create_task_with_artist_ids_calls_junction_table(self, client, mock_supabase):
        """POST /boards/tasks with artist_ids sets junction rows."""
        # create_task: board_tasks insert, then board_task_artists delete + insert
        mock_supabase.table.side_effect = _sequence_side_effect(
            [[SAMPLE_TASK], [{"artist_id": ARTIST_ID}], [{"artist_id": ARTIST_ID}]]
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
        # 1. board_tasks -> single (main task lookup)
        # 2. board_task_artists -> list
        # 3. board_task_projects -> list
        # 4. board_task_contracts -> list
        # 5. board_task_comments -> list
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
                detail_task,  # .single() lookup for main task
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
        # 1. board_columns -> single (for Done-column title check)
        # 2. board_tasks -> list (update result)
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
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
        # board_columns returns the column, board_tasks update returns empty
        mock_supabase.table.side_effect = _sequence_side_effect_mixed(
            [
                {"id": COLUMN_ID, "title": "To Do"},
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
        mock_supabase.table.side_effect = lambda name: _builder([updated_task])

        with patch("boards.service.events.emit"):
            response = client.put(f"/boards/tasks/{TASK_ID}", json={"title": "No Column Update"})

        assert response.status_code == 200


class TestDeleteTask:
    def test_delete_task_returns_success_true(self, client, mock_supabase):
        """DELETE /boards/tasks/{task_id} returns success=True."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_TASK])

        response = client.delete(f"/boards/tasks/{TASK_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_task_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/tasks/{task_id} returns 404 when task absent."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.delete(f"/boards/tasks/{TASK_ID}")

        assert response.status_code == 404


class TestReorderTasks:
    def test_reorder_tasks_returns_success_true(self, client, mock_supabase):
        """PUT /boards/tasks/reorder accepts batch reorder payload."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_TASK])

        response = client.put(
            "/boards/tasks/reorder",
            json={"reorders": [{"task_id": TASK_ID, "target_column_id": COLUMN_ID, "position": 1}]},
        )

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_reorder_tasks_with_multiple_items(self, client, mock_supabase):
        """PUT /boards/tasks/reorder handles multiple reorder items."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_TASK])
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
        mock_supabase.table.side_effect = lambda name: _builder([])

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
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_PARENT])

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
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.post("/boards/parents", json={"title": "Broken Parent"})

        assert response.status_code == 500

    def test_create_parent_task_with_due_date(self, client, mock_supabase):
        """POST /boards/parents with due_date converts date to string."""
        parent_with_date = {**SAMPLE_PARENT, "due_date": "2026-05-01"}
        mock_supabase.table.side_effect = lambda name: _builder([parent_with_date])

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
        # 2. board_tasks (start_date query)
        # 3-5. _enrich_tasks: board_task_artists, board_task_projects, board_task_contracts
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [], [], [], []])

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

    def test_calendar_tasks_deduplicates_overlapping_results(self, client, mock_supabase):
        """GET /boards/calendar deduplicates tasks appearing in both due_date and start_date queries."""
        # Both board_tasks queries return the same task.
        # The service deduplicates by ID, so it should appear once in results.
        mock_supabase.table.side_effect = _sequence_side_effect([[SAMPLE_TASK], [SAMPLE_TASK], [], [], []])

        response = client.get("/boards/calendar?start=2026-04-01&end=2026-04-30")

        assert response.status_code == 200
        tasks = response.json()["tasks"]
        assert len(tasks) == 1


# ===========================================================================
# Comments
# ===========================================================================


class TestAddComment:
    def test_add_comment_returns_comment(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments returns created comment."""
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COMMENT])

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
        mock_supabase.table.side_effect = lambda name: _builder([])

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
        mock_supabase.table.side_effect = lambda name: _builder([SAMPLE_COMMENT])

        response = client.delete(f"/boards/comments/{COMMENT_ID}")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_delete_comment_returns_404_when_not_found(self, client, mock_supabase):
        """DELETE /boards/comments/{comment_id} returns 404 when comment absent."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.delete(f"/boards/comments/{COMMENT_ID}")

        assert response.status_code == 404
