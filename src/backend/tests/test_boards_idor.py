"""IDOR tests for boards: comment ownership and junction id validation."""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VICTIM_TASK_ID = "victim-task-0000-0000-0000-000000000001"
OWN_TASK_ID = "own-task-000-0000-0000-0000-000000000001"
OWN_ARTIST_ID = "own-artist-00-0000-0000-0000-000000000001"
VICTIM_ARTIST_ID = "victim-artist-0000-0000-0000-000000000001"
OWN_PROJECT_ID = "own-proj-0000-0000-0000-0000-000000000001"
VICTIM_PROJECT_ID = "victim-proj-0000-0000-0000-000000000001"
OWN_CONTRACT_ID = "own-contract-0000-0000-0000-000000000001"
VICTIM_CONTRACT_ID = "victim-contract-0000-0000-000000000001"
COLUMN_ID = "col-00000000-0000-0000-0000-000000000001"
BOARD_ID = "b0000000-0000-0000-0000-000000000001"

# A different user's board — used to prove the board-access gate (not row ownership)
# is what protects reads/writes on another user's task under team boards.
OTHER_USER_ID = "00000000-0000-0000-0000-000000000099"
VICTIM_BOARD_ID = "b0000000-0000-0000-0000-000000000099"

SAMPLE_TASK = {
    "id": OWN_TASK_ID,
    "user_id": TEST_USER_ID,
    "column_id": COLUMN_ID,
    "title": "My Task",
    "priority": "medium",
    "start_date": "2026-04-01",
    "due_date": "2026-04-30",
    "position": 0,
    "is_parent": False,
    "parent_task_id": None,
}

_PRO_TIER_ROW = {
    "tier": "pro",
    "max_artists": -1,
    "max_projects": -1,
    "max_boards": -1,
    "max_tasks": -1,
    "max_storage_bytes": -1,
    "max_split_sheets_per_month": -1,
    "max_oneclick_runs_per_month": -1,
    "zoe_enabled": True,
    "oneclick_enabled": True,
    "registry_enabled": True,
    "integrations_allowed": ["google_drive", "slack", "notion"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
_PRO_SUB_ROW = {
    "id": "s-default",
    "user_id": TEST_USER_ID,
    "tier": "pro",
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
_DEFAULT_USAGE_ROW = {
    "user_id": TEST_USER_ID,
    "total_storage_bytes": 0,
    "split_sheets_this_period": 0,
    "zoe_queries_this_period": 0,
    "oneclick_runs_this_period": 0,
    "period_start": "2026-05-09T00:00:00+00:00",
    "period_end": "2099-05-09T00:00:00+00:00",
    "updated_at": "2026-05-09T00:00:00+00:00",
}
_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_builder(name):
    b = MockQueryBuilder()
    if name == "subscriptions":
        b.execute.return_value = MagicMock(data=[_PRO_SUB_ROW], count=1)
    elif name == "tier_entitlements":
        b.execute.return_value = MagicMock(data=[_PRO_TIER_ROW], count=1)
    elif name == "tier_overrides":
        b.execute.return_value = MagicMock(data=[], count=0)
    elif name == "usage_counters":
        b.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
    elif name == "profiles":
        b.execute.return_value = MagicMock(data=[], count=0)
    return b


def _make_builder(data):
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data)
    return b


# ===========================================================================
# ITEM A — Comment on task the caller doesn't own → 404
# ===========================================================================


class TestCommentOnUnownedTask:
    def test_comment_on_unowned_task_returns_404(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments on a task not owned by caller → 404."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                # ownership check returns no row (task not owned by this user)
                b.execute.return_value = MagicMock(data=None)
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.post(
            f"/boards/tasks/{VICTIM_TASK_ID}/comments",
            json={"content": "hi"},
        )
        assert resp.status_code == 404

    def test_comment_on_owned_task_succeeds(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments on a task the caller owns → 200."""
        sample_comment = {
            "id": "cmnt-0001",
            "task_id": OWN_TASK_ID,
            "user_id": TEST_USER_ID,
            "content": "valid comment",
        }
        own_task_with_board = {**SAMPLE_TASK, "board_id": BOARD_ID}

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                # _task_board_id: resolves the task's board
                b.execute.return_value = MagicMock(data=[own_task_with_board])
            elif name == "boards":
                # require_board_edit: the resolved board is owned by the caller
                b.execute.return_value = MagicMock(
                    data=[{"id": BOARD_ID, "team_id": None, "owner_id": TEST_USER_ID, "archived": False}]
                )
            else:
                # the insert into board_task_comments
                b.execute.return_value = MagicMock(data=[sample_comment])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.post(
            f"/boards/tasks/{OWN_TASK_ID}/comments",
            json={"content": "valid comment"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "valid comment"


# ===========================================================================
# ITEM A2 — Board-access isolation: victim task exists but lives on a board
# owned by a DIFFERENT user. Under the board_id-scoping model, ownership of
# the *row* no longer matters — every read/write gates on require_board_access
# /require_board_edit for the task's resolved board_id. These assert the
# security property still holds (404, not a data leak or a write-through).
# ===========================================================================


class TestBoardAccessIsolation:
    def test_get_task_detail_on_foreign_board_returns_404(self, client, mock_supabase):
        """GET /boards/tasks/{task_id}/detail for a task on another user's board → 404."""
        victim_task = {
            "id": VICTIM_TASK_ID,
            "board_id": VICTIM_BOARD_ID,
            "is_parent": False,
            "parent_task_id": None,
        }

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                # get_task_detail fetches the task by id (no user_id filter) ...
                b.execute.return_value = MagicMock(data=[victim_task])
            elif name == "boards":
                # ... then gates on the resolved board, owned by someone else.
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.get(f"/boards/tasks/{VICTIM_TASK_ID}/detail")
        assert resp.status_code == 404

    def test_update_task_on_foreign_board_returns_404(self, client, mock_supabase):
        """PUT /boards/tasks/{task_id} for a task on another user's board → 404, no write."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                # Router pre-read and service task_row fetch both hit board_tasks; either
                # shape resolves the same victim board, so a single response for both is fine.
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_TASK_ID, "board_id": VICTIM_BOARD_ID, "parent_task_id": None}]
                )
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.put(
            f"/boards/tasks/{VICTIM_TASK_ID}",
            json={"title": "Hijacked"},
        )
        assert resp.status_code == 404

    def test_delete_task_on_foreign_board_returns_404(self, client, mock_supabase):
        """DELETE /boards/tasks/{task_id} for a task on another user's board → 404, no delete."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                b.execute.return_value = MagicMock(data=[{"board_id": VICTIM_BOARD_ID}])
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.delete(f"/boards/tasks/{VICTIM_TASK_ID}")
        assert resp.status_code == 404

    def test_comment_on_foreign_board_task_returns_404(self, client, mock_supabase):
        """POST /boards/tasks/{task_id}/comments for a task on another user's board → 404."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_tasks":
                b.execute.return_value = MagicMock(data=[{"board_id": VICTIM_BOARD_ID}])
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.post(
            f"/boards/tasks/{VICTIM_TASK_ID}/comments",
            json={"content": "sneaky"},
        )
        assert resp.status_code == 404

    def test_update_column_on_foreign_board_returns_404(self, client, mock_supabase):
        """PUT /boards/columns/{column_id} for a column on another user's board → 404, no write."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_columns":
                b.execute.return_value = MagicMock(data=[{"board_id": VICTIM_BOARD_ID}])
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.put(f"/boards/columns/{COLUMN_ID}", json={"title": "Hijacked"})
        assert resp.status_code == 404

    def test_delete_column_on_foreign_board_returns_404(self, client, mock_supabase):
        """DELETE /boards/columns/{column_id} for a column on another user's board → 404, no delete."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "board_columns":
                b.execute.return_value = MagicMock(data=[{"board_id": VICTIM_BOARD_ID}])
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.delete(f"/boards/columns/{COLUMN_ID}")
        assert resp.status_code == 404

    def test_get_columns_with_foreign_board_id_returns_404(self, client, mock_supabase):
        """GET /boards/columns?board_id=<foreign> → 404, does not leak another user's columns."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": VICTIM_BOARD_ID, "team_id": None, "owner_id": OTHER_USER_ID, "archived": False}]
                )
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        resp = client.get(f"/boards/columns?board_id={VICTIM_BOARD_ID}")
        assert resp.status_code == 404


# ===========================================================================
# ITEM B — Junction id validation in create_task / update_task
# ===========================================================================


class TestCreateTaskJunctionFiltering:
    def test_foreign_artist_id_is_silently_dropped(self, client, mock_supabase):
        """POST /boards/tasks with a victim artist_id: the junction must not link it."""

        # Tracks which artist IDs were actually passed to _merge_junction for board_task_artists
        linked_artists = []

        # Counted per-table-name (not globally) so it's robust to the board-resolution
        # calls (board_columns / boards) that create_task now makes before the insert.
        board_tasks_calls = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()

            if name == "board_tasks":
                idx = board_tasks_calls[0]
                board_tasks_calls[0] += 1
                if idx == 0:
                    # count gate
                    b.execute.return_value = MagicMock(data=[], count=0)
                else:
                    # insert
                    b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
            elif name == "board_columns":
                # _column_board_id: COLUMN_ID resolves to the caller's own board
                b.execute.return_value = MagicMock(data=[{"board_id": BOARD_ID}])
            elif name == "boards":
                # require_board_edit: the resolved board is owned by the caller
                b.execute.return_value = MagicMock(
                    data=[{"id": BOARD_ID, "team_id": None, "owner_id": TEST_USER_ID, "archived": False}]
                )
            elif name == "artists":
                # _owned_artist_ids: caller owns OWN_ARTIST_ID, not VICTIM_ARTIST_ID
                b.execute.return_value = MagicMock(data=[{"id": OWN_ARTIST_ID}])
            elif name == "board_task_artists":
                # capture what rows get inserted
                def capturing_insert(rows, *args, **kwargs):
                    for r in rows:
                        linked_artists.append(r.get("artist_id"))
                    return b

                b.insert = capturing_insert
                b.execute.return_value = MagicMock(data=[])
                # delete side (called first by _merge_junction)
                b.delete.return_value = MagicMock()
                b.delete.return_value.eq = MagicMock(return_value=MagicMock(execute=MagicMock(return_value=None)))
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        with patch("boards.service.events.emit"):
            resp = client.post(
                "/boards/tasks",
                json={
                    "title": "Task",
                    "column_id": COLUMN_ID,
                    "artist_ids": [OWN_ARTIST_ID, VICTIM_ARTIST_ID],
                },
            )

        assert resp.status_code == 200
        # VICTIM_ARTIST_ID must NOT have been linked
        assert VICTIM_ARTIST_ID not in linked_artists

    def test_own_artist_id_is_preserved(self, client, mock_supabase):
        """POST /boards/tasks with owned artist_id: the junction links the owned id."""
        linked_artists = []

        # Counted per-table-name (not globally) so it's robust to the board-resolution
        # calls (board_columns / boards) that create_task now makes before the insert.
        board_tasks_calls = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()

            if name == "board_tasks":
                idx = board_tasks_calls[0]
                board_tasks_calls[0] += 1
                if idx == 0:
                    b.execute.return_value = MagicMock(data=[], count=0)
                else:
                    b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
            elif name == "board_columns":
                b.execute.return_value = MagicMock(data=[{"board_id": BOARD_ID}])
            elif name == "boards":
                b.execute.return_value = MagicMock(
                    data=[{"id": BOARD_ID, "team_id": None, "owner_id": TEST_USER_ID, "archived": False}]
                )
            elif name == "artists":
                b.execute.return_value = MagicMock(data=[{"id": OWN_ARTIST_ID}])
            elif name == "board_task_artists":

                def capturing_insert(rows, *args, **kwargs):
                    for r in rows:
                        linked_artists.append(r.get("artist_id"))
                    return b

                b.insert = capturing_insert
                b.execute.return_value = MagicMock(data=[])
                b.delete.return_value = MagicMock()
                b.delete.return_value.eq = MagicMock(return_value=MagicMock(execute=MagicMock(return_value=None)))
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        with patch("boards.service.events.emit"):
            resp = client.post(
                "/boards/tasks",
                json={
                    "title": "Task",
                    "column_id": COLUMN_ID,
                    "artist_ids": [OWN_ARTIST_ID],
                },
            )

        assert resp.status_code == 200
        assert OWN_ARTIST_ID in linked_artists


class TestUpdateTaskJunctionFiltering:
    def test_foreign_artist_id_is_dropped_on_update(self, client, mock_supabase):
        """PUT /boards/tasks/{task_id} with a victim artist_id: the junction must not link it."""
        linked_artists = []

        # Counted per-table-name (not globally) so it's robust to the board-resolution calls
        # (service task_row fetch, require_board_edit's "boards" lookup, and _column_board_id's
        # §7.3 dst-board check) that update_task now makes before the actual update.
        board_tasks_calls = [0]
        board_columns_calls = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()

            if name == "board_tasks":
                idx = board_tasks_calls[0]
                board_tasks_calls[0] += 1
                if idx == 0:
                    # router pre-read of existing column_id
                    b.execute.return_value = MagicMock(data={"column_id": COLUMN_ID})
                elif idx == 1:
                    # service task_row fetch (id, board_id, parent_task_id) — gates require_board_edit
                    b.execute.return_value = MagicMock(
                        data=[{"id": OWN_TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}]
                    )
                else:
                    # update result
                    b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
            elif name == "boards":
                # require_board_edit: the resolved board is owned by the caller
                b.execute.return_value = MagicMock(
                    data=[{"id": BOARD_ID, "team_id": None, "owner_id": TEST_USER_ID, "archived": False}]
                )
            elif name == "board_columns":
                idx = board_columns_calls[0]
                board_columns_calls[0] += 1
                if idx == 0:
                    # service _column_board_id (§7.3 dst-board resolution; same board, no move)
                    b.execute.return_value = MagicMock(data=[{"board_id": BOARD_ID}])
                else:
                    # done-check lookup
                    b.execute.return_value = MagicMock(data={"id": COLUMN_ID, "title": "In Progress"})
            elif name == "artists":
                # caller only owns OWN_ARTIST_ID
                b.execute.return_value = MagicMock(data=[{"id": OWN_ARTIST_ID}])
            elif name == "board_task_artists":

                def capturing_insert(rows, *args, **kwargs):
                    for r in rows:
                        linked_artists.append(r.get("artist_id"))
                    return b

                b.insert = capturing_insert
                b.execute.return_value = MagicMock(data=[])
                b.delete.return_value = MagicMock()
                b.delete.return_value.eq = MagicMock(return_value=MagicMock(execute=MagicMock(return_value=None)))
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        with patch("boards.service.events.emit"):
            resp = client.put(
                f"/boards/tasks/{OWN_TASK_ID}",
                json={
                    "title": "Updated",
                    "column_id": COLUMN_ID,
                    "artist_ids": [OWN_ARTIST_ID, VICTIM_ARTIST_ID],
                },
            )

        assert resp.status_code == 200
        assert VICTIM_ARTIST_ID not in linked_artists
