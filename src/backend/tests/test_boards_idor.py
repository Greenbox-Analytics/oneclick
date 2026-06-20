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

        call_count = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0 and name == "board_tasks":
                # ownership check passes — returns the task row
                b.execute.return_value = MagicMock(data=SAMPLE_TASK)
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
# ITEM B — Junction id validation in create_task / update_task
# ===========================================================================


class TestCreateTaskJunctionFiltering:
    def test_foreign_artist_id_is_silently_dropped(self, client, mock_supabase):
        """POST /boards/tasks with a victim artist_id: the junction must not link it."""

        # Tracks which artist IDs were actually passed to _set_junction for board_task_artists
        linked_artists = []

        call_count = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            idx = call_count[0]
            call_count[0] += 1

            if name == "board_tasks" and idx == 0:
                # count gate
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "board_tasks" and idx == 1:
                # insert
                b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
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
                # delete side (called first by _set_junction)
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

        call_count = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            idx = call_count[0]
            call_count[0] += 1

            if name == "board_tasks" and idx == 0:
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "board_tasks" and idx == 1:
                b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
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

        call_count = [0]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            idx = call_count[0]
            call_count[0] += 1

            if name == "board_tasks" and idx == 0:
                # router pre-read of existing column_id
                b.execute.return_value = MagicMock(data={"column_id": COLUMN_ID})
            elif name == "board_columns" and idx == 1:
                # done-check lookup
                b.execute.return_value = MagicMock(data={"id": COLUMN_ID, "title": "In Progress"})
            elif name == "board_tasks" and idx == 2:
                # update result
                b.execute.return_value = MagicMock(data=[SAMPLE_TASK])
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
