"""Tests for Boards / Calendar event instrumentation.

Verifies that the boards router fires the right PostHog events:
- board_created on POST /boards/parents
- task_created on POST /boards/tasks
- task_status_changed on PUT /boards/tasks/{id} when column changes
- task_completed on PUT /boards/tasks/{id} when moved into a column titled "done"
"""

from unittest.mock import patch

from tests.test_boards import (
    BOARD_ID,
    COLUMN_ID,
    SAMPLE_COLUMN,
    SAMPLE_PARENT,
    SAMPLE_TASK,
    TASK_ID,
    _authz_board_builder,
    _builder,
    _pro_sub_builder,
    _sequence_side_effect_mixed,
)

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})

OLD_COLUMN_ID = "col-old-0000-0000-0000-000000000001"
NEW_COLUMN_ID = "col-new-0000-0000-0000-000000000001"
DONE_COLUMN_ID = "col-done-0000-0000-0000-000000000001"


def _capture():
    """Returns (sink, fake_capture) — sink collects (event, props) tuples."""
    sink: list[tuple[str, dict]] = []

    def _fake(uid, event, props=None):
        sink.append((event, dict(props or {})))

    return sink, _fake


# ===========================================================================
# board_created on POST /boards/parents
# ===========================================================================


def test_create_parent_board_fires_board_created(client, mock_supabase):
    """POST /boards/parents fires board_created with tool=boards on success."""
    mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([SAMPLE_PARENT])

    sink, fake = _capture()
    with patch("boards.router.analytics_capture", side_effect=fake):
        resp = client.post("/boards/parents", json={"title": "Q3 Album Rollout"})

    assert resp.status_code == 200, resp.text
    created = [(e, p) for e, p in sink if e == "board_created"]
    assert len(created) == 1, f"expected board_created, got {sink}"
    assert created[0][1]["tool"] == "boards"


def test_create_parent_board_no_event_on_failure(client, mock_supabase):
    """POST /boards/parents does NOT fire board_created on insert failure (500)."""
    mock_supabase.table.side_effect = lambda name: _authz_board_builder(name) or _builder([])

    sink, fake = _capture()
    with patch("boards.router.analytics_capture", side_effect=fake):
        resp = client.post("/boards/parents", json={"title": "Broken"})

    assert resp.status_code == 500
    assert not [e for e, _ in sink if e == "board_created"], f"unexpected board_created in {sink}"


# ===========================================================================
# task_created on POST /boards/tasks
# ===========================================================================


def test_create_task_fires_task_created(client, mock_supabase):
    """POST /boards/tasks fires task_created with tool=boards, source=manual."""
    # Sequence: 1) board_tasks count (gate) → 0, 2) board_columns (_column_board_id), 3) board_tasks insert → row
    seq = [[], [SAMPLE_COLUMN], [SAMPLE_TASK]]
    idx = [0]

    def _stateful(name):
        if name in _SUBSCRIPTION_TABLES:
            return _pro_sub_builder(name)
        _b = _authz_board_builder(name)
        if _b is not None:
            return _b
        data = seq[idx[0]] if idx[0] < len(seq) else []
        idx[0] += 1
        return _builder(data)

    mock_supabase.table.side_effect = _stateful

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.post(
            "/boards/tasks",
            json={"title": "Mix the bridge", "column_id": COLUMN_ID, "priority": "medium"},
        )

    assert resp.status_code == 200, resp.text
    created = [(e, p) for e, p in sink if e == "task_created"]
    assert len(created) == 1, f"expected task_created, got {sink}"
    assert created[0][1]["tool"] == "boards"
    assert created[0][1]["source"] == "manual"


def test_create_task_no_event_on_failure(client, mock_supabase):
    """No task_created event if the insert fails (500)."""
    # gate count → 0, board_columns (_column_board_id) → row, insert → empty → 500
    seq = [[], [SAMPLE_COLUMN], []]
    idx = [0]

    def _stateful(name):
        if name in _SUBSCRIPTION_TABLES:
            return _pro_sub_builder(name)
        _b = _authz_board_builder(name)
        if _b is not None:
            return _b
        data = seq[idx[0]] if idx[0] < len(seq) else []
        idx[0] += 1
        return _builder(data)

    mock_supabase.table.side_effect = _stateful

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.post(
            "/boards/tasks",
            json={"title": "Broken Task", "column_id": COLUMN_ID},
        )

    assert resp.status_code == 500
    assert not [e for e, _ in sink if e == "task_created"], f"unexpected task_created: {sink}"


# ===========================================================================
# task_status_changed + task_completed on PUT /boards/tasks/{task_id}
# ===========================================================================


def test_update_task_fires_status_changed_when_column_changes(client, mock_supabase):
    """PUT /boards/tasks/{id} with a different column_id fires task_status_changed
    (but NOT task_completed when the new column title isn't 'done')."""
    updated_task = {**SAMPLE_TASK, "column_id": NEW_COLUMN_ID}

    # Router order:
    #   1. board_tasks .select("column_id").eq("id").single()  -> old column lookup
    # Then service.update_task runs:
    #   2. board_tasks   .select("id, board_id, parent_task_id").limit(1)  -> task_row (gates require_board_edit)
    #   3. board_columns .select("board_id").limit(1)          -> _column_board_id (§7.3; same board, no move)
    #   4. board_columns .select("title").eq("id").single()    -> "In Progress"
    #   5. board_tasks   .update().eq().execute()               -> [updated_task]
    # Then router post-update:
    #   6. board_columns .select("title").eq("id").single()    -> "In Progress"
    mock_supabase.table.side_effect = _sequence_side_effect_mixed(
        [
            {"column_id": OLD_COLUMN_ID},  # router pre-read of existing task
            [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
            [{"board_id": BOARD_ID}],  # service _column_board_id (dst == src, no cross-board move)
            {"id": NEW_COLUMN_ID, "title": "In Progress"},  # service done-check
            [updated_task],  # service update result
            {"id": NEW_COLUMN_ID, "title": "In Progress"},  # router post-read for done-check
        ]
    )

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.put(
            f"/boards/tasks/{TASK_ID}",
            json={"column_id": NEW_COLUMN_ID},
        )

    assert resp.status_code == 200, resp.text
    status_changed = [(e, p) for e, p in sink if e == "task_status_changed"]
    completed = [(e, p) for e, p in sink if e == "task_completed"]
    assert len(status_changed) == 1, f"expected task_status_changed, got {sink}"
    assert status_changed[0][1]["tool"] == "boards"
    assert status_changed[0][1]["from_column_id"] == OLD_COLUMN_ID
    assert status_changed[0][1]["to_column_id"] == NEW_COLUMN_ID
    assert len(completed) == 0, f"should NOT fire task_completed for non-done column, got {sink}"


def test_update_task_fires_completed_when_moved_to_done(client, mock_supabase):
    """PUT /boards/tasks/{id} into a column titled 'done' (case-insensitive)
    fires BOTH task_status_changed AND task_completed."""
    updated_task = {**SAMPLE_TASK, "column_id": DONE_COLUMN_ID}

    # Same call sequence as above but the column title is "Done".
    mock_supabase.table.side_effect = _sequence_side_effect_mixed(
        [
            {"column_id": OLD_COLUMN_ID},  # router pre-read
            [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
            [{"board_id": BOARD_ID}],  # service _column_board_id (dst == src, no cross-board move)
            {"id": DONE_COLUMN_ID, "title": "Done"},  # service done-check → matches
            [updated_task],  # service update
            {"id": DONE_COLUMN_ID, "title": "Done"},  # router post-read → matches "done"
        ]
    )

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.put(
            f"/boards/tasks/{TASK_ID}",
            json={"column_id": DONE_COLUMN_ID},
        )

    assert resp.status_code == 200, resp.text
    status_changed = [(e, p) for e, p in sink if e == "task_status_changed"]
    completed = [(e, p) for e, p in sink if e == "task_completed"]
    assert len(status_changed) == 1, f"expected task_status_changed, got {sink}"
    assert len(completed) == 1, f"expected task_completed for 'done' column, got {sink}"
    assert completed[0][1]["tool"] == "boards"


def test_update_task_no_event_when_column_unchanged(client, mock_supabase):
    """No column change in body → no task_status_changed / task_completed event."""
    updated_task = {**SAMPLE_TASK, "title": "renamed only"}

    # Router pre-read returns the same column_id as the task already has, and
    # since the request body omits column_id we expect NO status/completed events.
    # Service in this path: no "column_id" in data → no §7.3 lookup, no board_columns lookup.
    # Then: board_tasks update.
    mock_supabase.table.side_effect = _sequence_side_effect_mixed(
        [
            {"column_id": COLUMN_ID},  # router pre-read
            [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
            [updated_task],  # service update (no done-check because no column_id)
        ]
    )

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.put(
            f"/boards/tasks/{TASK_ID}",
            json={"title": "renamed only"},
        )

    assert resp.status_code == 200, resp.text
    status_changed = [e for e, _ in sink if e == "task_status_changed"]
    completed = [e for e, _ in sink if e == "task_completed"]
    assert not status_changed, f"no column change should yield no status event, got {sink}"
    assert not completed, f"no column change should yield no completed event, got {sink}"


def test_update_task_no_event_when_same_column_id_in_body(client, mock_supabase):
    """Sending column_id in body that matches the existing column_id should not fire."""
    updated_task = {**SAMPLE_TASK}

    mock_supabase.table.side_effect = _sequence_side_effect_mixed(
        [
            {"column_id": COLUMN_ID},  # router pre-read
            [{"id": TASK_ID, "board_id": BOARD_ID, "parent_task_id": None}],  # service task_row fetch
            [{"board_id": BOARD_ID}],  # service _column_board_id (dst == src, no cross-board move)
            {"id": COLUMN_ID, "title": "In Progress"},  # service done-check
            [updated_task],  # service update
            # No router post-read because column didn't change → skip
        ]
    )

    sink, fake = _capture()
    with (
        patch("boards.service.events.emit"),
        patch("boards.router.analytics_capture", side_effect=fake),
    ):
        resp = client.put(
            f"/boards/tasks/{TASK_ID}",
            json={"column_id": COLUMN_ID, "title": "same column"},
        )

    assert resp.status_code == 200, resp.text
    assert not [e for e, _ in sink if e == "task_status_changed"], f"unexpected status event: {sink}"
    assert not [e for e, _ in sink if e == "task_completed"], f"unexpected completed event: {sink}"
