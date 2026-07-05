"""Business logic for the Kanban board feature."""

from collections.abc import Callable
from datetime import UTC, date, datetime

from supabase import Client

from confirm import ConfirmationError, normalize_name
from integrations import events
from pagination import PaginatedResponse, paginate_query
from teams import authz

# --- Junction table helpers ---


def _merge_junction(
    supabase: Client,
    task_id: str,
    table: str,
    fk_column: str,
    submitted_ids: list[str],
    labels: dict[str, str] | None,
    access_fn,
    user_id: str,
) -> list[str]:
    """Non-destructive link write (spec §3.1). Replaces ONLY the rows the editor can access,
    preserving co-owners' rows (and their labels) untouched. Caller must skip this entirely
    when the link field was absent from the payload (absent = no-op).
    RETURNS the editor's accessible inserted ids — the caller sets these on the returned task
    dict before emitting the event (Slack routing reads task['project_ids']; see below)."""
    existing = supabase.table(table).select(fk_column).eq("task_id", task_id).execute().data or []
    existing_ids = [r[fk_column] for r in existing]
    # of the EXISTING rows, which may the editor replace?  (their accessible partition)
    editor_existing = access_fn(supabase, user_id, existing_ids)
    # the editor's new accessible set from the submitted selection
    new_accessible = access_fn(supabase, user_id, submitted_ids)
    if editor_existing:
        supabase.table(table).delete().eq("task_id", task_id).in_(fk_column, editor_existing).execute()
    rows = [{"task_id": task_id, fk_column: fid, "label": (labels or {}).get(fid)} for fid in new_accessible]
    if rows:
        supabase.table(table).insert(rows).execute()
    return new_accessible


def _owned_artist_ids(db: Client, user_id: str, ids: list[str]) -> list[str]:
    """Return only the ids from `ids` that belong to the calling user."""
    if not ids:
        return []
    rows = db.table("artists").select("id").in_("id", ids).eq("user_id", user_id).execute()
    return [r["id"] for r in (rows.data or [])]


def _accessible_project_ids(db: Client, user_id: str, ids: list[str]) -> list[str]:
    """Return only project ids the calling user can access (owns or is a member of)."""
    if not ids:
        return []
    allowed: set[str] = set()

    # Projects belonging to the user's own artists
    art = db.table("artists").select("id").eq("user_id", user_id).execute()
    my_artist_ids = [a["id"] for a in (art.data or [])]
    if my_artist_ids:
        owned = db.table("projects").select("id").in_("id", ids).in_("artist_id", my_artist_ids).execute()
        allowed.update(p["id"] for p in (owned.data or []))

    # Projects the user is a member of
    member = db.table("project_members").select("project_id").in_("project_id", ids).eq("user_id", user_id).execute()
    allowed.update(m["project_id"] for m in (member.data or []))

    return [i for i in ids if i in allowed]


def _accessible_contract_ids(db: Client, user_id: str, ids: list[str]) -> list[str]:
    """Return only contract (project_files) ids the calling user can access."""
    if not ids:
        return []
    rows = db.table("project_files").select("id, project_id").in_("id", ids).execute()
    ok = set(_accessible_project_ids(db, user_id, [r["project_id"] for r in (rows.data or [])]))
    return [r["id"] for r in (rows.data or []) if r["project_id"] in ok]


def _get_junction_ids(supabase: Client, table: str, fk_column: str, task_ids: list[str]) -> dict:
    """Batch-fetch junction rows for multiple tasks. Returns {task_id: [fk_ids]}."""
    if not task_ids:
        return {}
    result = supabase.table(table).select(f"task_id, {fk_column}").in_("task_id", task_ids).execute()
    mapping = {}
    for row in result.data or []:
        tid = row["task_id"]
        if tid not in mapping:
            mapping[tid] = []
        mapping[tid].append(row[fk_column])
    return mapping


def _get_junction_with_labels(
    supabase: Client, table: str, fk_column: str, task_ids: list[str]
) -> dict[str, list[dict]]:
    """Batch-fetch junction rows WITH their snapshot label. Returns
    {task_id: [{"id": fk_id, "label": label_or_None}]}."""
    if not task_ids:
        return {}
    rows = supabase.table(table).select(f"task_id, {fk_column}, label").in_("task_id", task_ids).execute().data or []
    mapping: dict[str, list[dict]] = {}
    for row in rows:
        mapping.setdefault(row["task_id"], []).append({"id": row[fk_column], "label": row.get("label")})
    return mapping


def _shape_links(
    supabase: Client,
    user_id: str,
    links_by_task: dict[str, list[dict]],
    name_table: str,
    name_col: str,
    access_fn: Callable[[Client, str, list[str]], list[str]],
) -> Callable[[list[dict]], list[dict]]:
    """Build a per-task shaper that turns junction links [{id, label}] → [{id, name, can_open}]
    (spec §4). Precomputes ONCE over the union of every task's links for this kind: a single
    access_fn call (→ the can_open set) and a single name-lookup over ONLY the NULL-label ids
    (the label is the snapshot; the entity-table lookup is the fallback). Returns a closure applied
    per task, so the board-load hot path issues ~1 access + ~1 name query per kind — not per task.
    Shared by _enrich_tasks and get_task_detail so both paths emit an identical shape."""
    all_ids = list({link["id"] for links in links_by_task.values() for link in links})
    accessible = set(access_fn(supabase, user_id, all_ids)) if all_ids else set()
    need_ids = list({link["id"] for links in links_by_task.values() for link in links if link["label"] is None})
    names: dict = {}
    if need_ids:  # fallback lookup only fills the NULL-label ids
        result = supabase.table(name_table).select(f"id, {name_col}").in_("id", need_ids).execute()
        names = {row["id"]: row[name_col] for row in (result.data or [])}

    def shaper(links: list[dict]) -> list[dict]:
        return [
            {
                "id": link["id"],
                "name": link["label"] or names.get(link["id"], "Unknown"),
                "can_open": link["id"] in accessible,
            }
            for link in links
        ]

    return shaper


def _enrich_tasks(supabase: Client, tasks: list, user_id: str) -> list:
    """Enrich tasks with linked entities ([{id, name, can_open}] per kind), their raw ids,
    creator, assignees, and parent_title. `name` comes from the junction label snapshot
    (fallback to the entity-table lookup on NULL); `can_open` is stamped via the access helpers."""
    if not tasks:
        return tasks
    task_ids = [t["id"] for t in tasks]

    artist_links = _get_junction_with_labels(supabase, "board_task_artists", "artist_id", task_ids)
    project_links = _get_junction_with_labels(supabase, "board_task_projects", "project_id", task_ids)
    contract_links = _get_junction_with_labels(supabase, "board_task_contracts", "project_file_id", task_ids)

    # Batched assignees (spec §5.6): one query for all task rows, one for their profiles.
    assignee_map = _get_junction_ids(supabase, "board_task_assignees", "user_id", task_ids)
    assignee_user_ids = list({uid for uids in assignee_map.values() for uid in uids})
    assignee_profiles = {}
    if assignee_user_ids:
        presult = supabase.table("profiles").select("id, full_name, avatar_url").in_("id", assignee_user_ids).execute()
        assignee_profiles = {p["id"]: p for p in (presult.data or [])}

    # Batched creators (spec §4): one profiles query for all tasks' user_ids. Resolves even for a
    # creator who has left the team (the profile row survives the membership removal).
    creator_ids = list({t.get("user_id") for t in tasks if t.get("user_id")})
    creators = {}
    if creator_ids:
        cresult = supabase.table("profiles").select("id, full_name, avatar_url").in_("id", creator_ids).execute()
        creators = {p["id"]: p for p in (cresult.data or [])}

    # Fetch parent titles for child tasks
    parent_ids = list({t["parent_task_id"] for t in tasks if t.get("parent_task_id")})
    parent_titles = {}
    if parent_ids:
        result = supabase.table("board_tasks").select("id, title").in_("id", parent_ids).execute()
        parent_titles = {p["id"]: p["title"] for p in (result.data or [])}

    # Build each kind's shaper ONCE (one access_fn call + one name-lookup per kind over the union of
    # all tasks' links) — NOT per task — then apply the closure inside the loop (avoids N+1).
    shape_artists = _shape_links(supabase, user_id, artist_links, "artists", "name", _owned_artist_ids)
    shape_projects = _shape_links(supabase, user_id, project_links, "projects", "name", _accessible_project_ids)
    shape_documents = _shape_links(
        supabase, user_id, contract_links, "project_files", "file_name", _accessible_contract_ids
    )

    for task in tasks:
        tid = task["id"]
        task["artists"] = shape_artists(artist_links.get(tid, []))
        task["projects"] = shape_projects(project_links.get(tid, []))
        task["documents"] = shape_documents(contract_links.get(tid, []))
        # Keep the raw ids alongside the shaped lists — the board filter reads them directly.
        task["artist_ids"] = [a["id"] for a in task["artists"]]
        task["project_ids"] = [p["id"] for p in task["projects"]]
        task["contract_ids"] = [d["id"] for d in task["documents"]]
        creator = creators.get(task.get("user_id")) or {}
        task["creator"] = {
            "user_id": task.get("user_id"),
            "full_name": creator.get("full_name"),
            "avatar_url": creator.get("avatar_url"),
        }
        task["assignees"] = [
            {
                "user_id": uid,
                "full_name": assignee_profiles.get(uid, {}).get("full_name"),
                "avatar_url": assignee_profiles.get(uid, {}).get("avatar_url"),
            }
            for uid in assignee_map.get(tid, [])
        ]
        if task.get("parent_task_id"):
            task["parent_title"] = parent_titles.get(task["parent_task_id"], "")

    return tasks


# --- Board resolution ---


def ensure_personal_board(db: Client, user_id: str, artist_id: str | None = None) -> str:
    """Find (or create) the caller's PERSONAL board for artist_id (NULL = unscoped 'Personal').
    Personal boards have team_id NULL and a persisted boards.artist_id (Phase 1). on_conflict
    tolerates the create race guarded by the Task-2 partial unique index (artist boards)."""
    q = db.table("boards").select("id").eq("owner_id", user_id).is_("team_id", "null")
    q = q.eq("artist_id", artist_id) if artist_id else q.is_("artist_id", "null")
    existing = q.limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    name = "Personal"
    if artist_id:
        a = db.table("artists").select("name").eq("id", artist_id).limit(1).execute()
        if a.data:
            name = a.data[0]["name"]
    ins = db.table("boards").insert({"owner_id": user_id, "artist_id": artist_id, "name": name}).execute()
    if ins.data:
        return ins.data[0]["id"]
    # lost the race → re-read
    again = q.limit(1).execute()
    return again.data[0]["id"]


def _column_board_id(db: Client, column_id: str) -> str | None:
    r = db.table("board_columns").select("board_id").eq("id", column_id).limit(1).execute()
    return r.data[0]["board_id"] if r.data else None


def _task_board_id(db: Client, task_id: str) -> str | None:
    r = db.table("board_tasks").select("board_id").eq("id", task_id).limit(1).execute()
    return r.data[0]["board_id"] if r.data else None


def _personal_board_ids(db: Client, user_id: str) -> list[str]:
    rows = db.table("boards").select("id").eq("owner_id", user_id).is_("team_id", "null").execute().data or []
    return [r["id"] for r in rows]


def _resolve_read_board_ids(db: Client, user_id: str, artist_id: str | None, board_id: str | None) -> list[str]:
    """Reads: explicit board_id (gated) → [board_id]; artist_id → [personal board];
    neither → the caller's personal boards (backward-compat default)."""
    if board_id:
        authz.require_board_access(db, user_id, board_id)
        return [board_id]
    if artist_id:
        return [ensure_personal_board(db, user_id, artist_id)]
    return _personal_board_ids(db, user_id)


# --- Columns ---


async def get_columns(
    supabase: Client, user_id: str, artist_id: str | None = None, board_id: str | None = None
) -> list:
    """Get board columns for a user, scoped by board_id (explicit, artist alias, or personal boards)."""
    board_ids = _resolve_read_board_ids(supabase, user_id, artist_id, board_id)
    if not board_ids:
        return []
    return supabase.table("board_columns").select("*").in_("board_id", board_ids).order("position").execute().data or []


async def create_column(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a new board column."""
    board_id = data.get("board_id") or ensure_personal_board(supabase, user_id, data.get("artist_id"))
    authz.require_board_edit(supabase, user_id, board_id)  # personal owner passes; foreign board → 404
    data["board_id"] = board_id
    data["user_id"] = user_id
    result = supabase.table("board_columns").insert(data).execute()
    return result.data[0] if result.data else {}


async def update_column(supabase: Client, user_id: str, column_id: str, data: dict) -> dict:
    """Update a board column. Gated on the column's board (require_board_edit)."""
    board_id = _column_board_id(supabase, column_id)
    if not board_id:
        return {}
    authz.require_board_edit(supabase, user_id, board_id)
    clean = {k: v for k, v in data.items() if v is not None}
    result = supabase.table("board_columns").update(clean).eq("id", column_id).execute()
    return result.data[0] if result.data else {}


async def delete_column(supabase: Client, user_id: str, column_id: str) -> bool:
    """Delete a board column and all its tasks. Gated on the column's board (require_board_edit)."""
    board_id = _column_board_id(supabase, column_id)
    if not board_id:
        return False
    authz.require_board_edit(supabase, user_id, board_id)
    result = supabase.table("board_columns").delete().eq("id", column_id).execute()
    return bool(result.data)


# --- Tasks (board tasks only — excludes parent tasks) ---


async def get_tasks(
    supabase: Client,
    user_id: str,
    column_id: str | None = None,
    page: int = None,
    page_size: int = 50,
    board_id: str | None = None,
    artist_id: str | None = None,
):
    """Get board tasks (non-parent) with junction data, scoped by board_id, optionally filtered by column."""
    board_ids = _resolve_read_board_ids(supabase, user_id, artist_id, board_id)
    if not board_ids:
        return PaginatedResponse(data=[], total=0, page=page, page_size=page_size) if page else []
    query = (
        supabase.table("board_tasks")
        .select("*", count="exact")
        .in_("board_id", board_ids)
        .or_("is_parent.eq.false,is_parent.is.null")
        .order("position")
    )
    if column_id:
        query = query.eq("column_id", column_id)

    result = paginate_query(query, page, page_size)

    # Enrich with linked entities (name + can_open), creator, assignees
    if isinstance(result, list):
        return _enrich_tasks(supabase, result, user_id)
    else:
        result.data = _enrich_tasks(supabase, result.data, user_id)
        return result


async def create_task(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a new task with junction relations."""
    artist_ids = data.pop("artist_ids", [])
    project_ids = data.pop("project_ids", [])
    contract_ids = data.pop("contract_ids", [])
    artist_labels = data.pop("artist_labels", None) or {}
    project_labels = data.pop("project_labels", None) or {}
    contract_labels = data.pop("contract_labels", None) or {}

    if not data.get("start_date"):
        data["start_date"] = str(date.today())

    # Auto-assign subtasks to the first board column so they appear on the main board.
    # Board-scoped when the caller specified a board (a team-board subtask must not land in
    # one of the caller's personal columns); legacy user-scoped fallback otherwise.
    if data.get("parent_task_id") and not data.get("column_id"):
        col_q = supabase.table("board_columns").select("id")
        col_q = col_q.eq("board_id", data["board_id"]) if data.get("board_id") else col_q.eq("user_id", user_id)
        first_col = col_q.order("position").limit(1).execute()
        if first_col.data:
            data["column_id"] = first_col.data[0]["id"]

    # Resolve the authoritative board and gate on it (owned artists only for personal boards).
    owned_first_artist = None
    if artist_ids:
        owned = _owned_artist_ids(supabase, user_id, artist_ids)
        owned_first_artist = owned[0] if owned else None
    if data.get("board_id"):
        board_id = data["board_id"]
    elif data.get("column_id"):
        board_id = _column_board_id(supabase, data["column_id"])
        if not board_id:
            raise ValueError("Column not found")
    else:
        board_id = ensure_personal_board(supabase, user_id, owned_first_artist)
    authz.require_board_edit(supabase, user_id, board_id)
    data["board_id"] = board_id
    data["user_id"] = user_id
    result = supabase.table("board_tasks").insert(data).execute()
    task = result.data[0] if result.data else {}

    if task:
        task_id = task["id"]

        # Merge-on-write: preserve co-owners' rows, snapshot labels. Re-set the accessible ids
        # on the task dict BEFORE emit — the Slack bridge routes via task["project_ids"][0].
        task["artist_ids"] = _merge_junction(
            supabase,
            task_id,
            table="board_task_artists",
            fk_column="artist_id",
            submitted_ids=artist_ids,
            labels=artist_labels,
            access_fn=_owned_artist_ids,
            user_id=user_id,
        )
        task["project_ids"] = _merge_junction(
            supabase,
            task_id,
            table="board_task_projects",
            fk_column="project_id",
            submitted_ids=project_ids,
            labels=project_labels,
            access_fn=_accessible_project_ids,
            user_id=user_id,
        )
        task["contract_ids"] = _merge_junction(
            supabase,
            task_id,
            table="board_task_contracts",
            fk_column="project_file_id",
            submitted_ids=contract_ids,
            labels=contract_labels,
            access_fn=_accessible_contract_ids,
            user_id=user_id,
        )
        await events.emit(events.TASK_CREATED, {"user_id": user_id, "task": task})

    return task


def _apply_cross_board_move(db: Client, user_id: str, task: dict, dst_board: str) -> None:
    """Validate + perform a task's board change (§7.3). task = {id, board_id, parent_task_id}."""
    if task.get("parent_task_id"):
        raise ValueError("A subtask cannot be moved to another board on its own")
    boards = {
        r["id"]: r
        for r in (
            db.table("boards").select("id, team_id, owner_id").in_("id", [task["board_id"], dst_board]).execute().data
            or []
        )
    }
    s, d = boards.get(task["board_id"]), boards.get(dst_board)
    if not s or not d:
        raise ValueError("Board not found")
    authz.require_board_edit(db, user_id, s["id"])
    authz.require_board_edit(db, user_id, d["id"])
    if s["team_id"] != d["team_id"]:
        raise ValueError("Cannot move a task across teams")
    if s["team_id"] is None and s["owner_id"] != d["owner_id"]:
        raise ValueError("Cannot move a task across personal boards")
    # Move the subtree (direct children — single-level per the UI) with the parent.
    db.table("board_tasks").update({"board_id": dst_board}).eq("parent_task_id", task["id"]).execute()


async def update_task(supabase: Client, user_id: str, task_id: str, data: dict) -> dict:
    """Update a task and its junction relations. Gated on the task's board (require_board_edit)."""
    task_row = supabase.table("board_tasks").select("id, board_id, parent_task_id").eq("id", task_id).limit(1).execute()
    if not task_row.data:
        return {}
    src_board = task_row.data[0]["board_id"]
    authz.require_board_edit(supabase, user_id, src_board)

    # §7.3 — a column change that crosses boards routes through the move rules.
    if data.get("column_id"):
        dst_board = _column_board_id(supabase, data["column_id"])
        if dst_board and dst_board != src_board:
            _apply_cross_board_move(supabase, user_id, task_row.data[0], dst_board)
            data["board_id"] = dst_board

    artist_ids = data.pop("artist_ids", None)
    project_ids = data.pop("project_ids", None)
    contract_ids = data.pop("contract_ids", None)
    artist_labels = data.pop("artist_labels", None) or {}
    project_labels = data.pop("project_labels", None) or {}
    contract_labels = data.pop("contract_labels", None) or {}

    # Empty string column_id means "clear it" (backlog)
    if "column_id" in data and data["column_id"] == "":
        data["column_id"] = None

    # Handle completed_at based on column change
    if "column_id" in data and data["column_id"] is not None:
        col_result = supabase.table("board_columns").select("title").eq("id", data["column_id"]).single().execute()
        if col_result.data:
            col_title = col_result.data.get("title", "").lower()
            if col_title == "done":
                data["completed_at"] = datetime.now(UTC).isoformat()
            else:
                data["completed_at"] = None

    # Build update dict — include explicit None for column_id to clear it
    clean = {}
    for k, v in data.items():
        if k in ("column_id", "completed_at", "parent_task_id"):
            clean[k] = v  # Allow None to clear these fields
        elif v is not None:
            clean[k] = v
    if clean:
        result = supabase.table("board_tasks").update(clean).eq("id", task_id).execute()
        task = result.data[0] if result.data else {}
    else:
        result = supabase.table("board_tasks").select("*").eq("id", task_id).single().execute()
        task = result.data or {}

    if task:
        # Merge-on-write: replace only the editor's accessible partition (preserve co-owners'
        # rows) and snapshot labels. Absent link field (None) stays a no-op — do NOT touch it.
        # Re-set the accessible ids on the task dict BEFORE emit (only for present fields) so a
        # project-link change routes to the project channel, symmetric with create_task.
        if artist_ids is not None:
            task["artist_ids"] = _merge_junction(
                supabase,
                task_id,
                table="board_task_artists",
                fk_column="artist_id",
                submitted_ids=artist_ids,
                labels=artist_labels,
                access_fn=_owned_artist_ids,
                user_id=user_id,
            )
        if project_ids is not None:
            task["project_ids"] = _merge_junction(
                supabase,
                task_id,
                table="board_task_projects",
                fk_column="project_id",
                submitted_ids=project_ids,
                labels=project_labels,
                access_fn=_accessible_project_ids,
                user_id=user_id,
            )
        if contract_ids is not None:
            task["contract_ids"] = _merge_junction(
                supabase,
                task_id,
                table="board_task_contracts",
                fk_column="project_file_id",
                submitted_ids=contract_ids,
                labels=contract_labels,
                access_fn=_accessible_contract_ids,
                user_id=user_id,
            )

        await events.emit(events.TASK_UPDATED, {"user_id": user_id, "task": task})

    return task


async def delete_task(supabase: Client, user_id: str, task_id: str) -> bool:
    """Delete a task (junction rows cascade). Children get parent_task_id set to NULL."""
    board_id = _task_board_id(supabase, task_id)
    if not board_id:
        return False
    authz.require_board_edit(supabase, user_id, board_id)
    result = supabase.table("board_tasks").delete().eq("id", task_id).execute()
    return bool(result.data)


async def get_task_detail(supabase: Client, user_id: str, task_id: str) -> dict:
    """Get a single task with full related data including children and parent."""
    result = supabase.table("board_tasks").select("*").eq("id", task_id).limit(1).execute()
    task = result.data[0] if result.data else None
    if not task:
        return {}
    authz.require_board_access(supabase, user_id, task["board_id"])

    # Linked entities — normalized to the SAME [{id, name, can_open}] shape as _enrich_tasks
    # (label-first, fallback lookup on NULL, can_open via the access helpers), via the shared
    # _shape_links closure (single task's links wrapped as {task_id: links} → same code path).
    artist_links = _get_junction_with_labels(supabase, "board_task_artists", "artist_id", [task_id])
    project_links = _get_junction_with_labels(supabase, "board_task_projects", "project_id", [task_id])
    contract_links = _get_junction_with_labels(supabase, "board_task_contracts", "project_file_id", [task_id])

    shape_artists = _shape_links(supabase, user_id, artist_links, "artists", "name", _owned_artist_ids)
    shape_projects = _shape_links(supabase, user_id, project_links, "projects", "name", _accessible_project_ids)
    shape_documents = _shape_links(
        supabase, user_id, contract_links, "project_files", "file_name", _accessible_contract_ids
    )
    task["artists"] = shape_artists(artist_links.get(task_id, []))
    task["projects"] = shape_projects(project_links.get(task_id, []))
    task["documents"] = shape_documents(contract_links.get(task_id, []))
    # Keep the raw ids alongside the shaped lists for the frontend.
    task["artist_ids"] = [a["id"] for a in task["artists"]]
    task["project_ids"] = [p["id"] for p in task["projects"]]
    task["contract_ids"] = [d["id"] for d in task["documents"]]

    # Creator — same {user_id, full_name, avatar_url} shape as _enrich_tasks (single-row lookup;
    # resolves even for a creator who has left the team). Null-safe when the task has no user_id.
    creator = {}
    if task.get("user_id"):
        cres = (
            supabase.table("profiles").select("id, full_name, avatar_url").eq("id", task["user_id"]).limit(1).execute()
        )
        creator = (cres.data or [{}])[0]
    task["creator"] = {
        "user_id": task.get("user_id"),
        "full_name": creator.get("full_name"),
        "avatar_url": creator.get("avatar_url"),
    }

    comments_result = (
        supabase.table("board_task_comments").select("*").eq("task_id", task_id).order("created_at").execute()
    )

    # Fetch children if this is a parent task
    children = []
    if task.get("is_parent"):
        children_result = (
            supabase.table("board_tasks").select("*").eq("parent_task_id", task_id).order("position").execute()
        )
        children = _enrich_tasks(supabase, children_result.data or [], user_id)

    # Fetch parent info if this is a child task
    parent = None
    if task.get("parent_task_id"):
        parent_result = (
            supabase.table("board_tasks").select("id, title").eq("id", task["parent_task_id"]).single().execute()
        )
        parent = parent_result.data

    task["comments"] = comments_result.data or []
    task["children"] = children
    task["parent"] = parent

    _attach_assignees(supabase, task)

    return task


async def get_tasks_by_date_range(
    supabase: Client,
    user_id: str,
    start: str,
    end: str,
    board_id: str | None = None,
    artist_id: str | None = None,
) -> list:
    """Get non-parent tasks that have due_date or start_date within a date range."""
    board_ids = _resolve_read_board_ids(supabase, user_id, artist_id, board_id)
    if not board_ids:
        return []
    due_result = (
        supabase.table("board_tasks")
        .select("*")
        .in_("board_id", board_ids)
        .or_("is_parent.eq.false,is_parent.is.null")
        .gte("due_date", start)
        .lte("due_date", end)
        .execute()
    )
    start_result = (
        supabase.table("board_tasks")
        .select("*")
        .in_("board_id", board_ids)
        .or_("is_parent.eq.false,is_parent.is.null")
        .gte("start_date", start)
        .lte("start_date", end)
        .execute()
    )

    seen = set()
    tasks = []
    for task in (due_result.data or []) + (start_result.data or []):
        if task["id"] not in seen:
            seen.add(task["id"])
            tasks.append(task)

    return _enrich_tasks(supabase, tasks, user_id)


# --- Period-based Tasks ---


async def get_tasks_by_period(
    supabase: Client,
    user_id: str,
    period_start: str,
    period_end: str,
    is_current: bool = True,
    board_id: str | None = None,
    artist_id: str | None = None,
) -> list:
    """Get tasks filtered by period for date-based board views."""
    board_ids = _resolve_read_board_ids(supabase, user_id, artist_id, board_id)
    if not board_ids:
        return []
    try:
        if is_current:
            # Current period: single query — all tasks except those completed before this period
            result = (
                supabase.table("board_tasks")
                .select("*")
                .in_("board_id", board_ids)
                .or_("is_parent.eq.false,is_parent.is.null")
                .or_(f"completed_at.is.null,completed_at.gte.{period_start}")
                .order("position")
                .execute()
            )
            tasks = result.data or []
        else:
            # Past period: done tasks completed in period + tasks created in period
            done_result = (
                supabase.table("board_tasks")
                .select("*")
                .in_("board_id", board_ids)
                .or_("is_parent.eq.false,is_parent.is.null")
                .filter("completed_at", "not.is", "null")
                .gte("completed_at", period_start)
                .lte("completed_at", period_end)
                .execute()
            )
            created_result = (
                supabase.table("board_tasks")
                .select("*")
                .in_("board_id", board_ids)
                .or_("is_parent.eq.false,is_parent.is.null")
                .gte("created_at", period_start)
                .lte("created_at", period_end)
                .execute()
            )
            seen = set()
            tasks = []
            for task in (done_result.data or []) + (created_result.data or []):
                if task["id"] not in seen:
                    seen.add(task["id"])
                    tasks.append(task)
    except Exception:
        # Fallback: if completed_at column missing or other error, return all tasks
        return await get_tasks(supabase, user_id, board_id=board_id, artist_id=artist_id)

    return _enrich_tasks(supabase, tasks, user_id)


# --- Parent Tasks ---


async def create_parent_task(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a parent task (no column_id, is_parent=True)."""
    artist_ids = data.pop("artist_ids", [])
    project_ids = data.pop("project_ids", [])
    artist_labels = data.pop("artist_labels", None) or {}
    project_labels = data.pop("project_labels", None) or {}

    if not data.get("start_date"):
        data["start_date"] = str(date.today())

    owned = _owned_artist_ids(supabase, user_id, artist_ids) if artist_ids else []
    board_id = data.get("board_id") or ensure_personal_board(supabase, user_id, owned[0] if owned else None)
    authz.require_board_edit(supabase, user_id, board_id)
    data["board_id"] = board_id
    data["user_id"] = user_id
    data["is_parent"] = True
    result = supabase.table("board_tasks").insert(data).execute()
    task = result.data[0] if result.data else {}

    if task:
        task_id = task["id"]
        task["artist_ids"] = _merge_junction(
            supabase,
            task_id,
            table="board_task_artists",
            fk_column="artist_id",
            submitted_ids=artist_ids,
            labels=artist_labels,
            access_fn=_owned_artist_ids,
            user_id=user_id,
        )
        task["project_ids"] = _merge_junction(
            supabase,
            task_id,
            table="board_task_projects",
            fk_column="project_id",
            submitted_ids=project_ids,
            labels=project_labels,
            access_fn=_accessible_project_ids,
            user_id=user_id,
        )

    return task


async def get_all_parents_with_children(
    supabase: Client,
    user_id: str,
    search: str | None = None,
    artist_id: str | None = None,
    board_id: str | None = None,
) -> dict:
    """Get all parent tasks with nested children for the overview tab."""
    board_ids = _resolve_read_board_ids(supabase, user_id, artist_id, board_id)
    if not board_ids:
        return {"parents": [], "ungrouped": []}

    # Fetch parent tasks
    query = (
        supabase.table("board_tasks")
        .select("*")
        .in_("board_id", board_ids)
        .eq("is_parent", True)
        .order("created_at", desc=True)
    )
    parents_result = query.execute()
    parents = parents_result.data or []

    # Fetch ALL child tasks on these boards (we'll group them)
    children_result = (
        supabase.table("board_tasks")
        .select("*")
        .in_("board_id", board_ids)
        .or_("is_parent.eq.false,is_parent.is.null")
        .order("position")
        .execute()
    )
    all_children = children_result.data or []

    # Enrich both sets
    all_tasks = parents + all_children
    enriched = _enrich_tasks(supabase, all_tasks, user_id)

    # Split back into parents and children
    parent_map = {}
    enriched_children = []
    for t in enriched:
        if t.get("is_parent"):
            parent_map[t["id"]] = t
        else:
            enriched_children.append(t)

    # Get column names for child task display
    columns_result = supabase.table("board_columns").select("id, title").in_("board_id", board_ids).execute()
    column_names = {c["id"]: c["title"] for c in (columns_result.data or [])}

    # Add column_title to parents (for status display) and children
    for parent in parent_map.values():
        parent["column_title"] = column_names.get(parent.get("column_id"), "")
    for child in enriched_children:
        child["column_title"] = column_names.get(child.get("column_id"), "")

    children_by_parent = {}
    ungrouped = []
    for child in enriched_children:
        pid = child.get("parent_task_id")
        if pid and pid in parent_map:
            if pid not in children_by_parent:
                children_by_parent[pid] = []
            children_by_parent[pid].append(child)
        elif not pid:
            ungrouped.append(child)

    # Build result
    result = []
    for pid, parent in parent_map.items():
        children = children_by_parent.get(pid, [])
        parent["children"] = children
        parent["child_count"] = len(children)
        result.append(parent)

    # Apply filters
    if artist_id:
        result = [
            p
            for p in result
            if artist_id in p.get("artist_ids", [])
            or any(artist_id in c.get("artist_ids", []) for c in p.get("children", []))
        ]
        ungrouped = [c for c in ungrouped if artist_id in c.get("artist_ids", [])]

    if search:
        search_lower = search.lower()
        filtered = []
        for p in result:
            # Match parent title or any child title
            if search_lower in p.get("title", "").lower():
                filtered.append(p)
            else:
                matching_children = [c for c in p.get("children", []) if search_lower in c.get("title", "").lower()]
                if matching_children:
                    p["children"] = matching_children
                    p["child_count"] = len(matching_children)
                    filtered.append(p)
        result = filtered
        ungrouped = [c for c in ungrouped if search_lower in c.get("title", "").lower()]

    return {"parents": result, "ungrouped": ungrouped}


# --- Comments ---


async def create_comment(supabase: Client, user_id: str, task_id: str, content: str) -> dict:
    """Add a comment to a task. Raises PermissionError if the task doesn't exist, or
    HTTPException(404) (via require_board_edit) if the caller can't edit its board."""
    board_id = _task_board_id(supabase, task_id)
    if not board_id:
        raise PermissionError("denied")
    authz.require_board_edit(supabase, user_id, board_id)

    result = (
        supabase.table("board_task_comments")
        .insert(
            {
                "task_id": task_id,
                "user_id": user_id,
                "content": content,
            }
        )
        .execute()
    )
    return result.data[0] if result.data else {}


async def delete_comment(supabase: Client, user_id: str, comment_id: str) -> bool:
    """Delete a comment (only the author can delete)."""
    result = supabase.table("board_task_comments").delete().eq("id", comment_id).eq("user_id", user_id).execute()
    return bool(result.data)


# --- Assignees (spec §5.6, multiple assignees per task) ---


def _attach_assignees(db: Client, task: dict) -> dict:
    """Attach task["assignees"] = [{user_id, full_name, avatar_url}, ...] to a task dict."""
    rows = db.table("board_task_assignees").select("user_id").eq("task_id", task["id"]).execute().data or []
    user_ids = [r["user_id"] for r in rows]
    profiles_by_id = {}
    if user_ids:
        profiles_result = db.table("profiles").select("id, full_name, avatar_url").in_("id", user_ids).execute()
        profiles_by_id = {p["id"]: p for p in (profiles_result.data or [])}
    task["assignees"] = [
        {
            "user_id": uid,
            "full_name": profiles_by_id.get(uid, {}).get("full_name"),
            "avatar_url": profiles_by_id.get(uid, {}).get("avatar_url"),
        }
        for uid in user_ids
    ]
    return task


async def add_assignee(db: Client, actor_id: str, task_id: str, target_user_id: str) -> dict:
    """Assign target_user_id to a task. Gated on the task's board (require_board_edit), then
    validated per §5.6 (personal board -> owner only, team board -> target must be a member)."""
    board_id = _task_board_id(db, task_id)
    if not board_id:
        raise ValueError("Task not found")
    authz.require_board_edit(db, actor_id, board_id)
    if not authz.can_assign_user(db, target_user_id, board_id):
        raise PermissionError("This user cannot be assigned to this board")
    db.table("board_task_assignees").upsert(
        {"task_id": task_id, "user_id": target_user_id, "assigned_by": actor_id},
        on_conflict="task_id,user_id",
        ignore_duplicates=True,
    ).execute()
    return {"task_id": task_id, "user_id": target_user_id}


async def remove_assignee(db: Client, actor_id: str, task_id: str, target_user_id: str) -> dict:
    """Unassign target_user_id from a task. Gated on the task's board (require_board_edit)."""
    board_id = _task_board_id(db, task_id)
    if not board_id:
        raise ValueError("Task not found")
    authz.require_board_edit(db, actor_id, board_id)
    db.table("board_task_assignees").delete().eq("task_id", task_id).eq("user_id", target_user_id).execute()
    return {"unassigned": target_user_id}


# --- Reorder + Defaults ---


async def batch_reorder(supabase: Client, user_id: str, reorders: list[dict]) -> bool:
    """Batch reorder (drag-and-drop). If a task's target column is on a different board, move
    the task (and its subtree) to that board — but only within the same team/owner scope
    (cross-team moves are rejected, §7.3). A subtask's board always follows its parent."""
    for reorder in reorders:
        task_id, target_col = reorder["task_id"], reorder["target_column_id"]
        task = supabase.table("board_tasks").select("id, board_id, parent_task_id").eq("id", task_id).limit(1).execute()
        col = supabase.table("board_columns").select("board_id").eq("id", target_col).limit(1).execute()
        if not task.data or not col.data:
            continue
        t = task.data[0]
        src_board, dst_board = t["board_id"], col.data[0]["board_id"]
        update = {"column_id": target_col, "position": reorder["position"]}
        if dst_board != src_board:
            _apply_cross_board_move(supabase, user_id, t, dst_board)  # validates + gates both boards
            update["board_id"] = dst_board
        else:
            authz.require_board_edit(supabase, user_id, src_board)
        supabase.table("board_tasks").update(update).eq("id", task_id).execute()  # scoped by id, not user_id
    return True


# --- Board CRUD ---


async def create_board(
    db: Client,
    user_id: str,
    name: str,
    team_id: str | None = None,
    artist_id: str | None = None,
    description: str | None = None,
) -> dict:
    if team_id is not None and not authz.is_team_member(db, user_id, team_id):
        raise PermissionError("Not a team member")
    res = (
        db.table("boards")
        .insert(
            {"owner_id": user_id, "team_id": team_id, "artist_id": artist_id, "name": name, "description": description}
        )
        .execute()
    )
    return res.data[0] if res.data else {}


async def list_boards(db: Client, user_id: str, team_id: str | None = None) -> list[dict]:
    if team_id is not None:
        if not authz.is_team_member(db, user_id, team_id):
            raise PermissionError("Not a team member")
        # §5.8: an archived team's boards are hidden.
        team = db.table("teams").select("archived_at").eq("id", team_id).limit(1).execute()
        if team.data and team.data[0].get("archived_at"):
            return []
        q = db.table("boards").select("*").eq("team_id", team_id)
    else:
        q = db.table("boards").select("*").eq("owner_id", user_id).is_("team_id", "null")
    return q.eq("archived", False).order("position").order("created_at").execute().data or []


async def rename_board(db: Client, user_id: str, board_id: str, fields: dict) -> dict:
    authz.require_board_edit(db, user_id, board_id)
    clean = {k: v for k, v in fields.items() if v is not None}
    if not clean:
        return authz.get_board(db, board_id) or {}
    return (db.table("boards").update(clean).eq("id", board_id).execute().data or [{}])[0]


async def archive_board(db: Client, user_id: str, board_id: str) -> dict:
    """Archive a board. Personal → owner; team → team admin (destructive for all members)."""
    board = authz.get_board(db, board_id)
    if not board:
        raise ValueError("Board not found")
    if board["team_id"] is None:
        if board["owner_id"] != user_id:
            raise PermissionError("Not your board")
    elif not authz.is_team_admin(db, user_id, board["team_id"]):
        raise PermissionError("Admin access required to archive a team board")
    db.table("boards").update({"archived": True}).eq("id", board_id).execute()
    return {"archived": board_id}


def _can_archive_board(db, user_id: str, board: dict) -> bool:
    """Same gate as archive_board: personal → owner; team board → team admin."""
    if board.get("team_id") is None:
        return board.get("owner_id") == user_id
    return authz.is_team_admin(db, user_id, board["team_id"])


async def delete_board(db: Client, user_id: str, board_id: str, confirm_name: str) -> dict:
    """Permanently delete a board (cascade removes its columns/tasks/assignees/junction rows).
    Gated identically to archive_board, plus a normalized typed-name confirmation.
    Reads the row directly (authz.get_board omits `name`)."""
    res = db.table("boards").select("id, name, team_id, owner_id").eq("id", board_id).limit(1).execute()
    board = (res.data or [None])[0]
    if not board:
        raise ValueError("Board not found")
    if not _can_archive_board(db, user_id, board):
        raise PermissionError("Not allowed to delete this board")
    if normalize_name(confirm_name) != normalize_name(board.get("name")):
        raise ConfirmationError("Confirmation does not match")
    tasks = db.table("board_tasks").select("id", count="exact").eq("board_id", board_id).execute()
    db.table("boards").delete().eq("id", board_id).execute()  # FK cascade: columns/tasks/assignees/junctions
    return {"deleted": board_id, "tasks": tasks.count or 0}


async def restore_board(db: Client, user_id: str, board_id: str) -> dict:
    """Un-archive a board. Same gate as archive (no name needed → authz.get_board is fine)."""
    board = authz.get_board(db, board_id)
    if not board:
        raise ValueError("Board not found")
    if not _can_archive_board(db, user_id, board):
        raise PermissionError("Not allowed to restore this board")
    db.table("boards").update({"archived": False}).eq("id", board_id).execute()
    return {"restored": board_id}


async def list_archived_boards(db: Client, user_id: str, team_id: str | None = None) -> list[dict]:
    """Archived boards with a task_count each. Team context → team admin only; else the
    caller's archived personal boards. Same gate as restore/delete (no view-vs-act split)."""
    if team_id is not None:
        if not authz.is_team_admin(db, user_id, team_id):
            raise PermissionError("Admin access required")
        q = db.table("boards").select("*").eq("team_id", team_id)
    else:
        q = db.table("boards").select("*").eq("owner_id", user_id).is_("team_id", "null")
    boards = q.eq("archived", True).order("created_at").execute().data or []
    for b in boards:
        c = db.table("board_tasks").select("id", count="exact").eq("board_id", b["id"]).execute()
        b["task_count"] = c.count or 0
    return boards


async def create_default_columns(
    supabase: Client, user_id: str, artist_id: str | None = None, board_id: str | None = None
) -> list:
    """Create default Kanban columns for a new board. When board_id is given (e.g. a team
    board), target it directly (gated by require_board_edit); otherwise resolve/create the
    caller's personal board for artist_id (legacy default)."""
    if board_id:
        authz.require_board_edit(supabase, user_id, board_id)
    else:
        board_id = ensure_personal_board(supabase, user_id, artist_id)
        authz.require_board_edit(supabase, user_id, board_id)
    defaults = [
        {"title": "Backlog", "position": 0, "color": "#8b5cf6"},
        {"title": "To Do", "position": 1, "color": "#6366f1"},
        {"title": "In Progress", "position": 2, "color": "#f59e0b"},
        {"title": "Review", "position": 3, "color": "#3b82f6"},
        {"title": "Done", "position": 4, "color": "#10b981"},
    ]
    columns = []
    for col in defaults:
        col.update({"user_id": user_id, "artist_id": artist_id, "board_id": board_id})
        result = supabase.table("board_columns").insert(col).execute()
        if result.data:
            columns.append(result.data[0])
    return columns
