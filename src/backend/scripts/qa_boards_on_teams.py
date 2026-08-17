"""HTTP QA loop for Boards on Teams (spec 2026-08-16) against the REAL stack.

Every case here is one that pytest structurally CANNOT cover: `MockQueryBuilder`
(`tests/conftest.py`) returns `self` from `.eq()` / `.in_()` and throws the
arguments away, so no mock test can prove a filter is actually on the query, and
none of them can reach a Postgres type error or an RLS/trigger refusal. This
script talks to a real backend with real JWTs, real RLS and real triggers.

PREREQUISITES
  * BOTH migrations applied: 20260818000001_boards_to_orgs.sql AND
    20260818000002_drop_board_teams.sql.
  * A backend running locally with CREDITS_ENABLED=true and
    LICENSING_ENABLED=true (self-serve `POST /orgs` 503s without both).
  * VITE_SUPABASE_URL / VITE_SUPABASE_SECRET_KEY / VITE_SUPABASE_ANON_KEY in the
    process environment. This script reads them BY NAME and never prints one —
    let the shell supply the values:

        cd src/backend
        set -a; . ../../.env; set +a
        LICENSING_ENABLED=true CREDITS_ENABLED=true RESEND_API_KEY=qa-disabled \
          POSTHOG_ENABLED=false poetry run uvicorn main:app --port 8017 &
        poetry run python scripts/qa_boards_on_teams.py --port 8017

Creates 3 throwaway `qa-boards-*@example.com` users and one team, and CLEANS UP
EVERY ROW in a finally block — the DB is shared with prod.
"""

import os
import secrets
import sys

import httpx
from supabase import create_client

URL = os.environ["VITE_SUPABASE_URL"]
SECRET = os.environ["VITE_SUPABASE_SECRET_KEY"]
ANON = os.environ["VITE_SUPABASE_ANON_KEY"]
PORT = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8017"
API = f"http://127.0.0.1:{PORT}"

sb = create_client(URL, SECRET)
SUFFIX = secrets.token_hex(4)

results = []


def check(label, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    results.append((tag, label, detail))
    print(f"[{tag}] {label}" + (f" — {detail}" if detail and not cond else ""))


def api(token):
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {token}"}, timeout=30)


users = {}  # role -> {id, email, password, token}
org_id = None
board_id = None


def _create_user(role):
    """Throwaway auth user + signed-in session, registered in `users` for both
    the flow and the finally-block cleanup."""
    email = f"qa-boards-{role}-{SUFFIX}@example.com"
    password = secrets.token_urlsafe(16)
    res = sb.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
    anon = create_client(URL, ANON)
    session = anon.auth.sign_in_with_password({"email": email, "password": password})
    users[role] = {
        "id": res.user.id,
        "email": email,
        "password": password,
        "token": session.session.access_token,
        "client": anon,  # end-user JWT client — case 7 needs one
    }
    print(f"       created {role}: {email} ({res.user.id})")
    return users[role]


def _board_ids(token, team_id):
    with api(token) as c:
        r = c.get("/boards/boards", params={"team_id": team_id})
        return r.status_code, [b["id"] for b in (r.json().get("boards") or [])]


try:
    # -------------------------------------------------------------- setup --
    owner = _create_user("owner")  # covers the team (Basic), creates the board
    member = _create_user("member")  # active seat on the team
    outsider = _create_user("outsider")  # never invited — the negative case

    # A real Basic subscription: POST /orgs is slot-gated, and a Free user
    # cannot own a team at all (that IS the "Free = personal boards only" rule).
    sb.table("subscriptions").upsert(
        {"user_id": owner["id"], "tier": "basic", "status": "active"}, on_conflict="user_id"
    ).execute()

    # ------------------------------ 1. team -> invite -> accept -> board --
    with api(owner["token"]) as c:
        r = c.post("/orgs", json={"name": f"QA Boards {SUFFIX}"})
        check("1. owner (Basic) creates a self-serve team", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        org_id = r.json()["id"]
        r = c.post(f"/orgs/{org_id}/invites", json={"email": member["email"], "role": "member"})
        check("1b. owner invites the member", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    token_row = (
        sb.table("pending_org_invites").select("token").eq("org_id", org_id).eq("email", member["email"]).execute().data
    )
    with api(member["token"]) as c:
        r = c.post(f"/orgs/invites/{token_row[0]['token']}/accept")
        check("1c. member accepts", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")

    with api(owner["token"]) as c:
        r = c.post("/boards/boards", json={"name": f"QA Board {SUFFIX}", "team_id": org_id})
        check("1d. owner creates a team board", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        board_id = r.json()["id"]

    status, ids = _board_ids(member["token"], org_id)
    check("1e. member sees the open team board", status == 200 and board_id in ids, f"{status}: {ids}")

    with api(outsider["token"]) as c:
        r = c.get("/boards/boards", params={"team_id": org_id})
        check("1f. outsider listing the team's boards -> 403", r.status_code == 403, f"{r.status_code}: {r.text[:150]}")

    # ------------------- 2. restricted with an EMPTY member list -> 200 --
    # The most common save the dialog makes when visibility goes back to
    # "Everyone": an empty .in_() would need a placeholder, and any non-uuid
    # placeholder makes Postgres raise 22P02 against this uuid column.
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"restricted": True, "member_user_ids": []})
        check(
            "2. PUT restricted=true with member_user_ids=[] -> 200 (no 22P02)",
            r.status_code == 200 and r.json().get("member_user_ids") == [],
            f"{r.status_code}: {r.text[:250]}",
        )

    # ---------------------------- 3. a user outside the org -> 422 --
    # NOTE this only proves the ORG_ID filter: an outsider has no org_members
    # row at all, so status='active' is never exercised here. The discriminating
    # input for that half is a SUSPENDED member of this same org — case 6d,
    # after the suspend.
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"member_user_ids": [outsider["id"]]})
        check(
            "3. PUT member_user_ids=[outsider] -> 422 (the org_id filter is real)",
            r.status_code == 422,
            f"{r.status_code}: {r.text[:200]}",
        )

    # ------------------------- 4. narrowing hides, listing restores --
    status, ids = _board_ids(member["token"], org_id)
    check("4. restricted board is gone from the member's list", status == 200 and board_id not in ids, f"{status}")
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"member_user_ids": [member["id"]]})
        check(
            "4b. owner adds the member to the board",
            r.status_code == 200 and r.json().get("member_user_ids") == [member["id"]],
            f"{r.status_code}: {r.text[:250]}",
        )
    # NOW the rejected-write probe means something: the list is non-empty, so a
    # rejected PUT that wrongly wrote would be visible as a changed set. (Run
    # before case 2 it was vacuous — board_members was already empty.)
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"member_user_ids": [outsider["id"]]})
    rows = [
        x["user_id"]
        for x in (sb.table("board_members").select("user_id").eq("board_id", board_id).execute().data or [])
    ]
    check(
        "4b-i. a rejected PUT leaves the existing member set untouched",
        r.status_code == 422 and rows == [member["id"]],
        f"{r.status_code} rows={rows}",
    )
    status, ids = _board_ids(member["token"], org_id)
    check("4c. member sees the board again", status == 200 and board_id in ids, f"{status}: {ids}")

    with api(member["token"]) as c:
        r = c.get(f"/orgs/{org_id}/members")
        roster = (r.json() or {}).get("members") or [] if r.status_code == 200 else []
        check(
            "4d. member-visible roster: both seats, names, NO emails",
            r.status_code == 200
            and {m["user_id"] for m in roster} == {owner["id"], member["id"]}
            and all("email" not in m for m in roster),
            f"{r.status_code}: {str(roster)[:250]}",
        )

    # ---------------------------------------------- 5. assignment gate --
    with api(owner["token"]) as c:
        r = c.post("/boards/columns", json={"title": "To Do", "board_id": board_id})
        check("5. owner creates a column", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        column_id = r.json()["id"]
        r = c.post("/boards/tasks", json={"board_id": board_id, "column_id": column_id, "title": "QA task"})
        check("5b. owner creates a task", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        task_id = r.json()["id"]

    with api(member["token"]) as c:
        r = c.post(f"/boards/tasks/{task_id}/assignees", json={"user_id": member["id"]})
        check("5c. member assigns themselves -> 200", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        r = c.post(f"/boards/tasks/{task_id}/assignees", json={"user_id": outsider["id"]})
        check(
            "5d. assigning an outsider -> 403 (can_assign_user gates the TARGET)",
            r.status_code == 403,
            f"{r.status_code}: {r.text[:200]}",
        )
    with api(outsider["token"]) as c:
        r = c.post(f"/boards/tasks/{task_id}/assignees", json={"user_id": outsider["id"]})
        check(
            "5e. outsider assigning -> 404 (require_board_edit never leaks existence)",
            r.status_code == 404,
            f"{r.status_code}: {r.text[:200]}",
        )

    # ------------------- 6. suspend destroys nothing; remove purges only --
    member_row = sb.table("org_members").select("id").eq("org_id", org_id).eq("user_id", member["id"]).execute().data[0]
    member_id = member_row["id"]

    def _footprint():
        bm = sb.table("board_members").select("id").eq("board_id", board_id).eq("user_id", member["id"]).execute().data
        asg = (
            sb.table("board_task_assignees")
            .select("id")
            .eq("task_id", task_id)
            .eq("user_id", member["id"])
            .execute()
            .data
        )
        return len(bm or []), len(asg or [])

    with api(owner["token"]) as c:
        r = c.post(f"/orgs/{org_id}/members/{member_id}/suspend")
        check("6. owner suspends the member", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    bm_n, asg_n = _footprint()
    check("6b. suspend keeps BOTH the board_members row and the assignment", (bm_n, asg_n) == (1, 1), f"{bm_n},{asg_n}")
    status, ids = _board_ids(member["token"], org_id)
    check("6c. the suspended seat can no longer list the team's boards", status == 403, f"{status}")

    # The status='active' half of the member-id filter, which case 3 could not
    # reach: this user IS in the org, so only the seat STATUS can reject them.
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"member_user_ids": [member["id"], owner["id"]]})
    check(
        "6c-i. ADDING a suspended seat to a board -> 422 (the status filter is real)",
        r.status_code == 422,
        f"{r.status_code}: {r.text[:200]}",
    )
    # ...but the member is ALREADY listed, so re-saving the board (e.g. a plain
    # rename) must still succeed — otherwise a suspend would wedge the board's
    # settings until the admin dropped the person.
    with api(owner["token"]) as c:
        r = c.put(f"/boards/boards/{board_id}", json={"name": "Renamed while a member is suspended"})
    check(
        "6c-ii. renaming a board that still lists a suspended member -> 200",
        r.status_code == 200,
        f"{r.status_code}: {r.text[:200]}",
    )

    with api(owner["token"]) as c:
        r = c.delete(f"/orgs/{org_id}/members/{member_id}")
        check("6d. owner removes the member", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    bm_n, asg_n = _footprint()
    check(
        "6e. remove purges board_members but KEEPS the assignment (history)",
        (bm_n, asg_n) == (0, 1),
        f"board_members={bm_n} assignments={asg_n}",
    )

    # ------------- 7. team_id is locked against an end-user JWT (trigger) --
    # Must be the BOARD OWNER: the UPDATE policy is owner-or-admin, so a plain
    # member's UPDATE is RLS-filtered to zero rows and "succeeds" having changed
    # nothing — that asserts on the policy, not on boards_lock_team_id_trg, and
    # passes vacuously. Assert on the RAISED ERROR, never on a row count.
    err = ""
    try:
        owner["client"].table("boards").update({"team_id": None}).eq("id", board_id).execute()
    except Exception as e:  # noqa: BLE001 — the refusal IS the assertion
        err = str(e)
    check(
        "7. owner cannot move a board between teams from the client (insufficient_privilege)",
        "insufficient_privilege" in err or "42501" in err,
        f"error={err[:250] or '<no error raised — the trigger did NOT fire>'}",
    )
    still = sb.table("boards").select("team_id").eq("id", board_id).execute().data[0]
    check("7b. team_id is unchanged", still["team_id"] == org_id, str(still))

finally:
    # ---------------------------------------------------------- cleanup --
    print("\n--- cleanup (shared DB) ---")
    try:
        # Boards FIRST: boards.team_id is ON DELETE RESTRICT, so a live board
        # blocks the org teardown. Columns/tasks/assignees/board_members all
        # cascade off boards.
        if org_id:
            sb.table("boards").delete().eq("team_id", org_id).execute()
        for u in users.values():
            sb.table("boards").delete().eq("owner_id", u["id"]).execute()
        # Then the org row: org_members cascades at trigger depth > 1, which is
        # where the last-admin guard yields (a direct delete of the sole admin's
        # row would raise).
        if org_id:
            sb.table("pending_org_invites").delete().eq("org_id", org_id).execute()
            sb.table("organizations").delete().eq("id", org_id).execute()
        wallet_ids = []
        for owner_type, owner_ids in (("user", [u["id"] for u in users.values()]), ("org", [org_id] if org_id else [])):
            for oid in owner_ids:
                w = sb.table("credit_wallets").select("id").eq("owner_type", owner_type).eq("owner_id", oid).execute()
                wallet_ids.extend(x["id"] for x in (w.data or []))
        if wallet_ids:
            sb.table("credit_ledger").delete().in_("wallet_id", wallet_ids).execute()
            sb.table("credit_wallets").delete().in_("id", wallet_ids).execute()
        for u in users.values():
            sb.table("subscriptions").delete().eq("user_id", u["id"]).execute()
            sb.auth.admin.delete_user(u["id"])
        print("cleanup complete")
    except Exception as e:  # noqa: BLE001
        print(f"CLEANUP ERROR (manual sweep needed for suffix {SUFFIX}): {e}")

fails = [r for r in results if r[0] == "FAIL"]
print(f"\n=== QA BOARDS-ON-TEAMS: {len(results) - len(fails)} passed, {len(fails)} failed ===")
for _tag, label, detail in fails:
    print(f"  FAIL: {label} — {detail}")
sys.exit(1 if fails else 0)
