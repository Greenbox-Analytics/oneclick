"""End-to-end QA loop for teams/licensing against the REAL local stack.

Drives the full org lifecycle over HTTP (real JWTs, real RLS, real RPCs)
against a locally-running backend with LICENSING_ENABLED + CREDITS_ENABLED.
Two acts:

ACT 1 (enterprise, admin-created): create org via POST /admin/orgs (the
only way to mint a kind='enterprise' row post-2026-08-16 — see spec
2026-08-15-pricing-tiers-teams) -> activate (purchase-grant path) ->
invite x2 -> claim -> set member caps + operator sets the dispersal ->
transfer artist to the team -> admin grants access -> derived billing
debits the ORG POOL against the member's cap -> cap wall + credit-request
raise.

ACT 2 (self-serve, spec 2026-08-15): a PAYING (Basic) user creates a team
via the now slot-gated POST /orgs -> invite to the team-size limit ->
over-limit invite refused -> fund the pool via personal-reserve transfer
-> a member spends from the pool via ambient billing context -> release
coverage -> simulated grace -> lapse (evaluate_standing) -> reactivate via
coverage claim -> archive -> unarchive -> dissolve-preview -> dissolve.

Creates 8 throwaway auth users (…@example.com) and CLEANS UP EVERY ROW in
a finally block — the DB is shared with prod.

Run (backend must be up on --port with LICENSING_ENABLED=true
CREDITS_ENABLED=true ENTERPRISE_MIN_INITIAL_CREDITS=500
RESEND_API_KEY=qa-disabled POSTHOG_ENABLED=false):

    cd src/backend && poetry run python scripts/qa_licensing_loop.py --port 8017
"""

import os
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace

# Backend modules (subscriptions, orgs) — needed for the two in-process
# steps (activation helper + money path).
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# Flags must be set BEFORE backend imports read them.
ENV_PATH = str(BACKEND_DIR.parents[1] / ".env")
with open(ENV_PATH) as _f:
    for line in _f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ["LICENSING_ENABLED"] = "true"
os.environ["CREDITS_ENABLED"] = "true"
os.environ["ENTERPRISE_MIN_INITIAL_CREDITS"] = "500"
os.environ["POSTHOG_ENABLED"] = "false"

import httpx  # noqa: E402
from supabase import create_client  # noqa: E402

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
org_id2 = None  # self-serve act's org (Act 2)
artist_id = None
project_id = None
wallet_ids = []  # every wallet we touch, for ledger+wallet cleanup


def _create_user(role):
    """Throwaway auth user + signed-in session, registered in `users` for
    both the act steps and the finally-block cleanup."""
    email = f"qa-licensing-{role}-{SUFFIX}@example.com"
    password = secrets.token_urlsafe(16)
    res = sb.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
    uid = res.user.id
    anon = create_client(URL, ANON)
    session = anon.auth.sign_in_with_password({"email": email, "password": password})
    u = {"id": uid, "email": email, "password": password, "token": session.session.access_token}
    users[role] = u
    print(f"       created {role}: {email} ({uid})")
    return u


try:
    # ------------------------------------------------------------- setup --
    for role in ("admin", "owner", "collab"):
        _create_user(role)

    admin, owner, collab = users["admin"], users["owner"], users["collab"]

    # ------------------------------------------- 1. create org (pending) --
    # Enterprise orgs are now created ONLY by a Msanii platform admin via
    # POST /admin/orgs (review r2 hole 7: plain POST /orgs is self-serve +
    # slot-gated as of 20260816). Promote the "admin" test user to a platform
    # admin (profiles.is_admin) so it can call the admin route, and pass its
    # own email as `admin_email` — create_enterprise_org seats that account
    # as the resulting org's customer admin via the auto_create_org_admin
    # trigger, exactly like the old open POST /orgs did for this act.
    sb.table("profiles").update({"is_admin": True}).eq("id", admin["id"]).execute()
    with api(admin["token"]) as c:
        r = c.post("/admin/orgs", json={"name": f"QA Licensing Loop {SUFFIX}", "admin_email": admin["email"]})
        check(
            "1. admin creates org (via /admin/orgs, enterprise)",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:200]}",
        )
        org_id = r.json()["id"]
    org_row = sb.table("organizations").select("status").eq("id", org_id).execute().data[0]
    check("1b. org starts pending", org_row["status"] == "pending", str(org_row))

    # ------------------- 2. activate via the real purchase-grant helper --
    # Same code path the Stripe topup webhook runs (grant kind='purchase'
    # into the pool wallet + cumulative re-check), minus the Stripe envelope.
    from subscriptions.stripe_events import _handle_org_topup_grant

    _handle_org_topup_grant(
        sb,
        admin["id"],
        org_id,
        SimpleNamespace(id=f"qa_{SUFFIX}"),
        "pack_500",
        500,
        1000,
    )
    org_row = sb.table("organizations").select("status").eq("id", org_id).execute().data[0]
    check("2. purchase of 500 >= min 500 activates org", org_row["status"] == "active", str(org_row))
    org_wallet = (
        sb.table("credit_wallets")
        .select("id, reserve_balance")
        .eq("owner_type", "org")
        .eq("owner_id", org_id)
        .execute()
        .data[0]
    )
    wallet_ids.append(org_wallet["id"])
    check("2b. pool wallet holds 500 reserve", org_wallet["reserve_balance"] == 500, str(org_wallet))

    # -------------------------------------------- 3. invite + claim x2 --
    with api(admin["token"]) as c:
        r1 = c.post(f"/orgs/{org_id}/invites", json={"email": owner["email"], "role": "member"})
        r2 = c.post(f"/orgs/{org_id}/invites", json={"email": collab["email"], "role": "member"})
        check(
            "3. admin invites owner+collab",
            r1.status_code == 200 and r2.status_code == 200,
            f"{r1.status_code}/{r2.status_code}: {r1.text[:120]} {r2.text[:120]}",
        )
    invites = sb.table("pending_org_invites").select("token, email").eq("org_id", org_id).execute().data
    tok = {i["email"]: i["token"] for i in invites}
    with api(owner["token"]) as c:
        r = c.post(f"/orgs/invites/{tok[owner['email']]}/accept")
        check("3b. owner claims membership", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    with api(collab["token"]) as c:
        r = c.post(f"/orgs/invites/{tok[collab['email']]}/accept")
        check("3c. collab claims membership", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    members = sb.table("org_members").select("id, user_id, status, email").eq("org_id", org_id).execute().data
    by_user = {m["user_id"]: m for m in members}
    check(
        "3d. 3 active members (admin auto + 2 claims), emails captured",
        len([m for m in members if m["status"] == "active"]) == 3 and by_user[owner["id"]]["email"] == owner["email"],
        str(members),
    )

    # ------------------------------------------- 4. caps, not allocations --
    owner_member_id = by_user[owner["id"]]["id"]
    collab_member_id = by_user[collab["id"]]["id"]
    with api(admin["token"]) as c:
        r = c.put(f"/orgs/{org_id}/members/{owner_member_id}/cap", json={"cap": 50})
        check("4. set owner cap = 50", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        # The collab gets a cap of 0: enough to prove the CAP wall fires while the
        # pool is still funded, which is the wall a dry pool can't produce.
        r = c.put(f"/orgs/{org_id}/members/{collab_member_id}/cap", json={"cap": 0})
        check("4b. set collab cap = 0", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        # default_member_cap is the customer's dial and rides the org update.
        r = c.put(f"/orgs/{org_id}", json={"default_member_cap": 100})
        check("4c. admin sets the default member cap", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        # The DISPERSAL is not theirs to set — proving the hole stays closed.
        r = c.put(f"/orgs/{org_id}/dispersal", json={"monthly_dispersal_credits": 1_000_000})
        check(
            "4d. org admin CANNOT set their own dispersal (no customer route)",
            r.status_code in (404, 405),
            f"{r.status_code}: {r.text[:200]}",
        )
    # Operator sets it directly, the way a signed contract is applied.
    sb.table("organizations").update({"monthly_dispersal_credits": 500}).eq("id", org_id).execute()
    pool_after = (
        sb.table("credit_wallets")
        .select("reserve_balance, bundle_balance")
        .eq("id", org_wallet["id"])
        .execute()
        .data[0]
    )
    check(
        "4d. setting caps moved NO money — the pool is untouched",
        pool_after["reserve_balance"] == 500,
        f"pool={pool_after}",
    )

    # -------------------- 5. owner creates an artist + transfers it to the team --
    artist_id = (
        sb.table("artists")
        .insert({"user_id": owner["id"], "name": f"QA Artist {SUFFIX}", "email": owner["email"]})
        .execute()
        .data[0]["id"]
    )
    project_id = (
        sb.table("projects").insert({"artist_id": artist_id, "name": f"QA Project {SUFFIX}"}).execute().data[0]["id"]
    )
    sb.table("project_members").insert({"project_id": project_id, "user_id": owner["id"], "role": "owner"}).execute()

    with api(collab["token"]) as c:
        r = c.post(f"/orgs/{org_id}/artists/{artist_id}/transfer")
        check(
            "5. non-owner transfer -> 403 (member of the org, but not the artist's owner)",
            r.status_code == 403,
            f"{r.status_code}: {r.text[:200]}",
        )
    with api(owner["token"]) as c:
        r = c.post(f"/orgs/{org_id}/artists/{artist_id}/transfer")
        check("5b. owner transfers the artist to the team", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        r2 = c.post(f"/orgs/{org_id}/artists/{artist_id}/transfer")
        check(
            "5c. re-transfer -> 409 (one-way in v1)",
            r2.status_code == 409 and "already belongs" in r2.json()["detail"],
            f"{r2.status_code}: {r2.text[:200]}",
        )
    with api(admin["token"]) as c:
        r = c.get(f"/orgs/{org_id}/projects")
        listed = r.status_code == 200 and any(
            p.get("projectId") == project_id for p in (r.json().get("projects") or [])
        )
        check("5d. admin console lists the team-owned project", listed, f"{r.status_code}: {r.text[:300]}")

    # --------------------------- 6. admin grants/adjusts/refuses access --
    with api(admin["token"]) as c:
        r = c.put(f"/orgs/{org_id}/projects/{project_id}/members/{collab_member_id}", json={"role": "editor"})
        check("6. admin grants collab editor", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    pm = (
        sb.table("project_members")
        .select("role, org_id")
        .eq("project_id", project_id)
        .eq("user_id", collab["id"])
        .execute()
        .data
    )
    check(
        "6b. project_members row: editor, org_id provenance stamped",
        len(pm) == 1 and pm[0]["role"] == "editor" and pm[0]["org_id"] == org_id,
        str(pm),
    )
    with api(admin["token"]) as c:
        r = c.put(f"/orgs/{org_id}/projects/{project_id}/members/{collab_member_id}", json={"role": "viewer"})
        pm2 = (
            sb.table("project_members")
            .select("role")
            .eq("project_id", project_id)
            .eq("user_id", collab["id"])
            .execute()
            .data
        )
        check(
            "6c. adjust to viewer persists",
            r.status_code == 200 and pm2[0]["role"] == "viewer",
            f"{r.status_code} {pm2}",
        )
        r = c.put(f"/orgs/{org_id}/projects/{project_id}/members/{owner_member_id}", json={"role": "viewer"})
        check("6d. owner is untouchable -> 409", r.status_code == 409, f"{r.status_code}: {r.text[:200]}")

    # ----------------- 7. entitlements + context switcher over HTTP --
    with api(owner["token"]) as c:
        ent = c.get("/me/entitlements").json()
        ctxs = ent.get("availableContexts") or []
        bc = ent.get("billingContext") or {}
        check(
            "7. claim auto-switched owner to org context (spec S5) + both contexts offered",
            any(x.get("orgId") == org_id for x in ctxs if x.get("type") == "org")
            and any(x.get("type") == "personal" for x in ctxs)
            and bc.get("type") == "org"
            and bc.get("orgId") == org_id
            # In org context the credits block is the POOL balance, and
            # monthlyGrant carries the member's CAP (nothing is allocated).
            and (ent.get("credits") or {}).get("balance") == 500
            and (ent.get("credits") or {}).get("memberCap") == 50,
            f"bc={bc} credits={ent.get('credits')} ctxs={str(ctxs)[:160]}",
        )
        r = c.put("/me/billing-context", json={"orgId": None})
        ent2 = c.get("/me/entitlements").json()
        check(
            "7b. switch to personal works",
            r.status_code == 200 and (ent2.get("billingContext") or {}).get("type") == "personal",
            f"{r.status_code} bc={ent2.get('billingContext')}",
        )
        r = c.put("/me/billing-context", json={"orgId": org_id})
        ent3 = c.get("/me/entitlements").json()
        check(
            "7c. switch back to org: enterprise shape + pool balance + member cap",
            r.status_code == 200
            and (ent3.get("billingContext") or {}).get("orgId") == org_id
            and (ent3.get("credits") or {}).get("balance") == 500
            and (ent3.get("credits") or {}).get("memberCap") == 50,
            str(ent3.get("credits"))[:200],
        )
        r = c.put("/me/billing-context", json={"orgId": None})
        check("7d. end in personal context (step 8 tests derivation FROM personal)", r.status_code == 200, r.text[:120])

    # --------- 8. money: derived billing debits the POOL against the cap --
    from subscriptions.models import CreditAction, CreditGrant
    from subscriptions.service import EntitlementsService

    svc = EntitlementsService(sb)
    res = svc.check_credits(owner["id"], CreditAction.ZOE_MESSAGE, resource_project_id=project_id)
    check(
        "8. owner personal-context check derives to the ORG POOL",
        res.allowed and res.wallet_id == org_wallet["id"] and res.managed_by_org and res.org_member_id,
        f"allowed={res.allowed} wallet={res.wallet_id} pool={org_wallet['id']} member={res.org_member_id}",
    )
    grant = CreditGrant(
        request_id=f"qa-debit-{SUFFIX}",
        action=str(CreditAction.ZOE_MESSAGE),
        price=res.price,
        kind="debit",
        enabled=True,
        wallet_id=res.wallet_id,
        org_member_id=res.org_member_id,
    )
    svc.debit_for_action(owner["id"], grant)
    pool_post = (
        sb.table("credit_wallets")
        .select("reserve_balance, bundle_balance")
        .eq("id", org_wallet["id"])
        .execute()
        .data[0]
    )
    check(
        "8b. debit lands on the pool: 500 -> 497 (zoe=3cr)",
        pool_post["reserve_balance"] == 497,
        str(pool_post),
    )
    member_post = sb.table("org_members").select("cap_used, cap_period_end").eq("id", owner_member_id).execute().data[0]
    check(
        "8c. the SAME transaction moved the member's cap counter to 3",
        member_post["cap_used"] == 3,
        str(member_post),
    )
    personal_w = (
        sb.table("credit_wallets")
        .select("id, reserve_balance, bundle_balance")
        .eq("owner_type", "user")
        .eq("owner_id", owner["id"])
        .execute()
        .data
    )
    if personal_w:
        wallet_ids.append(personal_w[0]["id"])
    ledger = (
        sb.table("credit_ledger").select("wallet_id, metadata").eq("request_id", f"qa-debit-{SUFFIX}").execute().data
    )
    check(
        "8d. ledger row targets the pool and attributes the member",
        ledger
        and ledger[0]["wallet_id"] == org_wallet["id"]
        and (ledger[0]["metadata"] or {}).get("org_member_id") == owner_member_id,
        str(ledger),
    )

    # The CAP wall (collab is capped at 0 while the pool still holds 497) —
    # distinct from a dry pool, and the reason cap_reached exists.
    res_c = svc.check_credits(collab["id"], CreditAction.ZOE_MESSAGE, resource_project_id=project_id)
    check(
        "8e. collab cap wall: cap_reached, managedByOrg, NOT ownerCanUnlink",
        not res_c.allowed and res_c.cap_reached and res_c.managed_by_org and not res_c.owner_can_unlink,
        f"allowed={res_c.allowed} cap_reached={res_c.cap_reached} ocu={res_c.owner_can_unlink}",
    )

    # A member asks for a raise; the admin approves; the wall clears — with no
    # money moving at any point.
    with api(collab["token"]) as c:
        r = c.post(f"/orgs/{org_id}/credit-requests", json={"requested_cap": 100, "note": "qa"})
        check("8f. collab requests a higher cap", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        request_id = r.json()["id"]
    with api(admin["token"]) as c:
        r = c.post(f"/orgs/{org_id}/credit-requests/{request_id}/approve", json={"cap": 100})
        check("8g. admin approves the raise", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    res_c2 = svc.check_credits(collab["id"], CreditAction.ZOE_MESSAGE, resource_project_id=project_id)
    check("8h. raise clears the cap wall", res_c2.allowed and res_c2.managed_by_org, f"allowed={res_c2.allowed}")

    # =====================================================================
    # ACT 2: self-serve team lifecycle (spec 2026-08-15) — a PAYING (Basic)
    # user creates via POST /orgs (now slot-gated self-serve, not the
    # enterprise path Act 1 exercises) -> invites to the team-size limit ->
    # an over-limit invite is refused -> funds the pool via personal-reserve
    # transfer -> a member spends from the pool via ambient billing context
    # -> releases coverage -> grace/lapse is simulated (direct write +
    # evaluate_standing, not a 14-day wait) -> reactivates by re-claiming
    # coverage -> archive -> unarchive -> dissolve-preview -> dissolve.
    # =====================================================================
    from orgs.wallets import read_or_create_user_wallet

    for role in ("ss_owner", "ss_m1", "ss_m2", "ss_m3", "ss_m4"):
        _create_user(role)
    ss_owner = users["ss_owner"]
    ss_members = [users["ss_m1"], users["ss_m2"], users["ss_m3"]]  # fills the Basic limit (3, excl. owner)
    ss_overflow = users["ss_m4"]  # the 4th — must be refused

    # A real subscription row makes ss_owner a PAYING Basic user: 1 team
    # slot, 3 members excl. the owner, 100 GiB pool (20260817000001 dials) —
    # "as a paying user" per the task, not an admin-implicit-Pro shortcut.
    sb.table("subscriptions").upsert(
        {"user_id": ss_owner["id"], "tier": "basic", "status": "active"}, on_conflict="user_id"
    ).execute()

    org2_name = f"QA Self-Serve {SUFFIX}"
    with api(ss_owner["token"]) as c:
        r = c.post("/orgs", json={"name": org2_name})
        check(
            "9. paying user creates a self-serve org via POST /orgs",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:200]}",
        )
        org_id2 = r.json()["id"]
    org2_row = sb.table("organizations").select("kind, status, covered_by").eq("id", org_id2).execute().data[0]
    check(
        "9b. self-serve org is born active, covered by its creator (no activation floor)",
        org2_row["kind"] == "self_serve"
        and org2_row["status"] == "active"
        and org2_row["covered_by"] == ss_owner["id"],
        str(org2_row),
    )

    # ------------------------------------------- 10. invite to the limit --
    with api(ss_owner["token"]) as c:
        for m in ss_members:
            r = c.post(f"/orgs/{org_id2}/invites", json={"email": m["email"], "role": "member"})
            check(f"10. invite {m['email']}", r.status_code == 200, f"{r.status_code}: {r.text[:150]}")
    invites2 = sb.table("pending_org_invites").select("token, email").eq("org_id", org_id2).execute().data
    tok2 = {i["email"]: i["token"] for i in invites2}
    for m in ss_members:
        with api(m["token"]) as c:
            r = c.post(f"/orgs/invites/{tok2[m['email']]}/accept")
            check(f"10b. {m['email']} accepts", r.status_code == 200, f"{r.status_code}: {r.text[:150]}")
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/invites", json={"email": ss_overflow["email"], "role": "member"})
        check(
            "10c. a 4th invite past the Basic team-size limit (3, excl. owner) is refused",
            r.status_code == 402,
            f"{r.status_code}: {r.text[:200]}",
        )

    # -------------------------------------- 11. fund the pool by transfer --
    ss_personal_wallet = read_or_create_user_wallet(sb, ss_owner["id"])
    wallet_ids.append(ss_personal_wallet["id"])
    sb.rpc(
        "grant_credits",
        {
            "p_wallet_id": ss_personal_wallet["id"],
            "p_amount": 200,
            "p_kind": "admin_grant",
            "p_bucket": "reserve",
            "p_metadata": {"reason": "qa_self_serve_seed"},
            "p_request_id": f"qa-ss-seed-{SUFFIX}",
        },
    ).execute()
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/transfer-credits", json={"amount": 150})
        check(
            "11. owner transfers 150 personal reserve credits to the pool",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:200]}",
        )
    ss_pool_wallet = (
        sb.table("credit_wallets")
        .select("id, reserve_balance")
        .eq("owner_type", "org")
        .eq("owner_id", org_id2)
        .execute()
        .data[0]
    )
    wallet_ids.append(ss_pool_wallet["id"])
    check("11b. pool holds the transferred 150 reserve", ss_pool_wallet["reserve_balance"] == 150, str(ss_pool_wallet))

    # ---------------------- 12. a member spends from the pool (ambient) --
    m1 = ss_members[0]
    with api(m1["token"]) as c:
        r = c.put("/me/billing-context", json={"orgId": org_id2})
        check(
            "12. member points ambient billing context at the team",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:150]}",
        )
    res_m1 = svc.check_credits(m1["id"], CreditAction.ZOE_MESSAGE)
    check(
        "12b. ambient context derives the check to the pool wallet",
        res_m1.allowed and res_m1.wallet_id == ss_pool_wallet["id"] and res_m1.managed_by_org,
        f"allowed={res_m1.allowed} wallet={res_m1.wallet_id} pool={ss_pool_wallet['id']}",
    )
    grant2 = CreditGrant(
        request_id=f"qa-ss-debit-{SUFFIX}",
        action=str(CreditAction.ZOE_MESSAGE),
        price=res_m1.price,
        kind="debit",
        enabled=True,
        wallet_id=res_m1.wallet_id,
        org_member_id=res_m1.org_member_id,
    )
    svc.debit_for_action(m1["id"], grant2)
    ss_pool_post = sb.table("credit_wallets").select("reserve_balance").eq("id", ss_pool_wallet["id"]).execute().data[0]
    check(
        "12c. member spend lands on the pool: 150 -> 147 (zoe=3cr)",
        ss_pool_post["reserve_balance"] == 147,
        str(ss_pool_post),
    )

    # ------------------------------------------- 13. release coverage --
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/coverage/release")
        check("13. owner releases coverage", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    org2_row = sb.table("organizations").select("covered_by, covered_at").eq("id", org_id2).execute().data[0]
    check(
        "13b. covered_by stays (last-coverer attribution), covered_at clears",
        org2_row["covered_by"] == ss_owner["id"] and org2_row["covered_at"] is None,
        str(org2_row),
    )

    # ------------------------- 14. simulated grace -> lapse via the sweep --
    from datetime import UTC, datetime, timedelta

    from orgs.standing import evaluate_standing, grace_days

    # Direct write instead of waiting ORG_GRACE_DAYS: back-date grace as if
    # the sweep had already started it well past the grace window, then let
    # evaluate_standing (the ONE writer of 'lapsed') do the actual flip.
    stale = (datetime.now(UTC) - timedelta(days=grace_days() + 1)).isoformat()
    sb.table("organizations").update({"grace_started_at": stale}).eq("id", org_id2).execute()
    evaluate_standing(sb)
    org2_row = sb.table("organizations").select("status").eq("id", org_id2).execute().data[0]
    check("14. sweep lapses an uncovered org past its grace window", org2_row["status"] == "lapsed", str(org2_row))

    # ------------------------------------------- 15. reactivate via claim --
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/coverage/claim")
        check(
            "15. owner reclaims coverage on a lapsed org -> reactivates",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:200]}",
        )
    org2_row = sb.table("organizations").select("status, covered_at").eq("id", org_id2).execute().data[0]
    check(
        "15b. org is active again, covered_at restamped",
        org2_row["status"] == "active" and org2_row["covered_at"],
        str(org2_row),
    )

    # ------------------------------------------- 16. archive / unarchive --
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/archive")
        check("16. owner archives the team", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
    org2_row = sb.table("organizations").select("archived_at").eq("id", org_id2).execute().data[0]
    check("16b. archived_at stamped", org2_row["archived_at"] is not None, str(org2_row))

    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/unarchive")
        check(
            "17. owner unarchives (free slot + reactivation storage guard both pass)",
            r.status_code == 200,
            f"{r.status_code}: {r.text[:200]}",
        )
    org2_row = sb.table("organizations").select("archived_at, status").eq("id", org_id2).execute().data[0]
    check(
        "17b. archived_at cleared, active again",
        org2_row["archived_at"] is None and org2_row["status"] == "active",
        str(org2_row),
    )

    # ------------------------------------- 18. dissolve-preview + dissolve --
    with api(ss_owner["token"]) as c:
        r = c.get(f"/orgs/{org_id2}/dissolve-preview")
        preview = r.json() if r.status_code == 200 else {}
        check(
            "18. dissolve-preview lists the forfeited reserve and member count",
            r.status_code == 200 and preview.get("forfeitReserve") == 147 and preview.get("memberCount") == 4,
            f"{r.status_code}: {str(preview)[:250]}",
        )
    with api(ss_owner["token"]) as c:
        r = c.post(f"/orgs/{org_id2}/dissolve", json={"confirm_name": "the wrong name"})
        check(
            "19. dissolve refuses a mismatched confirm name", r.status_code == 400, f"{r.status_code}: {r.text[:150]}"
        )
        r = c.post(f"/orgs/{org_id2}/dissolve", json={"confirm_name": org2_name})
        check("19b. dissolve with the correct name succeeds", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        r = c.post(f"/orgs/{org_id2}/dissolve", json={"confirm_name": org2_name})
        check(
            "19c. re-dissolve is idempotent (already: true, nothing double-moves)",
            r.status_code == 200 and r.json().get("already") is True,
            f"{r.status_code}: {r.text[:150]}",
        )
    org2_row = sb.table("organizations").select("dissolved_at").eq("id", org_id2).execute().data[0]
    check(
        "19d. dissolved_at stamped (soft dissolve — org row retained)",
        org2_row["dissolved_at"] is not None,
        str(org2_row),
    )

finally:
    # ------------------------------------------------------- cleanup --
    print("\n--- cleanup (shared DB) ---")
    try:
        for uid in [u["id"] for u in users.values()]:
            w = sb.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", uid).execute().data
            wallet_ids.extend(x["id"] for x in w)
        wallet_ids = list(set(wallet_ids))
        if wallet_ids:
            sb.table("credit_ledger").delete().in_("wallet_id", wallet_ids).execute()
            sb.table("credit_wallets").delete().in_("id", wallet_ids).execute()
        if project_id:
            # Delete the PROJECT (cascades project_members) — a direct delete of
            # the owner row trips prevent_owner_deletion; the cascade is exempt.
            sb.table("projects").delete().eq("id", project_id).execute()
        if artist_id:
            # Detach first: artists.team_id is ON DELETE RESTRICT, so a
            # team-owned artist would block the org teardown below.
            sb.table("artists").update({"team_id": None}).eq("id", artist_id).execute()
            sb.table("artists").delete().eq("id", artist_id).execute()
        if org_id:
            sb.table("credit_requests").delete().eq("org_id", org_id).execute()
            sb.table("pending_org_invites").delete().eq("org_id", org_id).execute()
        if org_id2:
            # dissolve_org expires open invites itself; delete is a harmless no-op
            # if none remain. Soft-dissolve retains the org row by design — this
            # is cleanup of a throwaway QA row in a shared DB, not a product path.
            sb.table("pending_org_invites").delete().eq("org_id", org_id2).execute()
        # Users BEFORE org_members: a direct delete of the sole admin's row
        # trips the last-admin guard; the auth-user deletion CASCADE is the
        # sanctioned escape (guard yields to it and auto-archives the org).
        for u in users.values():
            sb.table("subscriptions").delete().eq("user_id", u["id"]).execute()
            sb.auth.admin.delete_user(u["id"])
        if org_id:
            sb.table("org_members").delete().eq("org_id", org_id).execute()
            sb.table("organizations").delete().eq("id", org_id).execute()
        if org_id2:
            sb.table("org_members").delete().eq("org_id", org_id2).execute()
            sb.table("organizations").delete().eq("id", org_id2).execute()
        print("cleanup complete")
    except Exception as e:  # noqa: BLE001
        print(f"CLEANUP ERROR (manual sweep needed for suffix {SUFFIX}): {e}")

fails = [r for r in results if r[0] == "FAIL"]
print(f"\n=== QA LOOP: {len(results) - len(fails)} passed, {len(fails)} failed ===")
for _tag, label, detail in fails:
    print(f"  FAIL: {label} — {detail}")
sys.exit(1 if fails else 0)
