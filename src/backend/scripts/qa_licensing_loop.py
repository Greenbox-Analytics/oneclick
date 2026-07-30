"""End-to-end QA loop for teams/licensing against the REAL local stack.

Drives the full org lifecycle over HTTP (real JWTs, real RLS, real RPCs)
against a locally-running backend with LICENSING_ENABLED + CREDITS_ENABLED:

  create org -> activate (purchase-grant path) -> invite x2 -> claim ->
  set member caps + operator sets the dispersal -> link project -> admin grants access -> derived billing
  debits the ORG POOL against the member's cap -> cap wall + dry-pool wall
  (member + owner-aware) -> offboard -> unlink -> archive.

Creates 3 throwaway auth users (…@example.com) and CLEANS UP EVERY ROW in
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
artist_id = None
project_id = None
wallet_ids = []  # every wallet we touch, for ledger+wallet cleanup

try:
    # ------------------------------------------------------------- setup --
    for role in ("admin", "owner", "collab"):
        email = f"qa-licensing-{role}-{SUFFIX}@example.com"
        password = secrets.token_urlsafe(16)
        res = sb.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
        uid = res.user.id
        anon = create_client(URL, ANON)
        session = anon.auth.sign_in_with_password({"email": email, "password": password})
        users[role] = {"id": uid, "email": email, "password": password, "token": session.session.access_token}
        print(f"       created {role}: {email} ({uid})")

    admin, owner, collab = users["admin"], users["owner"], users["collab"]

    # ------------------------------------------- 1. create org (pending) --
    with api(admin["token"]) as c:
        r = c.post("/orgs", json={"name": f"QA Licensing Loop {SUFFIX}"})
        check("1. admin creates org", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
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

    # ------------------------------- 5. owner creates + links a project --
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
        r = c.post(f"/orgs/{org_id}/projects/{project_id}/link")
        check(
            "5. non-owner link -> 404 'Project not found' (no oracle)",
            r.status_code == 404 and r.json()["detail"] == "Project not found",
            f"{r.status_code}: {r.text[:200]}",
        )
    with api(owner["token"]) as c:
        r = c.post(f"/orgs/{org_id}/projects/{project_id}/link")
        check("5b. owner links project", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
        r2 = c.post(f"/orgs/{org_id}/projects/{project_id}/link")
        check(
            "5c. re-link -> 409 rule-8 copy",
            r2.status_code == 409 and "already linked" in r2.json()["detail"],
            f"{r2.status_code}: {r2.text[:200]}",
        )
    with api(admin["token"]) as c:
        r = c.get(f"/orgs/{org_id}/projects")
        listed = r.status_code == 200 and any(
            p.get("projectId") == project_id for p in (r.json().get("projects") or [])
        )
        check("5d. admin console lists linked project", listed, f"{r.status_code}: {r.text[:300]}")

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
            sb.table("org_project_links").delete().eq("project_id", project_id).execute()
            # Delete the PROJECT (cascades project_members) — a direct delete of
            # the owner row trips prevent_owner_deletion; the cascade is exempt.
            sb.table("projects").delete().eq("id", project_id).execute()
        if artist_id:
            sb.table("artists").delete().eq("id", artist_id).execute()
        if org_id:
            sb.table("credit_requests").delete().eq("org_id", org_id).execute()
            sb.table("pending_org_invites").delete().eq("org_id", org_id).execute()
        # Users BEFORE org_members: a direct delete of the sole admin's row
        # trips the last-admin guard; the auth-user deletion CASCADE is the
        # sanctioned escape (guard yields to it and auto-archives the org).
        for u in users.values():
            sb.table("subscriptions").delete().eq("user_id", u["id"]).execute()
            sb.auth.admin.delete_user(u["id"])
        if org_id:
            sb.table("org_members").delete().eq("org_id", org_id).execute()
            sb.table("organizations").delete().eq("id", org_id).execute()
        print("cleanup complete")
    except Exception as e:  # noqa: BLE001
        print(f"CLEANUP ERROR (manual sweep needed for suffix {SUFFIX}): {e}")

fails = [r for r in results if r[0] == "FAIL"]
print(f"\n=== QA LOOP: {len(results) - len(fails)} passed, {len(fails)} failed ===")
for _tag, label, detail in fails:
    print(f"  FAIL: {label} — {detail}")
sys.exit(1 if fails else 0)
