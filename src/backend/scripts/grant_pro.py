"""CLI admin tool for granting/revoking Pro and managing per-user overrides.

Usage:
  poetry run python scripts/grant_pro.py grant <email>
  poetry run python scripts/grant_pro.py revoke <email>
  poetry run python scripts/grant_pro.py override <email> [--max-artists N]
        [--max-projects N] [--max-tasks N]
        [--max-storage-gb N] [--max-split-sheets N]
        [--zoe-enabled] [--no-zoe-enabled]
        [--oneclick-enabled] [--no-oneclick-enabled]
        [--registry-enabled] [--no-registry-enabled]
        [--integrations slack google_drive]
        [--reason "..."] [--expires-days N]
  poetry run python scripts/grant_pro.py clear-override <email>
  poetry run python scripts/grant_pro.py list
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _get_supabase():
    """Lazy supabase client. Patched in tests."""
    import os

    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()
    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SECRET_KEY")
    if not url or not key:
        print("ERROR: VITE_SUPABASE_URL / VITE_SUPABASE_SECRET_KEY not set in .env", file=sys.stderr)
        raise SystemExit(2)
    return create_client(url, key)


def _resolve_user_id(supabase, email: str) -> str | None:
    """Look up a user_id by email via Supabase auth admin API.

    Paginates: keeps fetching pages until either the user is found OR an empty
    (or partial) page. Default Supabase per_page is 50 — without pagination,
    users past page 1 silently won't be found.
    """
    page = 1
    per_page = 50
    while True:
        try:
            users = supabase.auth.admin.list_users(page=page, per_page=per_page)
        except TypeError:
            # Fallback for clients that don't accept kwargs — single call only.
            users = supabase.auth.admin.list_users()
            for u in users:
                if getattr(u, "email", None) == email:
                    return getattr(u, "id", None)
            return None
        except Exception as e:
            print(f"ERROR: failed to list users: {e}", file=sys.stderr)
            # Exit 3 for API/connectivity errors so operators can distinguish
            # them from exit 1 ("user not found") and exit 2 (env vars missing).
            raise SystemExit(3)

        if not users:
            return None
        for u in users:
            if getattr(u, "email", None) == email:
                return getattr(u, "id", None)
        if len(users) < per_page:
            # Last (partial) page reached — no more results.
            return None
        page += 1


_PROPAGATION_NOTE = "NOTE: changes propagate immediately on the next /me/entitlements read (no server cache)."


def _cmd_grant(args) -> int:
    supabase = _get_supabase()
    user_id = _resolve_user_id(supabase, args.email)
    if not user_id:
        print(f"ERROR: user not found for email '{args.email}'", file=sys.stderr)
        return 1
    supabase.table("subscriptions").upsert(
        {
            "user_id": user_id,
            "tier": "pro",
            "status": "active",
            "updated_at": datetime.now(UTC).isoformat(),
        },
        on_conflict="user_id",
    ).execute()
    print(f"Granted Pro to {args.email} (user_id={user_id})")
    print(_PROPAGATION_NOTE)
    return 0


def _cmd_revoke(args) -> int:
    supabase = _get_supabase()
    user_id = _resolve_user_id(supabase, args.email)
    if not user_id:
        print(f"ERROR: user not found for email '{args.email}'", file=sys.stderr)
        return 1
    supabase.table("subscriptions").upsert(
        {
            "user_id": user_id,
            "tier": "free",
            "status": "active",
            "updated_at": datetime.now(UTC).isoformat(),
        },
        on_conflict="user_id",
    ).execute()
    print(f"Revoked Pro from {args.email} (user_id={user_id})")
    print(_PROPAGATION_NOTE)
    return 0


def _cmd_override(args) -> int:
    supabase = _get_supabase()
    user_id = _resolve_user_id(supabase, args.email)
    if not user_id:
        print(f"ERROR: user not found for email '{args.email}'", file=sys.stderr)
        return 1

    # NOTE: granted_by column is intentionally NOT in this payload — it was removed
    # from the schema in Sub-project 1. Audit trail returns with the admin UI in
    # Sub-project 3 when there's a real user identity attached.
    # NOTE: granted_at is reset on every re-apply (i.e., re-issuing an override
    # to add a new field updates the timestamp). This is "most recent change"
    # semantics; if "first granted" is needed for audit, Sub-project 3 (admin UI)
    # will handle that properly when it re-adds the granted_by column.
    payload: dict = {"user_id": user_id, "granted_at": datetime.now(UTC).isoformat()}

    if args.max_artists is not None:
        payload["max_artists"] = args.max_artists
    if args.max_projects is not None:
        payload["max_projects"] = args.max_projects
    if args.max_tasks is not None:
        payload["max_tasks"] = args.max_tasks
    if args.max_storage_gb is not None:
        payload["max_storage_bytes"] = int(args.max_storage_gb * 1024 * 1024 * 1024)
    if args.max_split_sheets is not None:
        payload["max_split_sheets_per_month"] = args.max_split_sheets

    if args.zoe_enabled is True:
        payload["zoe_enabled"] = True
    elif args.zoe_enabled is False:
        payload["zoe_enabled"] = False
    if args.oneclick_enabled is True:
        payload["oneclick_enabled"] = True
    elif args.oneclick_enabled is False:
        payload["oneclick_enabled"] = False
    if args.registry_enabled is True:
        payload["registry_enabled"] = True
    elif args.registry_enabled is False:
        payload["registry_enabled"] = False

    if args.integrations is not None:
        payload["integrations_allowed"] = args.integrations
    if args.reason:
        payload["reason"] = args.reason
    if args.expires_days is not None:
        payload["expires_at"] = (datetime.now(UTC) + timedelta(days=args.expires_days)).isoformat()

    supabase.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()
    print(f"Override applied for {args.email} (user_id={user_id}): {payload}")
    print(_PROPAGATION_NOTE)
    return 0


def _cmd_clear_override(args) -> int:
    supabase = _get_supabase()
    user_id = _resolve_user_id(supabase, args.email)
    if not user_id:
        print(f"ERROR: user not found for email '{args.email}'", file=sys.stderr)
        return 1
    supabase.table("tier_overrides").delete().eq("user_id", user_id).execute()
    print(f"Override cleared for {args.email} (user_id={user_id})")
    print(_PROPAGATION_NOTE)
    return 0


def _cmd_list(args) -> int:
    supabase = _get_supabase()
    pro_subs = supabase.table("subscriptions").select("user_id, tier, status").eq("tier", "pro").execute()
    overrides = supabase.table("tier_overrides").select("*").execute()

    print("Pro subscriptions:")
    for row in pro_subs.data or []:
        print(f"  - user_id={row['user_id']} tier={row['tier']} status={row['status']}")
    print()
    print("Per-user overrides:")
    for row in overrides.data or []:
        print(f"  - user_id={row['user_id']} reason={row.get('reason')!r}")
        for k, v in row.items():
            if k in ("user_id", "reason", "granted_at", "expires_at"):
                continue
            if v is not None:
                print(f"      {k} = {v}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grant_pro", description="Manage user subscription tier and overrides.")
    sp = p.add_subparsers(dest="command", required=True)

    g = sp.add_parser("grant")
    g.add_argument("email")
    g.set_defaults(func=_cmd_grant)

    r = sp.add_parser("revoke")
    r.add_argument("email")
    r.set_defaults(func=_cmd_revoke)

    o = sp.add_parser("override")
    o.add_argument("email")
    o.add_argument("--max-artists", type=int, default=None)
    o.add_argument("--max-projects", type=int, default=None)
    o.add_argument("--max-tasks", type=int, default=None)
    o.add_argument("--max-storage-gb", type=float, default=None)
    o.add_argument("--max-split-sheets", type=int, default=None)
    o.add_argument("--zoe-enabled", dest="zoe_enabled", action="store_true", default=None)
    o.add_argument("--no-zoe-enabled", dest="zoe_enabled", action="store_false")
    o.add_argument("--oneclick-enabled", dest="oneclick_enabled", action="store_true", default=None)
    o.add_argument("--no-oneclick-enabled", dest="oneclick_enabled", action="store_false")
    o.add_argument("--registry-enabled", dest="registry_enabled", action="store_true", default=None)
    o.add_argument("--no-registry-enabled", dest="registry_enabled", action="store_false")
    o.add_argument("--integrations", nargs="+", default=None)
    o.add_argument("--reason", default=None)
    o.add_argument("--expires-days", type=int, default=None)
    o.set_defaults(func=_cmd_override)

    c = sp.add_parser("clear-override")
    c.add_argument("email")
    c.set_defaults(func=_cmd_clear_override)

    listp = sp.add_parser("list")
    listp.set_defaults(func=_cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
