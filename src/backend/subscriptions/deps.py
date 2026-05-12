"""Module-level singleton + accessor for EntitlementsService.

Owned by the subscriptions package (NOT main.py) so that downstream modules
in this package — enforcement.py, admin_*, pro_requests_router — can depend
on it without a circular `from main import …`. main.py re-imports the
accessor for backwards compat with SP1/SP2 callers.
"""

from subscriptions.service import EntitlementsService

_entitlements_service: EntitlementsService | None = None


def _get_entitlements_service() -> EntitlementsService:
    """Return a process-wide EntitlementsService bound to the lazy Supabase client.

    Lazy-built on first call so importing this module does not require Supabase
    env vars to be set (matters for tests + tooling that imports without going
    through FastAPI startup).
    """
    global _entitlements_service
    if _entitlements_service is None:
        # Local import to avoid pulling main.py into the import graph
        from main import get_supabase_client

        _entitlements_service = EntitlementsService(get_supabase_client())
    return _entitlements_service
