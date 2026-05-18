"""PostHog analytics wrapper.

All capture/identify calls are no-ops when POSTHOG_ENABLED is not "true",
so dev/test environments don't pollute the analytics project. Tests can
force this by setting POSTHOG_ENABLED=false in conftest (or just rely on
the env-var being unset).
"""

import os

import posthog

_initialized = False


def _set_posthog_config() -> None:
    """Set posthog module-level config. Separated so tests can patch."""
    posthog.api_key = os.environ["POSTHOG_PROJECT_TOKEN"]
    posthog.host = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")


def init_analytics() -> None:
    """Initialize PostHog. Idempotent. No-op when POSTHOG_ENABLED != "true"."""
    global _initialized
    if _initialized:
        return
    if os.environ.get("POSTHOG_ENABLED", "false").lower() != "true":
        return
    _set_posthog_config()
    _initialized = True


def capture(distinct_id: str, event: str, properties: dict | None = None) -> None:
    """Send a capture event. No-op if not initialized."""
    if not _initialized:
        return
    # APP_ENV tags every event so the PostHog dashboard can separate dev/prod
    # traffic. Local runs default to "local" and are filtered out at the dashboard.
    merged = {**(properties or {}), "environment": os.environ.get("APP_ENV", "local")}
    posthog.capture(distinct_id=distinct_id, event=event, properties=merged)


def identify(distinct_id: str, properties: dict) -> None:
    """Send an identify call. No-op if not initialized."""
    if not _initialized:
        return
    posthog.identify(distinct_id=distinct_id, properties=properties)
