"""Thin client for PostHog HogQL queries.

Uses the personal API key (read scope) — separate from the project write
key used for event capture.
"""

import os

import httpx


class PostHogClient:
    def __init__(self):
        self.api_key = os.getenv("POSTHOG_PERSONAL_API_KEY")
        self.project_id = os.getenv("POSTHOG_PROJECT_ID")
        # POSTHOG_HOST is the *ingest* host (e.g. us.i.posthog.com), used by
        # /capture and /identify. The REST/HogQL API lives on the *app* host
        # (us.posthog.com / eu.posthog.com) — calling /api/projects/.../query
        # against the ingest host returns 404. Tracked separately so we can
        # share POSTHOG_HOST with the ingest SDK without breaking queries.
        self.ingest_host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")

    @property
    def api_host(self) -> str:
        """App/REST host used for HogQL queries.

        Resolution order: POSTHOG_API_HOST override → derive from ingest host
        by stripping the `.i.` segment (us.i.posthog.com → us.posthog.com).
        """
        override = os.getenv("POSTHOG_API_HOST")
        if override:
            return override
        return self.ingest_host.replace(".i.posthog.com", ".posthog.com")

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.project_id)

    async def query(self, hogql: str) -> dict:
        """Run a HogQL query and return the parsed JSON response."""
        url = f"{self.api_host}/api/projects/{self.project_id}/query"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={"query": {"kind": "HogQLQuery", "query": hogql}},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
