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
        self.host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.project_id)

    async def query(self, hogql: str) -> dict:
        """Run a HogQL query and return the parsed JSON response."""
        url = f"{self.host}/api/projects/{self.project_id}/query"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={"query": {"kind": "HogQLQuery", "query": hogql}},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
