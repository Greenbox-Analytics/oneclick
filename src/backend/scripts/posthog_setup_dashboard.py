"""Idempotent setup script for the Msanii PostHog dashboard.

Usage:
    poetry run python -m scripts.posthog_setup_dashboard [--adopt] [--env=dev|prod]

State file: scripts/.posthog_dashboard_state.{env}.json
  Maps logical names → PostHog entity IDs. Survives renames in the PostHog UI.
  On --adopt, discovers existing entities by name and seeds the state file.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

STATE_DIR = Path(__file__).resolve().parent


def _state_path(env: str) -> Path:
    return STATE_DIR / f".posthog_dashboard_state.{env}.json"


def _load_state(env: str) -> dict:
    p = _state_path(env)
    if p.exists():
        return json.loads(p.read_text())
    return {"cohorts": {}, "insights": {}, "dashboards": {}, "dashboard_tiles": {}}


def _save_state(env: str, state: dict) -> None:
    _state_path(env).write_text(json.dumps(state, indent=2))


# Cohort definitions — logical_name → spec.
# Dropped "active_testers_7d" from v1 — its filter was identical to "testers"
# without the "performed event in last N days" predicate.
COHORTS = {
    "testers": {
        "name": "Testers",
        "filters": {
            "properties": {
                "type": "AND",
                "values": [
                    {
                        "key": "is_tester",
                        "value": ["true"],
                        "operator": "exact",
                        "type": "person",
                    }
                ],
            }
        },
    },
    "pro_users": {
        "name": "Pro users",
        "filters": {
            "properties": {
                "type": "AND",
                "values": [
                    {
                        "key": "plan",
                        "value": ["pro"],
                        "operator": "exact",
                        "type": "person",
                    }
                ],
            }
        },
    },
    "free_users": {
        "name": "Free users",
        "filters": {
            "properties": {
                "type": "AND",
                "values": [
                    {
                        "key": "plan",
                        "value": ["free"],
                        "operator": "exact",
                        "type": "person",
                    }
                ],
            }
        },
    },
}

# Insight definitions (12 tiles per dashboard).
# These are simplified spec stubs; PostHog's insight API may demand richer
# filter trees on first save — adjust at run-time per actual API responses.
INSIGHTS = {
    "dau_by_tool": {
        "name": "DAU by tool",
        "kind": "trends",
        "events": [{"id": "tool_opened"}],
        "breakdown": "tool",
    },
    "wau_total": {
        "name": "WAU total",
        "kind": "trends",
        "events": [{"id": "tool_opened"}],
    },
    "tool_engagement": {
        "name": "Tool engagement",
        "kind": "trends",
        "events": [{"id": "tool_used"}],
        "breakdown": "tool",
    },
    "tool_retention": {"name": "Tool retention", "kind": "retention"},
    "oneclick_funnel": {
        "name": "OneClick funnel",
        "kind": "funnels",
        "funnel_window_interval": 1,
        "funnel_window_interval_unit": "hour",
        "events": [
            {
                "id": "tool_opened",
                "properties": [{"key": "tool", "value": "oneclick"}],
            },
            {"id": "oneclick_contract_selected"},
            {"id": "oneclick_calc_started"},
            {"id": "oneclick_calc_completed"},
        ],
    },
    "zoe_funnel": {
        "name": "Zoe funnel",
        "kind": "funnels",
        "funnel_window_interval": 1,
        "funnel_window_interval_unit": "hour",
        "events": [
            {"id": "tool_opened", "properties": [{"key": "tool", "value": "zoe"}]},
            {"id": "zoe_query_submitted"},
            {"id": "zoe_response_received"},
        ],
    },
    "splitsheet_funnel": {
        "name": "SplitSheet funnel",
        "kind": "funnels",
        "funnel_window_interval": 1,
        "funnel_window_interval_unit": "hour",
        "events": [
            {
                "id": "tool_opened",
                "properties": [{"key": "tool", "value": "splitsheet"}],
            },
            {"id": "splitsheet_form_started"},
            {"id": "splitsheet_generated"},
        ],
    },
    "registry_funnel": {
        "name": "Registry funnel",
        "kind": "funnels",
        "funnel_window_interval": 7,
        "funnel_window_interval_unit": "day",
        "events": [
            {
                "id": "tool_opened",
                "properties": [{"key": "tool", "value": "registry"}],
            },
            {"id": "work_created"},
            {"id": "work_submitted_for_registration"},
            {"id": "registry_work_registered"},
        ],
    },
    "errors_by_tool": {
        "name": "Errors by tool",
        "kind": "trends",
        "events": [
            {"id": "oneclick_calc_failed"},
            {"id": "zoe_query_failed"},
            {"id": "splitsheet_generation_failed"},
            {"id": "integration_connect_failed"},
        ],
        "breakdown": "tool",
    },
    "stickiness": {
        "name": "Stickiness",
        "kind": "stickiness",
        "events": [{"id": "tool_used"}],
    },
    "integration_adoption": {
        "name": "Integration adoption",
        "kind": "trends",
        "events": [{"id": "integration_connected"}],
        "breakdown": "tool",
    },
    "integration_usage": {
        "name": "Integration usage",
        "kind": "trends",
        "events": [{"id": "integration_used"}],
        "breakdown": "tool",
    },
}

DASHBOARDS = {
    "testers": {"name": "Msanii — Tool Usage (Testers)", "cohort": "testers"},
    "all_users": {"name": "Msanii — Tool Usage (All Users)", "cohort": None},
}


def _api(host: str, project_id: str, key: str) -> dict:
    return {
        "base_url": f"{host}/api/projects/{project_id}",
        "headers": {"Authorization": f"Bearer {key}"},
    }


def _upsert(
    client: httpx.Client,
    api: dict,
    resource: str,
    state_section: dict,
    logical_name: str,
    spec: dict,
) -> str:
    """Create or PATCH a PostHog entity. Returns its ID. Updates state in-place."""
    existing_id = state_section.get(logical_name)
    if existing_id:
        url = f"{api['base_url']}/{resource}/{existing_id}/"
        r = client.patch(url, json=spec, headers=api["headers"])
        if r.status_code == 404:
            existing_id = None  # entity was deleted; fall through to create
        else:
            r.raise_for_status()
            return existing_id
    # Create
    url = f"{api['base_url']}/{resource}/"
    r = client.post(url, json=spec, headers=api["headers"])
    r.raise_for_status()
    new_id = r.json()["id"]
    state_section[logical_name] = new_id
    return new_id


def _adopt_by_name(client: httpx.Client, api: dict, resource: str, name: str) -> str | None:
    url = f"{api['base_url']}/{resource}/"
    r = client.get(url, params={"search": name}, headers=api["headers"])
    r.raise_for_status()
    for row in r.json().get("results", []):
        if row.get("name") == name:
            return row["id"]
    return None


def run(env: str = "dev", adopt: bool = False) -> None:
    key = os.getenv("POSTHOG_PERSONAL_API_KEY")
    project_id = os.getenv("POSTHOG_PROJECT_ID")
    host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
    if not key or not project_id:
        print(
            "ERROR: POSTHOG_PERSONAL_API_KEY and POSTHOG_PROJECT_ID must be set",
            file=sys.stderr,
        )
        sys.exit(2)

    api = _api(host, project_id, key)
    state = _load_state(env)
    state.setdefault("dashboard_tiles", {})

    with httpx.Client(timeout=60.0) as client:
        # Adopt path: discover existing entities by name and exit early.
        if adopt:
            for logical, spec in COHORTS.items():
                if logical not in state["cohorts"]:
                    found = _adopt_by_name(client, api, "cohorts", spec["name"])
                    if found:
                        state["cohorts"][logical] = found
            for logical, spec in INSIGHTS.items():
                if logical not in state["insights"]:
                    found = _adopt_by_name(client, api, "insights", spec["name"])
                    if found:
                        state["insights"][logical] = found
            for logical, spec in DASHBOARDS.items():
                if logical not in state["dashboards"]:
                    found = _adopt_by_name(client, api, "dashboards", spec["name"])
                    if found:
                        state["dashboards"][logical] = found
            _save_state(env, state)
            print(f"Adopted entities written to {_state_path(env)}. Re-run without --adopt to PATCH.")
            return

        # Pass 1: Cohorts
        for logical, spec in COHORTS.items():
            _upsert(client, api, "cohorts", state["cohorts"], logical, spec)

        # Pass 2: Insights
        for logical, spec in INSIGHTS.items():
            _upsert(client, api, "insights", state["insights"], logical, spec)

        # Pass 3: Dashboards (shells)
        for logical, spec in DASHBOARDS.items():
            _upsert(
                client,
                api,
                "dashboards",
                state["dashboards"],
                logical,
                {"name": spec["name"]},
            )

        # Pass 4: Tile mounting — bind every insight to both dashboards.
        # Without this, dashboards render empty even after Pass 3 succeeds.
        for dash_logical, dash_id in state["dashboards"].items():
            mounted = state["dashboard_tiles"].setdefault(dash_logical, {})
            for insight_logical, insight_id in state["insights"].items():
                if insight_logical in mounted:
                    continue  # already bound
                url = f"{api['base_url']}/dashboards/{dash_id}/tiles/"
                r = client.post(url, json={"insight": insight_id}, headers=api["headers"])
                if r.status_code in (200, 201):
                    mounted[insight_logical] = r.json().get("id")
                elif r.status_code in (400, 409):
                    mounted[insight_logical] = "already-bound"
                else:
                    r.raise_for_status()

        _save_state(env, state)

    print("\n✓ Setup complete.")
    for logical, dash_id in state["dashboards"].items():
        print(f"  {DASHBOARDS[logical]['name']}: {host}/project/{project_id}/dashboard/{dash_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="Discover existing entities by name; write state file; exit.",
    )
    parser.add_argument(
        "--env",
        default="dev",
        choices=["dev", "prod"],
        help="State file suffix.",
    )
    args = parser.parse_args()
    run(env=args.env, adopt=args.adopt)


if __name__ == "__main__":
    main()
