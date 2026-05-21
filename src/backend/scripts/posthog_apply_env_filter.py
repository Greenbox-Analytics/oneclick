"""One-time backfill: apply env + date filter to insights on existing PostHog dashboards.

Walks insights on the given dashboards via the PostHog REST API and PATCHes each
filter-based insight's filter tree to include:
  - properties.environment IN ('dev', 'prod')  (the load-bearing exclusion)
  - date_from = POSTHOG_DATA_CUTOFF            (a default; dashboard date pickers
                                                 override this in the UI)

Idempotent: an insight is considered fully patched only when BOTH the env
predicate is present AND date_from equals the cutoff. Half-patched insights
(e.g. env added but date missing from a prior partial failure) get re-PATCHed
to complete them.

Query-based insights (insight.query non-null) use a different schema where
filter trees live under `query.source.properties`. PATCHing `filters` on those
would be a silent no-op, so the script logs a WARNING and skips them. Migrating
query-based insights is out of scope for this backfill.

Top-level dedup only: if a manually-edited insight has a nested AND/OR group
containing its own `environment` predicate, this script will not detect it and
will add a second top-level predicate. Our existing dashboards don't have such
shapes, but worth knowing for future hand-edited insights.

Usage:
    poetry run python -m scripts.posthog_apply_env_filter \\
        --dashboard-id 1593101 --dashboard-id 1597175 --dry-run
    poetry run python -m scripts.posthog_apply_env_filter \\
        --dashboard-id 1593101 --dashboard-id 1597175
"""

import argparse
import json
import os
import sys

import httpx

from analytics import ENV_FILTER_VALUES, POSTHOG_DATA_CUTOFF


def _mutate_filters(existing: dict) -> dict:
    """Return a new filters dict with env predicate added and date_from set.

    Normalizes properties to {type: AND, values: [...]} shape. Idempotent: any
    prior top-level `environment` predicate is removed before re-adding so we
    never duplicate.
    """
    out = {**existing, "date_from": POSTHOG_DATA_CUTOFF}

    raw_props = out.get("properties")
    # PostHog accepts both legacy list-shape and tree-shape for properties.
    # Normalize to tree. Use `or []` not default=[] — PostHog responses can have
    # explicit nulls where `get("values", [])` returns None and `list(None)` raises.
    if isinstance(raw_props, list):
        values = list(raw_props)
    elif isinstance(raw_props, dict):
        values = list(raw_props.get("values") or [])
    else:
        values = []

    # De-dup top-level environment predicates (nested groups not detected).
    values = [v for v in values if v.get("key") != "environment"]
    values.append(
        {
            "key": "environment",
            "value": list(ENV_FILTER_VALUES),
            "operator": "exact",
            "type": "event",
        }
    )
    out["properties"] = {"type": "AND", "values": values}
    return out


def _is_already_patched(filters: dict) -> bool:
    """True only if BOTH env predicate AND date_from match our target state.

    Half-patched insights (env added but date_from missing or wrong) intentionally
    return False so they get re-PATCHed to complete them.
    """
    if filters.get("date_from") != POSTHOG_DATA_CUTOFF:
        return False
    raw_props = filters.get("properties")
    if isinstance(raw_props, list):
        values = raw_props
    elif isinstance(raw_props, dict):
        values = raw_props.get("values") or []
    else:
        return False
    for v in values:
        if (
            v.get("key") == "environment"
            and sorted(v.get("value") or []) == sorted(ENV_FILTER_VALUES)
            and v.get("operator") == "exact"
        ):
            return True
    return False


def _api_host() -> str:
    """App/REST host. POSTHOG_HOST is ingest (us.i.posthog.com); we strip the .i."""
    override = os.getenv("POSTHOG_API_HOST")
    if override:
        return override
    ingest = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
    return ingest.replace(".i.posthog.com", ".posthog.com")


def run(dashboard_ids: list[int], dry_run: bool = False) -> None:
    key = os.getenv("POSTHOG_PERSONAL_API_KEY")
    project_id = os.getenv("POSTHOG_PROJECT_ID")
    if not key or not project_id:
        print(
            "ERROR: POSTHOG_PERSONAL_API_KEY and POSTHOG_PROJECT_ID must be set",
            file=sys.stderr,
        )
        sys.exit(2)

    host = _api_host()
    base = f"{host}/api/projects/{project_id}"
    headers = {"Authorization": f"Bearer {key}"}

    patched = 0
    skipped = 0
    skipped_query_based = 0
    failed = 0

    with httpx.Client(timeout=60.0) as client:
        for dash_id in dashboard_ids:
            print(f"\nDashboard {dash_id}:")
            # PostHog dashboard read returns `tiles` inline. The /tiles/ subpath
            # is POST-only for mounting new tiles, so we read the dashboard body.
            try:
                dash_resp = client.get(f"{base}/dashboards/{dash_id}/", headers=headers)
                dash_resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                print(f"  ERROR fetching dashboard: {type(e).__name__}: {e}")
                failed += 1
                continue

            tiles = dash_resp.json().get("tiles", [])
            for tile in tiles:
                insight = tile.get("insight") or {}
                insight_id = insight.get("id")
                if not insight_id:
                    continue

                try:
                    insight_resp = client.get(f"{base}/insights/{insight_id}/", headers=headers)
                    insight_resp.raise_for_status()
                    insight_full = insight_resp.json()
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    print(f"  insight {insight_id}: FAIL fetch ({type(e).__name__})")
                    failed += 1
                    continue

                name = insight_full.get("name") or insight_full.get("derived_name") or "?"

                # Query-based insights store filters under query.source.properties.
                # PATCHing `filters` on them would be a no-op — skip and warn.
                if insight_full.get("query"):
                    print(
                        f"  insight {insight_id} ({name}): WARNING query-based "
                        f"insight, filter PATCH not applicable — skipping"
                    )
                    skipped_query_based += 1
                    continue

                old_filters = insight_full.get("filters") or {}
                if _is_already_patched(old_filters):
                    print(f"  insight {insight_id} ({name}): already patched, skipping")
                    skipped += 1
                    continue

                new_filters = _mutate_filters(old_filters)
                print(f"  insight {insight_id} ({name}):")
                print(f"    BEFORE: {json.dumps(old_filters.get('properties'), default=str)[:120]}")
                print(f"    AFTER:  {json.dumps(new_filters['properties'], default=str)[:120]}")
                print(f"    date_from: {old_filters.get('date_from')} → {new_filters['date_from']}")

                if dry_run:
                    continue

                try:
                    patch_resp = client.patch(
                        f"{base}/insights/{insight_id}/",
                        json={"filters": new_filters},
                        headers=headers,
                    )
                    patch_resp.raise_for_status()
                    patched += 1
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    print(f"    PATCH FAILED ({type(e).__name__}): {e}")
                    failed += 1

    mode = "DRY-RUN" if dry_run else "APPLIED"
    print(
        f"\n[{mode}] patched={patched} already_patched={skipped} "
        f"query_based_skipped={skipped_query_based} failed={failed}"
    )


def main():
    parser = argparse.ArgumentParser(description="Apply env+date filter to PostHog insights on given dashboards.")
    parser.add_argument(
        "--dashboard-id",
        type=int,
        action="append",
        required=True,
        help="Dashboard ID to walk. Repeat for multiple.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed mutations without PATCHing.",
    )
    args = parser.parse_args()
    print(f"Running against dashboards={args.dashboard_id}, dry_run={args.dry_run}")
    run(dashboard_ids=args.dashboard_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
