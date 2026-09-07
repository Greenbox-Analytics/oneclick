"""Shared pagination utilities for list endpoints."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class PaginatedResponse(BaseModel):
    """Standard paginated response envelope."""

    data: list[Any]
    total: int
    page: int
    page_size: int


def paginate_query(
    query,
    page: int | None,
    page_size: int = 50,
) -> PaginatedResponse | list:
    """
    Apply pagination to a Supabase query builder.

    Backward compatible:
    - If page is None, executes the query and returns raw data array (old format).
    - If page is an int, applies .range() and returns PaginatedResponse.

    The query must already have .select("*", count="exact") called on it.
    """
    if page is None:
        result = query.execute()
        return result.data or []

    offset = (page - 1) * page_size
    result = query.range(offset, offset + page_size - 1).execute()

    return PaginatedResponse(
        data=result.data or [],
        total=result.count or 0,
        page=page,
        page_size=page_size,
    )


def fetch_all(make_query: Callable[[], object], page_size: int = 500, order_by: str = "id") -> list[dict]:
    """Every row of a PostgREST query, paged so Supabase's 1,000-row cap can't
    truncate silently. For AGGREGATIONS that must see every row (ledger
    rollups); list endpoints want paginate_query instead.

    page_size stays UNDER the server cap, or a page shortened by the cap would
    end the loop and reinstate the truncation.

    ORDERED: .range() is bare LIMIT/OFFSET and Postgres guarantees no row order
    between statements, so unordered pages can repeat or skip rows and mis-total
    an aggregation. `id` is the UUID primary key everywhere here. .order() uses
    params.set(), so a factory adding its own order still composes.

    Takes a zero-arg FACTORY, never a builder to reuse: postgrest 2.30's
    .range() APPENDS its params, so a second call on one builder sends
    `offset=0&offset=1000`, PostgREST honors the first, and page two re-requests
    page one forever. The factory must not add .range()/.limit() itself.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = make_query().order(order_by).range(offset, offset + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size
