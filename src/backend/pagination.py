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
    """Every row of a PostgREST query, paged with .range() so the server's
    default max-rows cap (1,000 on Supabase) can never truncate silently.
    page_size stays UNDER that cap: a page that comes back short because the
    server capped it would end the loop and reinstate the truncation.

    ORDERED, because .range() is bare LIMIT/OFFSET and Postgres guarantees no
    row order between two statements — unordered pages can repeat or skip rows
    and silently mis-total an aggregation. `id` is the UUID primary key on
    every table here, so it is a total order. postgrest's .order() uses
    params.set() with comma-appending (unlike .range()'s .add()), so a factory
    that adds its own order still composes: this becomes the tiebreaker.

    For AGGREGATIONS that must see every row (ledger rollups). Not for list
    endpoints — those want paginate_query and a page size the UI can render.

    Takes a zero-arg FACTORY and builds a fresh query per page, never a
    builder to reuse: on postgrest 2.30 `.range()` does
    `self.request.params = self.request.params.add("offset", ...)` — httpx
    QueryParams.add APPENDS, so a second .range() on the same builder sends
    `offset=0&offset=1000&limit=1000&limit=1000`, PostgREST honors the first,
    page two re-requests page one, and the loop never sees a short page.
    The factory must not add .range()/.limit() itself.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = make_query().order(order_by).range(offset, offset + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size
