"""Shared pagination utilities for list endpoints."""

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
