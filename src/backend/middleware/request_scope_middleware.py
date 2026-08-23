"""Opens and closes the per-request memo used by artist_access.

Kept as its own middleware rather than folded into AnalyticsMiddleware so the
memo's lifetime is not tied to whether a path is analytics-excluded — every
request gets a scope, including /health and the internal sweep endpoints.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

import artist_access


class RequestScopeMiddleware(BaseHTTPMiddleware):
    """Give each request its own artist_access memo, and always tear it down.

    The teardown is in a `finally` on purpose: leaking a memo into a recycled
    worker context would let one user's visibility answer serve another's
    request, which in this module means leaking authorization.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        token = artist_access.begin_request_scope()
        try:
            return await call_next(request)
        finally:
            artist_access.end_request_scope(token)
