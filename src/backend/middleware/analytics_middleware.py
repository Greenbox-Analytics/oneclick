"""FastAPI middleware that captures per-request analytics events."""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from analytics import capture as analytics_capture  # noqa: E402

# Paths we don't want to track (noise / health probes / docs)
_EXCLUDED_PREFIXES = ("/static", "/docs", "/redoc", "/openapi.json", "/health")


def _should_skip(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _EXCLUDED_PREFIXES)


class AnalyticsMiddleware(BaseHTTPMiddleware):
    """Capture `request_completed` for every response + `request_failed` on exception."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if _should_skip(request.url.path):
            return await call_next(request)

        # Distinct ID: prefer authenticated user from request.state if available.
        # Anonymous requests share distinct_id "anonymous" (acceptable for beta).
        distinct_id = getattr(request.state, "user_id", None) or "anonymous"

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            analytics_capture(
                distinct_id,
                "request_failed",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "status": 500,
                    "error_type": type(exc).__name__,
                    "duration_ms": duration_ms,
                },
            )
            # Re-raise so FastAPI's exception handlers run
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        analytics_capture(
            distinct_id,
            "request_completed",
            {
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
