"""Tests for root and health endpoints, and auth enforcement.

Acceptance criteria:
1. GET / returns 200 with {"message": "Msanii AI Backend is running"}
2. GET /health returns 200 with status "healthy"
3. Unauthenticated request to a protected endpoint returns 401
"""

from fastapi.testclient import TestClient


def test_root_returns_200_with_message(client):
    """GET / returns 200 with expected message."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Msanii AI Backend is running"


def test_health_returns_200_with_healthy_status(client):
    """GET /health returns 200 with status 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


def test_protected_endpoint_requires_auth():
    """Unauthenticated request to a protected endpoint returns 401.

    Uses a fresh TestClient without the auth override so the real
    get_current_user_id dependency runs and rejects missing credentials.
    """
    import main

    # Clear any lingering overrides so the real auth dependency is used
    main.app.dependency_overrides.clear()

    with TestClient(main.app, raise_server_exceptions=False) as unauthenticated_client:
        response = unauthenticated_client.get("/artists")

    assert response.status_code == 401
