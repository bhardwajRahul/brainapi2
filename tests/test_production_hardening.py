from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.api.errors import install_error_handlers
from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.routes.system import system_router
from src.workers.redis_url import redis_connection_url


def test_malformed_authorization_fails_closed_with_401():
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(BrainPATMiddleware)

    @app.get("/system/test")
    async def protected():
        return {"ok": True}

    with patch.dict(os.environ, {"BRAINPAT_TOKEN": "system-token"}):
        response = TestClient(app).get(
            "/system/test", headers={"Authorization": "malformed"}
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_INVALID"


@pytest.mark.parametrize(
    ("path", "method", "operation"),
    [
        ("/system/brains/demo/reset", "get", "reset"),
        ("/system/brains/demo/delete", "get", "delete"),
        ("/system/brains/demo/create-backup", "post", "create-backup"),
    ],
)
def test_unfinished_system_routes_return_structured_501(path, method, operation):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(system_router)

    response = getattr(TestClient(app), method)(path)

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"
    assert response.json()["operation"] == operation


def test_cors_defaults_and_production_wildcard_guard():
    from src.services.api.app import _cors_allowed_origins

    with patch.dict(os.environ, {"ENV": "production", "CORS_ALLOWED_ORIGINS": ""}):
        assert _cors_allowed_origins() == []
    with patch.dict(os.environ, {"ENV": "development", "CORS_ALLOWED_ORIGINS": ""}):
        assert _cors_allowed_origins() == ["*"]
    with patch.dict(
        os.environ, {"ENV": "production", "CORS_ALLOWED_ORIGINS": "*"}
    ):
        with pytest.raises(RuntimeError, match="development"):
            _cors_allowed_origins()


def test_plugin_failures_are_fatal_by_default_in_production():
    from src.services.api.app import _enforce_plugin_results

    with patch.dict(os.environ, {"ENV": "production"}, clear=False):
        with pytest.raises(RuntimeError, match="demo"):
            _enforce_plugin_results({"demo": False})
    with patch.dict(
        os.environ,
        {"ENV": "production", "PLUGIN_FAILURE_POLICY": "warn"},
        clear=False,
    ):
        _enforce_plugin_results({"demo": False})


def test_worker_redis_url_propagates_encoded_password():
    assert (
        redis_connection_url("redis", 6379, "secret:/@ value")
        == "redis://:secret%3A%2F%40%20value@redis:6379/0"
    )
    assert redis_connection_url("redis", "6379", None) == "redis://redis:6379/0"
