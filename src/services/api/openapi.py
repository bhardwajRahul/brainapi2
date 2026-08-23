"""OpenAPI guarantees for discovery and LLM function-calling clients."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
_CORE_TAGS = {"public", "ingest", "retrieve", "model", "meta", "tasks", "system"}
_LIST_PATHS = {
    "/meta/entity-labels",
    "/meta/entity-properties",
    "/meta/relationships-properties",
    "/retrieve/hops",
    "/system/brains-list",
}


def _error_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
            }
        },
    }


def install_openapi_contract(app: FastAPI) -> None:
    """Install a deterministic, fully-described OpenAPI schema builder."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        component_schemas = components.setdefault("schemas", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.update(
            {
                "BrainPAT": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "BrainPAT",
                    "description": "BrainAPI personal access token.",
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "BrainAPI PAT sent as a Bearer token.",
                },
            }
        )

        protected_errors = {
            "400": _error_response("Invalid request"),
            "401": _error_response("Authentication failed"),
            "403": _error_response("Operation forbidden"),
            "404": _error_response("Resource not found"),
            "405": _error_response("Method not allowed"),
            "406": _error_response("Brain not found"),
            "422": _error_response("Validation failed"),
            "429": _error_response("Rate limit exceeded"),
            "500": _error_response("Internal service error"),
            "503": _error_response("Dependency unavailable"),
        }

        for path, path_item in schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                    continue
                summary = operation.get("summary") or operation.get("operationId")
                if not operation.get("description"):
                    operation["description"] = (
                        f"{summary}. See the BrainAPI developer documentation for "
                        "request, response, authentication, and retry semantics."
                    )

                for status_code, response in operation.get("responses", {}).items():
                    if not str(status_code).startswith("2") or not isinstance(response, dict):
                        continue
                    for media in response.get("content", {}).values():
                        if isinstance(media, dict) and media.get("schema") == {}:
                            if _CORE_TAGS.intersection(operation.get("tags", [])):
                                raise ValueError(
                                    f"Core operation {method.upper()} {path} must define "
                                    "a typed success response model"
                                )
                            model = (
                                "GenericListResponse"
                                if path in _LIST_PATHS
                                else "GenericObjectResponse"
                            )
                            component_schemas.setdefault(
                                model,
                                (
                                    {
                                        "type": "array",
                                        "description": "A list of operation-specific JSON values.",
                                        "items": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                            ]
                                        },
                                    }
                                    if model == "GenericListResponse"
                                    else {
                                        "type": "object",
                                        "description": (
                                            "An operation-specific JSON object from an "
                                            "installed extension."
                                        ),
                                        "additionalProperties": True,
                                    }
                                ),
                            )
                            media["schema"] = {
                                "$ref": f"#/components/schemas/{model}"
                            }

                is_public = path in {"/health", "/demo/search"}
                if is_public:
                    operation["security"] = []
                    public_errors = {
                        "405": _error_response("Method not allowed"),
                        "500": _error_response("Internal service error"),
                        "503": _error_response("Dependency unavailable"),
                    }
                    if path == "/demo/search":
                        public_errors.update(
                            {
                                "404": _error_response("Public demo disabled"),
                                "422": _error_response("Validation failed"),
                            }
                        )
                    responses = operation.setdefault("responses", {})
                    for code, response in public_errors.items():
                        if code == "422" or code not in responses:
                            responses[code] = response
                else:
                    operation["security"] = [{"BrainPAT": []}, {"BearerAuth": []}]
                    responses = operation.setdefault("responses", {})
                    for code, response in protected_errors.items():
                        if code == "422" or code not in responses:
                            responses[code] = response

                if path == "/demo/search":
                    operation.setdefault("responses", {})["429"] = _error_response(
                        "Public demo rate limit exceeded"
                    )
                    operation["x-rate-limit"] = {
                        "requests": 30,
                        "window": "60 seconds",
                    }

        schema["servers"] = [
            {"url": "http://localhost:8000", "description": "Local BrainAPI"}
        ]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
