"""Redis connection URL helpers for worker transports."""

from __future__ import annotations

from urllib.parse import quote


def redis_connection_url(
    host: str,
    port: str | int,
    password: str | None,
    *,
    database: int = 0,
) -> str:
    """Build a Redis URL, percent-encoding credentials when authentication is enabled."""

    auth = f":{quote(password, safe='')}@" if password else ""
    return f"redis://{auth}{host}:{port}/{database}"
