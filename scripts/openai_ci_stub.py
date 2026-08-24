#!/usr/bin/env python3
"""Minimal deterministic OpenAI-compatible server for isolated CI smoke tests."""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _embedding(text: str, dimensions: int = 3072) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/").endswith("/embeddings"):
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            dimensions = int(request.get("dimensions") or 3072)
            data = [
                {"object": "embedding", "index": index, "embedding": _embedding(str(text), dimensions)}
                for index, text in enumerate(inputs)
            ]
            self._json(
                200,
                {
                    "object": "list",
                    "model": request.get("model", "ci-embedding"),
                    "data": data,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
            return
        self._json(404, {"error": {"message": "unsupported CI stub endpoint"}})

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
