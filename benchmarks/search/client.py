from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from search.config import TERMINAL_TASK_STATUSES, Settings


class SearchDisabledError(RuntimeError):
    pass


@dataclass
class TimedResult:
    data: Any
    latency_ms: float
    status_code: int = 200


@dataclass
class BrainAPIClient:
    settings: Settings
    timeout: float = 120.0
    max_retries: int = 5
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.settings.require_brainapi()
        self._client = httpx.Client(
            base_url=self.settings.brainapi_url,
            headers={
                "Authorization": f"Bearer {self.settings.brainpat_token}",
                "Content-Type": "application/json",
                "X-Brain-ID": self.settings.brain_id,
            },
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BrainAPIClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> TimedResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = self._client.request(
                    method, path, json=json, params=params
                )
                latency_ms = (time.perf_counter() - started) * 1000
                if (
                    response.status_code == 404
                    and path.rstrip("/").endswith("/retrieve/search")
                ):
                    detail = ""
                    try:
                        detail = response.text[:500]
                    except Exception:
                        detail = ""
                    raise SearchDisabledError(
                        "POST /retrieve/search returned 404. Set SEARCH_ENABLED=true "
                        "(and DATA_DB=postgresql) on the API under test. "
                        f"body={detail!r}"
                    ) from None
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = min(2**attempt * 0.5, 30)
                    time.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                if response.status_code >= 400:
                    detail = ""
                    try:
                        detail = response.text[:500]
                    except Exception:
                        detail = ""
                    raise RuntimeError(
                        f"{method} {path} returned {response.status_code}. "
                        f"body={detail!r}"
                    ) from None
                if response.status_code == 204 or not response.content:
                    return TimedResult(
                        data=None,
                        latency_ms=latency_ms,
                        status_code=response.status_code,
                    )
                return TimedResult(
                    data=response.json(),
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            except SearchDisabledError:
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                wait = min(2**attempt * 0.5, 30)
                time.sleep(wait)
        raise RuntimeError(
            f"Request failed after retries: {method} {path}: {last_error}"
        )

    def ingest_structured(
        self,
        triples: list[dict[str, Any]],
        brain_id: str | None = None,
        *,
        mode: str = "deterministic",
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        body = {
            "mode": mode,
            "brain_id": bid,
            "data": triples,
        }
        return self._request("POST", "/ingest/structured", json=body)

    def ingest_text(
        self,
        text: str,
        brain_id: str | None = None,
        *,
        skip_enrichment: bool = False,
        meta_keys: dict[str, Any] | None = None,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        body: dict[str, Any] = {
            "data": {"data_type": "text", "text_data": text},
            "brain_id": bid,
            "observate_for": [],
            "preferred_extraction_entities": [],
            "skip_enrichment": skip_enrichment,
        }
        if meta_keys:
            body["meta_keys"] = meta_keys
        return self._request("POST", "/ingest/", json=body)

    def get_task(self, task_id: str, brain_id: str | None = None) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        return self._request(
            "GET", f"/tasks/{task_id}", params={"brain_id": bid}
        )

    def wait_for_task(
        self,
        task_id: str,
        brain_id: str | None = None,
        *,
        timeout_s: float = 600.0,
        poll_interval_s: float = 2.0,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        started = time.perf_counter()
        while True:
            result = self.get_task(task_id, bid)
            status = (result.data or {}).get("status", "unknown")
            if status in TERMINAL_TASK_STATUSES:
                return TimedResult(
                    data=result.data,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    status_code=result.status_code,
                )
            if time.perf_counter() - started > timeout_s:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {timeout_s}s "
                    f"(last status={status})"
                )
            time.sleep(poll_interval_s)

    def search(
        self,
        query: str,
        brain_id: str | None = None,
        *,
        k: int = 20,
        fusion: str | None = None,
        fusion_alpha: float | None = None,
        rerank: str | None = None,
        mode: str | None = None,
        profile_stages: bool = True,
        channels: list[str] | None = None,
        node_labels: list[str] | None = None,
        community_labels: list[str] | None = None,
        expand: str | None = None,
        extras: dict[str, str] | None = None,
        target: str | None = None,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        body: dict[str, Any] = {
            "query": query,
            "brain_id": bid,
            "k": k,
            "channels": channels or ["passages"],
            "profile_stages": profile_stages,
        }
        if fusion:
            body["fusion"] = fusion
        if fusion_alpha is not None:
            body["fusion_alpha"] = fusion_alpha
        if rerank:
            body["rerank"] = rerank
        if mode and mode != "default":
            body["mode"] = mode
        if node_labels:
            body["node_labels"] = node_labels
        if community_labels:
            body["community_labels"] = community_labels
        if expand and expand != "none":
            body["expand"] = expand
        if extras:
            body["extras"] = extras
        if target:
            body["target"] = target
        return self._request("POST", "/retrieve/search", json=body)

    def get_neighbors(
        self,
        uuid: str,
        brain_id: str | None = None,
        *,
        limit: int = 10,
        look_for: str | None = None,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        params: dict[str, Any] = {
            "uuid": uuid,
            "brain_id": bid,
            "limit": limit,
        }
        if look_for:
            params["look_for"] = look_for
        return self._request("GET", "/retrieve/entities/neighbors", params=params)

    def list_text_chunks(
        self,
        brain_id: str | None = None,
        *,
        limit: int = 100,
        skip: int = 0,
        query_text: str | None = None,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        params: dict[str, Any] = {
            "brain_id": bid,
            "limit": limit,
            "skip": skip,
        }
        if query_text:
            params["query_text"] = query_text
        return self._request("GET", "/retrieve/text-chunks", params=params)
