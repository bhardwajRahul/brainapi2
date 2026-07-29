from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from locomo.config import TERMINAL_TASK_STATUSES, Settings


@dataclass
class TimedResult:
    data: Any
    latency_ms: float


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
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = min(2**attempt * 0.5, 30)
                    time.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return TimedResult(data=None, latency_ms=latency_ms)
                return TimedResult(data=response.json(), latency_ms=latency_ms)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                wait = min(2**attempt * 0.5, 30)
                time.sleep(wait)
        raise RuntimeError(f"Request failed after retries: {method} {path}: {last_error}")

    def ingest_text(
        self,
        text: str,
        brain_id: str,
        *,
        observate_for: list[str] | None = None,
        preferred_extraction_entities: list[str] | None = None,
        source_timestamp: str | None = None,
    ) -> TimedResult:
        body = {
            "data": {"data_type": "text", "text_data": text},
            "brain_id": brain_id,
            "observate_for": observate_for
            or [
                "facts about each speaker",
                "events and when they happened",
                "relationships between people",
            ],
            "preferred_extraction_entities": preferred_extraction_entities
            or ["Person", "Event", "Location", "Date"],
        }
        if source_timestamp:
            body["source_timestamp"] = source_timestamp
        return self._request("POST", "/ingest/", json=body)

    def get_task(self, task_id: str, brain_id: str) -> TimedResult:
        return self._request(
            "GET", f"/tasks/{task_id}", params={"brain_id": brain_id}
        )

    def wait_for_task(
        self,
        task_id: str,
        brain_id: str,
        *,
        timeout_s: float = 600.0,
        poll_interval_s: float = 2.0,
    ) -> TimedResult:
        started = time.perf_counter()
        while True:
            result = self.get_task(task_id, brain_id)
            status = (result.data or {}).get("status", "unknown")
            if status in TERMINAL_TASK_STATUSES:
                return TimedResult(
                    data=result.data,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            if time.perf_counter() - started > timeout_s:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {timeout_s}s "
                    f"(last status={status})"
                )
            time.sleep(poll_interval_s)

    def retrieve_context(
        self,
        text: str,
        brain_id: str,
        *,
        historical_limit: int = 10,
        max_passages: int = 8,
        max_facts: int = 40,
        apply_fact_filter: bool = True,
        use_ppr: bool = False,
        sufficiency_retry: bool = False,
        cross_event_bridges: int = 3,
    ) -> TimedResult:
        body: dict[str, Any] = {
            "text": text,
            "brain_id": brain_id,
            "historical_limit": historical_limit,
            "max_passages": max_passages,
            "max_facts": max_facts,
            "apply_fact_filter": apply_fact_filter,
            "use_ppr": use_ppr,
            "sufficiency_retry": sufficiency_retry,
            "cross_event_bridges": cross_event_bridges,
        }
        return self._request("POST", "/retrieve/context", json=body)

    def entity_status(self, target: str, brain_id: str) -> TimedResult:
        return self._request(
            "GET",
            "/retrieve/entity/status",
            params={"target": target, "brain_id": brain_id},
        )

    def list_entities(self, brain_id: str, *, limit: int = 5) -> TimedResult:
        return self._request(
            "GET",
            "/retrieve/entities",
            params={"brain_id": brain_id, "limit": limit},
        )
