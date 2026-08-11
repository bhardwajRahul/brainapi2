from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from recsys.config import TERMINAL_TASK_STATUSES, Settings


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
                    if (
                        response.status_code == 404
                        and "/recsys/recommend" in path
                        and "user_unknown" not in detail
                    ):
                        raise RuntimeError(
                            f"GET {path} returned 404. Ensure the recsys-gnn "
                            f"plugin is loaded. body={detail!r}"
                        ) from None
                    response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return TimedResult(data=None, latency_ms=latency_ms)
                return TimedResult(data=response.json(), latency_ms=latency_ms)
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
        poll_interval_s: float = 1.0,
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
                )
            if time.perf_counter() - started > timeout_s:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {timeout_s}s "
                    f"(last status={status})"
                )
            time.sleep(poll_interval_s)

    def train_lightgcn(
        self,
        brain_id: str | None = None,
        *,
        epochs: int = 20,
        embedding_dim: int = 64,
        n_layers: int = 3,
        wait: bool = True,
        timeout_s: float = 600.0,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        body = {
            "brain_id": bid,
            "model": "lightgcn",
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "n_layers": n_layers,
            "wait": wait,
            "timeout_s": timeout_s,
        }
        # Train can take a while; bump client timeout for this call.
        old = self._client.timeout
        self._client.timeout = timeout_s + 30.0
        try:
            return self._request("POST", "/recsys/train", json=body)
        finally:
            self._client.timeout = old

    def recommend(
        self,
        user_id: str,
        brain_id: str | None = None,
        *,
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        params: dict[str, Any] = {
            "user_id": user_id,
            "top_k": top_k,
            "brain_id": bid,
            "exclude_seen": str(exclude_seen).lower(),
        }
        return self._request("GET", "/recsys/recommend", params=params)

    def recommend_graph(
        self,
        user_id: str,
        brain_id: str | None = None,
        *,
        top_k: int = 20,
        exclude_seen: bool = True,
        labels: list[str] | None = None,
        include_attribute_pref: bool = True,
        dampen_degree: bool = True,
        recency_half_life_days: float | None = 90.0,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        target = str(user_id)
        params: dict[str, Any] = {
            "target": target,
            "top_k": top_k,
            "brain_id": bid,
            "exclude_seen": str(exclude_seen).lower(),
            "include_asymmetric": "true",
            "include_multi_interest": "true",
            "include_attribute_pref": str(include_attribute_pref).lower(),
            "dampen_degree": str(dampen_degree).lower(),
            "diversify": "true",
            "labels": labels or ["PRODUCT"],
        }
        if recency_half_life_days is not None:
            params["recency_half_life_days"] = recency_half_life_days
        return self._request("GET", "/retrieve/recommend", params=params)

    def features_rec_interaction(
        self,
        *,
        user_id: str,
        item_id: str,
        behavior: str,
        timestamp: str | None = None,
        attributes: dict[str, Any] | None = None,
        brain_id: str | None = None,
        wait: bool = True,
        timeout_s: float = 120.0,
        seq: int = 1,
    ) -> TimedResult:
        bid = brain_id or self.settings.brain_id
        body: dict[str, Any] = {
            "user_id": user_id,
            "item_id": item_id,
            "behavior": behavior,
            "brain_id": bid,
            "wait": wait,
            "timeout_s": timeout_s,
            "seq": seq,
        }
        if timestamp:
            body["timestamp"] = timestamp
        if attributes:
            body["attributes"] = attributes
        return self._request("POST", "/features-rec/interactions", json=body)
