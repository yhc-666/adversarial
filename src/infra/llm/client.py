from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
from typing import Any
from urllib.parse import urlsplit

import httpx


class LLMError(Exception):
    pass


class LLMClient:
    """Send chat-completion requests with bounded concurrency and retries."""

    def __init__(self, *, base_url: str, api_key: str, concurrency: int) -> None:
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.netloc
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("API_BASE_URL must be an HTTP(S) API base URL without credentials or query parameters")
        if not api_key.strip() or concurrency <= 0:
            raise ValueError("API_KEY must be nonempty and concurrency must be positive")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._slots = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
            trust_env=False,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._http.aclose()

    async def complete(self, payload: dict[str, Any]) -> str:
        error = "Request failed"
        for attempt in range(3):
            delay = float(2 ** attempt)
            try:
                async with self._slots:
                    response = await self._http.post(self.url, json=payload)
            except httpx.TransportError as exc:
                error = f"Transport failure: {type(exc).__name__}"
            else:
                if response.is_success:
                    try:
                        content = response.json()["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError):
                        content = None
                    if isinstance(content, str):
                        return content
                    error = "Response is missing choices[0].message.content"
                else:
                    error = f"API request failed with HTTP {response.status_code}"
                    if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                        raise LLMError(error)
                    delay = max(delay, retry_after(response.headers.get("retry-after")))
            if attempt < 2:
                await asyncio.sleep(delay)
        raise LLMError(f"{error}; exhausted 3 request attempts")


def retry_after(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0
