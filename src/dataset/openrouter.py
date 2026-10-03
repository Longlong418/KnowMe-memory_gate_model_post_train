"""Small OpenRouter HTTP client shared by dataset-construction scripts."""

from __future__ import annotations

import json
import os
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and value and name not in os.environ:
            os.environ[name] = value


_load_dotenv()
ENDPOINT = os.environ.get("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("OPENROUTER_MODEL", "xiaomi/mimo-v2.6-flash")
CACHE_MODEL = "mimo-v2.6-flash"
TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "90"))
RETRIES = max(1, int(os.environ.get("OPENROUTER_RETRIES", "1")))


def get_api_key() -> str:
    _load_dotenv()
    value = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter_api_key") or "").strip()
    if not value:
        raise RuntimeError("openrouter_api_key was not found in .env")
    return value


def chat(
    key: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    session: str,
    retries: int | None = None,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "messages": messages,
        },
        ensure_ascii=True,
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "knowme-memory-gate-data-prep/0.1",
            "HTTP-Referer": "https://github.com/openai/knowme-memory-gate-post-train",
            "X-Title": "KnowMe memory gate data preparation",
            "X-Session-ID": session,
        },
        method="POST",
    )
    last_error: Exception | None = None
    request_retries = max(1, retries if retries is not None else RETRIES)
    for attempt in range(request_retries):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            if attempt + 1 >= request_retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last_error or RuntimeError("OpenRouter request failed")
