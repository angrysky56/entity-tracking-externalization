"""Model backends.

Two backends, both exposing ``chat(prompt) -> str``:

* OllamaBackend  - local models via /api/chat (never starts/stops the server).
* MiniMaxBackend - cloud MiniMax via the Anthropic-compatible Messages endpoint
                   (https://api.minimax.io/anthropic/v1/messages). Only ``text``
                   blocks are returned; ``thinking`` blocks are dropped.

Both take a generation cap so a chatty/reasoning model can't run away — which
also prevents the long, truncated outputs that produced ``no_answer`` results.
"""

from __future__ import annotations

import os

import httpx


class OllamaBackend:
    """Minimal synchronous client for the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        timeout: float = 180.0,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout
        # think: None -> omit (model default); True/False -> explicit toggle.
        # For deepseek-r1/qwen3, False inserts a no_think token (reasoning off).
        self.think = think

    def chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        content = (msg.get("content") or "").strip()
        if content:
            return content
        # Reasoning-on with an empty content field: the answer may be only in
        # the separated `thinking` field. Fall back so grading still works.
        return (msg.get("thinking") or "").strip()

    def available_models(self) -> list[str]:
        """Model names Ollama currently has pulled (for a friendly error)."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError:
            return []


class MiniMaxBackend:
    """MiniMax via the Anthropic-compatible Messages endpoint.

    POSTs to ``{host}/messages`` with header ``X-Api-Key``. The response
    ``content`` is a list of blocks; only ``text`` blocks are kept (``thinking``
    blocks are dropped). The key comes from ``api_key`` or ``$MINIMAX_API_KEY``.
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        host: str = "https://api.minimax.io/anthropic/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.host}/messages",
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        texts = [
            b.get("text", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        out = "\n".join(t for t in texts if t).strip()
        if out:
            return out
        # Fallback: some replies emit only a `thinking` block (no text block).
        # The ANSWER line is often still in there, so grade off it rather than
        # recording a spurious no_answer.
        thinks = [
            b.get("thinking", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        return "\n".join(t for t in thinks if t).strip()

    def available_models(self) -> list[str]:
        # No cheap list endpoint; treat key presence as readiness.
        return ["<minimax>"] if self.api_key else []


def make_backend(
    backend: str,
    model: str,
    host: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """Construct a backend by name ('ollama' | 'minimax')."""
    if backend == "minimax":
        return MiniMaxBackend(
            model=model,
            host=host or "https://api.minimax.io/anthropic/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return OllamaBackend(
        model=model,
        host=host or "http://localhost:11434",
        temperature=temperature,
        num_predict=max_tokens,
    )
