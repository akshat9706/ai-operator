"""LLM backends. The agent loop only needs one thing: given the conversation, produce the next
turn (optional text + tool calls). Backends:

* AnthropicBackend - the Claude Messages API with tool use, over plain HTTPS (no SDK dependency).
* (see offline.py)  - deterministic stand-in policies used to test the platform without a model.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from .state import RunState


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    content: list[dict[str, Any]]           # raw content blocks, stored in the conversation history
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, messages: list[dict], tools: list[dict], state: RunState) -> AssistantTurn: ...


class AnthropicBackend:
    name = "anthropic"
    RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5-5", max_tokens: int = 4096,
                 base_url: str = "https://api.anthropic.com", session: requests.Session | None = None,
                 max_retries: int = 5, sleep=time.sleep, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self.model, self.max_tokens, self.base_url = model, max_tokens, base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries, self.sleep, self.timeout = max_retries, sleep, timeout

    def generate(self, system: str, messages: list[dict], tools: list[dict], state: RunState) -> AssistantTurn:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if tools:
            body["tools"] = tools
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(f"{self.base_url}/v1/messages", json=body, headers=headers,
                                         timeout=self.timeout)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
                resp = None
            if resp is not None:
                if resp.status_code == 200:
                    return self._parse(resp.json())
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in self.RETRYABLE:
                    raise LLMError(last)
            if attempt < self.max_retries:
                delay = min(30.0, 1.5 * (2 ** attempt))
                if resp is not None and resp.headers.get("retry-after", "").replace(".", "", 1).isdigit():
                    delay = max(delay, float(resp.headers["retry-after"]))
                state.log("llm_retry", attempt=attempt + 1, error=last, delay=delay)
                self.sleep(delay)
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last}")

    @staticmethod
    def _parse(data: dict) -> AssistantTurn:
        content, text, calls = [], [], []
        for b in data.get("content", []):
            if b.get("type") == "text":
                text.append(b["text"])
                content.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(b["id"], b["name"], b.get("input") or {}))
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}})
        return AssistantTurn("\n".join(text), calls, content, data.get("stop_reason", ""), data.get("usage", {}))
