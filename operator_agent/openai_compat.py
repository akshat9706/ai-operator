"""Backend for any server that speaks the OpenAI "chat completions" protocol with tool calling.

That covers most open-model hosts: Groq, Cerebras, OpenRouter, NVIDIA NIM, Together, and local servers such as
Ollama or LM Studio. The agent loop stores the conversation in the Claude Messages format; this backend
translates it to chat-completions messages on every call and translates the reply back.

Open models are more error-prone with tools than frontier models, so this backend is forgiving:
  * arguments that are not valid JSON become an empty call (the platform then tells the model what was missing);
  * a server-side "tool_use_failed" (the model wrote a malformed call) is retried, and if it keeps happening the
    turn is returned empty so the agent loop nudges the model instead of the whole run failing;
  * a spent DAILY quota stops the run with a clear message instead of retrying uselessly.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from .llm import AssistantTurn, LLMError, ToolCall
from .state import RunState


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: tuple[str, ...]            # environment variables that may hold the key (first one set wins)
    default_model: str | None = None
    needs_key: bool = True
    signup: str = ""


PRESETS: dict[str, Preset] = {
    "cerebras": Preset("https://api.cerebras.ai/v1", ("CEREBRAS_API_KEY",), None, True, "https://cloud.cerebras.ai"),
    "groq": Preset("https://api.groq.com/openai/v1", ("GROQ_API_KEY",), "openai/gpt-oss-120b", True,
                   "https://console.groq.com/keys"),
    "openrouter": Preset("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), None, True,
                         "https://openrouter.ai/keys"),
    "nvidia": Preset("https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY",), None, True,
                     "https://build.nvidia.com"),
    "ollama": Preset("http://localhost:11434/v1", ("OLLAMA_API_KEY",), None, False, "https://ollama.com/download"),
    "openai-compatible": Preset("", ("OPENAI_API_KEY",), None, True, ""),
}

DAILY_LIMIT = re.compile(r"per day|\(TPD\)|\(RPD\)|free-models-per-day|daily", re.I)


def resolve_key(preset: Preset, provider: str) -> str | None:
    for env in preset.key_env:
        if os.environ.get(env):
            return os.environ[env]
    return None if preset.needs_key else "not-needed"


class OpenAICompatBackend:
    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, provider: str = "openai-compatible", api_key: str | None = None, model: str | None = None,
                 max_tokens: int = 4096, base_url: str | None = None, session: requests.Session | None = None,
                 max_retries: int = 8, sleep=time.sleep, timeout: float = 180.0, min_interval: float = 0.0,
                 clock=time.monotonic):
        preset = PRESETS[provider]
        self.provider, self.name = provider, provider
        self.api_key = api_key or resolve_key(preset, provider)
        if not self.api_key:
            raise LLMError(f"{preset.key_env[0]} is not set" + (f" (get a free key at {preset.signup})." if preset.signup else "."))
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or preset.base_url or "").rstrip("/")
        if not self.base_url:
            raise LLMError("This provider needs --base-url (for example http://localhost:1234/v1).")
        self.model = model or preset.default_model
        if not self.model:
            raise LLMError(f"No model chosen for {provider}. Pass --model <name>; "
                           f"`python -m operator_agent models --provider {provider}` lists the names.")
        self.max_tokens, self.session = max_tokens, session or requests.Session()
        self.max_retries, self.sleep, self.timeout = max_retries, sleep, timeout
        self.min_interval, self.clock = min_interval, clock
        self._last_call: float | None = None

    # ------------------------------------------------------------------ request building
    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                text = "\n".join(b["text"] for b in content if b.get("type") == "text" and b.get("text", "").strip())
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                         for b in content if b.get("type") == "tool_use"]
                msg: dict[str, Any] = {"role": "assistant"}
                if text:
                    msg["content"] = text
                elif not calls:
                    msg["content"] = "(no output)"
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:
                for b in content:       # tool results first: they must directly follow the assistant's calls
                    if b.get("type") == "tool_result":
                        c = b.get("content", "")
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": c if isinstance(c, str) else json.dumps(c, default=str)})
                text = "\n".join(b["text"] for b in content if b.get("type") == "text" and b.get("text", "").strip())
                if text:
                    out.append({"role": "user", "content": text})
        return out

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                  "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
                for t in tools]

    # ------------------------------------------------------------------ call
    def generate(self, system: str, messages: list[dict], tools: list[dict], state: RunState) -> AssistantTurn:
        body: dict[str, Any] = {"model": self.model, "messages": self._messages(system, messages),
                                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = self._tools(tools)
            body["tool_choice"] = "auto"
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        last, malformed = "", 0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            resp: requests.Response | None
            try:
                resp = self.session.post(url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last, resp = f"{type(exc).__name__}: {exc}", None
            self._last_call = self.clock()
            if resp is not None:
                if resp.status_code == 200:
                    return self._parse(resp.json(), state)
                last = f"HTTP {resp.status_code}: {resp.text[:400]}"
                if resp.status_code == 429 and DAILY_LIMIT.search(resp.text):
                    raise LLMError(f"The DAILY free quota of {self.provider} is used up. Try again tomorrow, or use "
                                   "another --model / --provider. " + last)
                if resp.status_code == 400 and "tool_use_failed" in resp.text:
                    malformed += 1
                    state.log("llm_retry", attempt=attempt + 1, error="model wrote a malformed tool call", delay=0)
                    if malformed >= 3:       # give up on this turn; the agent loop will nudge the model
                        return AssistantTurn("(my previous tool call was malformed)", [],
                                             [{"type": "text", "text": "(my previous tool call was malformed)"}], "error")
                    continue
                if resp.status_code in (401, 403):
                    raise LLMError(f"{self.provider} rejected the API key. Check it in your .env file. " + last)
                if resp.status_code == 404:
                    raise LLMError(f"{self.provider} could not find model '{self.model}' (or the address is wrong). "
                                   f"Run `python -m operator_agent models --provider {self.provider}`. " + last)
                if resp.status_code not in self.RETRYABLE:
                    raise LLMError(last)
            if attempt < self.max_retries:
                delay = min(60.0, 2.0 * (2 ** attempt))
                if resp is not None:
                    delay = max(delay, min(self._retry_after(resp), 90.0))
                state.log("llm_retry", attempt=attempt + 1, error=last, delay=delay)
                self.sleep(delay)
        raise LLMError(f"{self.provider} call failed after {self.max_retries + 1} attempts: {last}")

    def _throttle(self) -> None:
        if self.min_interval and self._last_call is not None:
            wait = self.min_interval - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)

    @staticmethod
    def _retry_after(resp: requests.Response) -> float:
        hdr = resp.headers.get("retry-after", "")
        if hdr.replace(".", "", 1).isdigit():
            return float(hdr)
        m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)\b", resp.text, re.I)      # Groq style message
        if m:
            v = float(m.group(1))
            return v / 1000 if m.group(2).lower() == "ms" else v * (60 if m.group(2).lower() == "m" else 1) + 1
        return 0.0

    # ------------------------------------------------------------------ response parsing
    @staticmethod
    def _parse(data: dict, state: RunState) -> AssistantTurn:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"The server returned no answer: {json.dumps(data)[:300]}")
        ch = choices[0]
        msg = ch.get("message") or {}
        text = (msg.get("content") or "").strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        calls: list[ToolCall] = []
        used: set[str] = set()
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            cid = tc.get("id") or ""
            if not cid or cid in used:
                cid = f"call_{uuid.uuid4().hex[:10]}"
            used.add(cid)
            raw = fn.get("arguments")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
            except (ValueError, TypeError):
                state.log("bad_tool_arguments", tool=fn.get("name"), raw=str(raw)[:200])
                args = {}
            calls.append(ToolCall(cid, fn.get("name", ""), args))
            content.append({"type": "tool_use", "id": cid, "name": fn.get("name", ""), "input": args})
        u = data.get("usage") or {}
        return AssistantTurn(text, calls, content, str(ch.get("finish_reason") or ""),
                             {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)})


def list_openai_models(provider: str, base_url: str | None = None, session=None) -> list[str]:
    preset = PRESETS[provider]
    key = resolve_key(preset, provider)
    if not key:
        raise LLMError(f"{preset.key_env[0]} is not set" + (f" (get a free key at {preset.signup})." if preset.signup else "."))
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or preset.base_url).rstrip("/")
    if not base:
        raise LLMError("This provider needs --base-url.")
    try:
        r = (session or requests.Session()).get(f"{base}/models", headers={"authorization": f"Bearer {key}"}, timeout=30)
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {base}: {exc}") from exc
    if r.status_code != 200:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    items = j.get("data", j.get("models", []))
    return sorted(str(m.get("id") or m.get("name")) for m in items if isinstance(m, dict))
