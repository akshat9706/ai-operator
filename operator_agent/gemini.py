"""Google Gemini backend (free tier friendly).

The agent loop stores the conversation in the Claude Messages format (text / tool_use / tool_result blocks).
This backend translates that to Gemini's generateContent format on every call and translates the reply back.

Things that matter for Gemini specifically:

* Gemini 3 models return a `thoughtSignature` on function-call parts and REQUIRE it to be sent back with the
  history. We keep every raw Gemini part inside the stored block (key "_g") and replay it unchanged.
* Tool schemas are sent as `parametersJsonSchema` (full JSON Schema), so free-form object arguments such as
  `values` and `expect` work. Tools with no parameters send no schema at all.
* Free-tier limits are low (requests per minute / per day). `min_interval` spaces calls out, 429 answers are
  retried after the delay Google asks for, and an exhausted DAILY quota stops the run with a clear message
  instead of retrying uselessly.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

import requests

from .llm import AssistantTurn, LLMError, ToolCall
from .state import RunState

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
SKIP_SIGNATURE = "skip_thought_signature_validator"


def _model_path(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


class GeminiBackend:
    name = "gemini"
    RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, max_tokens: int = 8192,
                 base_url: str = DEFAULT_BASE_URL, session: requests.Session | None = None,
                 max_retries: int = 10, sleep=time.sleep, timeout: float = 120.0, min_interval: float = 0.0,
                 clock=time.monotonic):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey).")
        # thinking tokens count against the output limit on Gemini, so never go below a roomy minimum
        self.model, self.max_tokens, self.base_url = model, max(max_tokens, 8192), base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries, self.sleep, self.timeout = max_retries, sleep, timeout
        self.min_interval, self.clock = min_interval, clock
        self._last_call: float | None = None

    # ------------------------------------------------------------------ request building
    def _contents(self, messages: list[dict]) -> list[dict]:
        names: dict[str, tuple[str, str | None]] = {}      # tool_use id -> (function name, gemini call id)
        need_sig = "gemini-3" in self.model
        out: list[dict] = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            parts: list[dict] = []
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                signed_call_done = False
                for b in content:
                    t = b.get("type")
                    if t == "text":
                        if b.get("_g"):
                            parts.append(b["_g"])
                        elif b.get("text", "").strip():
                            parts.append({"text": b["text"]})
                    elif t == "gemini_thought":
                        parts.append(b["_g"])
                    elif t == "tool_use":
                        raw = b.get("_g")
                        gid = (raw or {}).get("functionCall", {}).get("id")
                        names[b["id"]] = (b["name"], gid)
                        if raw:
                            parts.append(raw)
                        else:   # history that did not come from Gemini (e.g. a resumed run): synthesize a part
                            part: dict[str, Any] = {"functionCall": {"name": b["name"], "args": b.get("input") or {}}}
                            if need_sig and not signed_call_done:
                                part["thoughtSignature"] = SKIP_SIGNATURE
                                signed_call_done = True
                            parts.append(part)
                    elif t == "tool_result":
                        name, gid = names.get(b["tool_use_id"], ("unknown", None))
                        text = b.get("content", "")
                        if not isinstance(text, str):
                            text = json.dumps(text, default=str)
                        fr: dict[str, Any] = {"name": name,
                                              "response": {"error": text} if b.get("is_error") else {"output": text}}
                        if gid:
                            fr["id"] = gid
                        parts.append({"functionResponse": fr})
            if not parts:
                parts.append({"text": "(no output)"})
            out.append({"role": role, "parts": parts})
        return out

    @staticmethod
    def _declarations(tools: list[dict]) -> list[dict]:
        decls = []
        for t in tools:
            d: dict[str, Any] = {"name": t["name"], "description": t.get("description", "")}
            schema = t.get("input_schema") or {}
            if schema.get("properties"):
                d["parametersJsonSchema"] = schema
            decls.append(d)
        return decls

    # ------------------------------------------------------------------ call
    def generate(self, system: str, messages: list[dict], tools: list[dict], state: RunState) -> AssistantTurn:
        body: dict[str, Any] = {"systemInstruction": {"parts": [{"text": system}]},
                                "contents": self._contents(messages),
                                "generationConfig": {"maxOutputTokens": self.max_tokens}}
        if tools:
            body["tools"] = [{"functionDeclarations": self._declarations(tools)}]
        url = f"{self.base_url}/v1beta/{_model_path(self.model)}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        last = ""
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
                    return self._parse(resp.json())
                last = f"HTTP {resp.status_code}: {resp.text[:400]}"
                if resp.status_code == 429 and re.search(r"PerDay|per day", resp.text, re.I):
                    raise LLMError("Gemini DAILY free quota is used up for this model (it resets at midnight "
                                   "Pacific time). Try again tomorrow or pass a different --model. " + last)
                if resp.status_code == 404:
                    raise LLMError(f"Gemini model '{self.model}' was not found for your key. Run "
                                   "`python -m operator_agent models` to see the names you can use. " + last)
                if resp.status_code not in self.RETRYABLE:
                    raise LLMError(last)
            if attempt < self.max_retries:
                delay = min(60.0, 2.0 * (2 ** attempt))
                if resp is not None:
                    delay = max(delay, min(self._retry_after(resp), 90.0))
                state.log("llm_retry", attempt=attempt + 1, error=last, delay=delay)
                self.sleep(delay)
        raise LLMError(f"Gemini call failed after {self.max_retries + 1} attempts: {last}")

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
        m = re.search(r'"retryDelay":\s*"([\d.]+)s"', resp.text)
        return float(m.group(1)) + 1 if m else 0.0

    # ------------------------------------------------------------------ response parsing
    @staticmethod
    def _parse(data: dict) -> AssistantTurn:
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback") or {}
            raise LLMError(f"Gemini returned no answer (blocked or empty): {json.dumps(fb)[:300]}")
        cand = cands[0]
        text, calls, content = [], [], []
        for p in (cand.get("content") or {}).get("parts") or []:
            if "functionCall" in p:
                fc = p["functionCall"]
                cid = fc.get("id") or f"call_{uuid.uuid4().hex[:10]}"
                calls.append(ToolCall(cid, fc["name"], dict(fc.get("args") or {})))
                content.append({"type": "tool_use", "id": cid, "name": fc["name"],
                                "input": dict(fc.get("args") or {}), "_g": p})
            elif p.get("thought"):
                if p.get("thoughtSignature"):
                    content.append({"type": "gemini_thought", "_g": p})
            elif "text" in p:
                if p["text"].strip():
                    text.append(p["text"])
                    block: dict[str, Any] = {"type": "text", "text": p["text"]}
                    if p.get("thoughtSignature"):
                        block["_g"] = p
                    content.append(block)
        um = data.get("usageMetadata") or {}
        return AssistantTurn("\n".join(text), calls, content, str(cand.get("finishReason", "")).lower(),
                             {"input_tokens": um.get("promptTokenCount", 0),
                              "output_tokens": um.get("candidatesTokenCount", 0)})


SKIP_WORDS = ("image", "tts", "transcribe", "computer-use", "robotics", "omni", "banana", "embedding", "gemma",
              "customtools", "argon", "live", "audio")


def usable_text_models(names: list[str]) -> list[str]:
    return [n for n in names if not any(w in n for w in SKIP_WORDS) and "pro" not in n.split("-")]


def probe_model(name: str, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL, session=None) -> str:
    """Send one tiny request and report what happened, in a few words."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    s = session or requests.Session()
    try:
        r = s.post(f"{base_url.rstrip('/')}/v1beta/{_model_path(name)}:generateContent",
                   headers={"x-goog-api-key": key or "", "content-type": "application/json"},
                   json={"contents": [{"role": "user", "parts": [{"text": "Reply with the word ok."}]}],
                         "generationConfig": {"maxOutputTokens": 64}}, timeout=60)
    except requests.RequestException as exc:
        return f"no connection ({type(exc).__name__})"
    if r.status_code == 200:
        return "WORKS"
    words = {404: "not available for your key", 429: "quota/rate limit hit (try later)",
             503: "overloaded right now (may work later)", 403: "key not allowed", 400: "rejected the request"}
    return words.get(r.status_code, f"HTTP {r.status_code}")


def list_models(api_key: str | None = None, base_url: str = DEFAULT_BASE_URL, session=None) -> list[str]:
    """Names of the Gemini models this key can call generateContent on (for `operator_agent models`)."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey).")
    s = session or requests.Session()
    names: list[str] = []
    token = None
    for _ in range(10):
        params = {"pageSize": 100, **({"pageToken": token} if token else {})}
        try:
            r = s.get(f"{base_url.rstrip('/')}/v1beta/models", params=params, headers={"x-goog-api-key": key}, timeout=30)
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Gemini: {exc}") from exc
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        names += [m["name"].removeprefix("models/") for m in j.get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m["name"]]
        token = j.get("nextPageToken")
        if not token:
            break
    return sorted(names)
