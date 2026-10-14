"""The operator's tool set. Deliberately small and generic: nothing here knows about expenses.

Each tool has a JSON schema (what the model sees), a `describe` step that reduces a call to its
real effect (what the permission gate sees), and a `handler` (what actually runs).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .browser import Browser, BrowserError, Page
from .context import KnowledgeBase, Sandbox, Systems, ToolError, resource_key
from .human import Human
from .state import RunState, ToolResult


@dataclass
class Action:
    """The real-world effect of a tool call, as seen by the platform."""
    string: str                 # permission-matching string, e.g. 'http:POST http://h/reports/3/submit'
    summary: str                # human-readable
    mutating: bool = False
    resource: str | None = None
    fingerprint: str = ""
    context: str = ""           # extra info shown to a human approver


@dataclass
class ToolContext:
    sandbox: Sandbox
    kb: KnowledgeBase
    systems: Systems
    browser: Browser
    state: RunState
    human: Human


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict
    describe: Callable[[ToolContext, dict], Action]
    handler: Callable[[ToolContext, dict, Action], ToolResult]


def _fp(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _obj(props: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": props}
    if required:                      # an empty `required` array is invalid in strict JSON-schema validators
        schema["required"] = required
    return schema


S = {"type": "string"}

# ---------------------------------------------------------------- simple tools

def _kb_search_desc(ctx, a): return Action("tool:kb_search", f"search knowledge base for '{a['query']}'")
def _kb_search(ctx, a, act):
    hits = ctx.kb.search(a["query"])
    if not hits:
        return ToolResult(True, "No matching documents. Try different keywords or fs_list knowledge/.", {"hits": []})
    txt = "\n".join(f"- {h['path']} > {h['heading']}: {h['snippet']}" for h in hits)
    return ToolResult(True, "Top matches (read the whole file with fs_read for details):\n" + txt, {"hits": hits})


def _fs_list_desc(ctx, a):
    p = ctx.sandbox.resolve(a.get("path", "."))
    return Action(f"fs_list:{ctx.sandbox.rel(p)}", f"list folder {ctx.sandbox.rel(p)}")
def _fs_list(ctx, a, act):
    p = ctx.sandbox.resolve(a.get("path", "."))
    if not p.is_dir():
        raise ToolError(f"'{ctx.sandbox.rel(p)}' is not a folder.")
    entries = [f"{c.name}/" if c.is_dir() else f"{c.name} ({c.stat().st_size} bytes)" for c in sorted(p.iterdir())]
    return ToolResult(True, f"{ctx.sandbox.rel(p) or '.'}:\n" + "\n".join(entries or ["(empty)"]),
                      {"entries": [c.name for c in sorted(p.iterdir())], "dirs": [c.name for c in sorted(p.iterdir()) if c.is_dir()]})


def _fs_read_desc(ctx, a):
    p = ctx.sandbox.resolve(a["path"])
    return Action(f"fs_read:{ctx.sandbox.rel(p)}", f"read file {ctx.sandbox.rel(p)}",
                  resource=f"fs:{ctx.sandbox.rel(p)}")
def _fs_read(ctx, a, act):
    p = ctx.sandbox.resolve(a["path"])
    if not p.is_file():
        raise ToolError(f"No such file: {ctx.sandbox.rel(p)}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError("File is not plain text.")
    if len(text) > 20000:
        text = text[:20000] + "\n...[truncated]"
    return ToolResult(True, text, {"path": ctx.sandbox.rel(p)})


def _fs_write_desc(ctx, a):
    p = ctx.sandbox.resolve(a["path"])
    rel = ctx.sandbox.rel(p)
    return Action(f"fs_write:{rel}", f"write file {rel} ({len(a['content'])} chars)", True,
                  f"fs:{rel}", _fp("fs_write", rel, a["content"]))
def _fs_write(ctx, a, act):
    p = ctx.sandbox.resolve(a["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(a["content"], encoding="utf-8")
    return ToolResult(True, f"Wrote {ctx.sandbox.rel(p)} ({len(a['content'])} chars).", {"path": ctx.sandbox.rel(p)})


def _list_systems_desc(ctx, a): return Action("tool:list_systems", "list company systems")
def _list_systems(ctx, a, act):
    items = ctx.systems.describe()
    return ToolResult(True, json.dumps(items, indent=2), {"systems": items})


def _plan_desc(ctx, a): return Action("tool:update_plan", "update plan")
def _plan(ctx, a, act):
    steps = []
    for s in a["steps"]:
        if isinstance(s, dict):         # {"text": ..., "status": ...}
            steps.append({"text": str(s.get("text", "")), "status": str(s.get("status") or "pending")})
        else:                           # some models send a plain list of strings
            steps.append({"text": str(s), "status": "pending"})
    ctx.state.plan = steps
    return ToolResult(True, "Plan updated.")


def _remember_desc(ctx, a): return Action("tool:remember", f"remember {a['key']}")
def _remember(ctx, a, act):
    ctx.state.facts[str(a["key"])] = str(a["value"])[:500]
    return ToolResult(True, f"Remembered '{a['key']}'.")


def _ask_desc(ctx, a): return Action("tool:ask_human", "ask the employee a question")
def _ask(ctx, a, act):
    ans = ctx.human.answer(a["question"], a.get("options"))
    if ans is None:
        return ToolResult(False, "No answer available right now; the run will pause until the employee answers.",
                          {"question": a["question"]}, "deferred")
    return ToolResult(True, f"The employee answered: {ans}", {"answer": ans})


def _finish_desc(ctx, a): return Action("tool:finish", "finish")
def _finish(ctx, a, act): return ToolResult(True, "finish")  # intercepted by the agent loop

# ---------------------------------------------------------------- browser tools

def _page_result(ctx: ToolContext, page: Page, mutating: bool, expected_system_login: bool = False) -> ToolResult:
    notes = []
    session_expired = page.is_login and not expected_system_login
    if session_expired:
        notes.append("NOTE: You were sent to a sign-in page, so your session has probably expired. "
                     "Call browser_login(system) and then repeat what you were doing.")
    elif page.status >= 500:
        notes.append("NOTE: Server error. " + ("This state-changing request MAY have taken effect: re-open the "
                     "resource and check its state before trying again." if mutating else "Retrying a read is safe."))
    elif page.status >= 400:
        notes.append(f"NOTE: The server rejected the request (HTTP {page.status}). Read the ALERT, fix the cause, "
                     "then try again; nothing was changed." if mutating else f"NOTE: HTTP {page.status}.")
    data = {"status": page.status, "url": page.url, "is_login": page.is_login, "alerts": page.alerts,
            "session_expired": session_expired,
            "forms": [{"id": f.id, "method": f.method, "action": f.action,
                       "fields": [x.name for x in f.visible_fields]} for f in page.forms.values()],
            "links": {k: {"href": h, "text": t} for k, (h, t) in page.links.items()}}
    # A fetched page is a successful tool call; HTTP status / session expiry are data that the executor
    # interprets (mutation outcome, retry safety, error counting).
    return ToolResult(True, page.render() + ("\n" + "\n".join(notes) if notes else ""), data)


def _http_action(ctx: ToolContext, method: str, url: str, summary: str, body: dict | None = None,
                 context: str = "") -> Action:
    mutating = method != "GET"
    return Action(f"http:{method} {url}", summary, mutating, resource_key(ctx.systems, url),
                  _fp(method, url, body or {}) if mutating else "", context)


def _goto_desc(ctx, a):
    return _http_action(ctx, "GET", a["url"], f"open {a['url']}")
def _goto(ctx, a, act):
    return _page_result(ctx, ctx.browser.goto(a["url"]), False)


def _click_desc(ctx, a):
    page = ctx.browser.page
    if page is None or a["link_id"] not in page.links:
        raise ToolError(f"No link {a['link_id']} on the current page.", "invalid")
    url, text = page.links[a["link_id"]]
    return _http_action(ctx, "GET", url, f"click '{text}' ({url})")
def _click(ctx, a, act):
    return _page_result(ctx, ctx.browser.click(a["link_id"]), False)


def _submit_desc(ctx, a):
    page = ctx.browser.page
    if page is None or a["form_id"] not in page.forms:
        raise ToolError(f"No form {a['form_id']} on the current page. "
                        f"Forms here: {list(page.forms) if page else 'none (open a page first)'}.", "invalid")
    f = page.forms[a["form_id"]]
    values = {k: str(v) for k, v in (a.get("values") or {}).items()}
    shown = {x.name: values.get(x.name, x.value) for x in f.visible_fields}
    ctx_txt = f"Page: {page.url}\n" + page.text[:3000] + f"\nForm values: {json.dumps(shown)}"
    return _http_action(ctx, f.method, f.action, f"{f.method} {f.action} with {json.dumps(shown)}",
                        shown, ctx_txt)
def _submit(ctx, a, act):
    try:
        page = ctx.browser.submit(a["form_id"], {k: str(v) for k, v in (a.get("values") or {}).items()})
    except BrowserError as exc:
        raise ToolError(str(exc), "invalid")
    return _page_result(ctx, page, act.mutating)


def _login_desc(ctx, a):
    s = ctx.systems.systems.get(a["system"])
    if not s:
        raise ToolError(f"Unknown system '{a['system']}'. Known: {list(ctx.systems.systems)}", "invalid")
    url = s["base_url"].rstrip("/") + s["login"]["path"]
    return Action(f"http:POST {url}", f"sign in to {a['system']} (credentials injected by the platform)",
                  False, None, "")
def _login(ctx, a, act):
    s = ctx.systems.systems[a["system"]]
    cred = ctx.systems.credential(s["login"]["credential"])
    url = s["base_url"].rstrip("/") + s["login"]["path"]
    page = ctx.browser.goto(url)
    form = next((f for f in page.forms.values() if any(x.type == "password" for x in f.fields)), None)
    if form is None:
        if not page.is_login:   # already signed in: the portal redirected us to the app
            return ToolResult(True, f"Already signed in to {a['system']}.", {"status": page.status})
        raise ToolError("Could not find a login form.")
    page = ctx.browser.submit(form.id, {s["login"]["username_field"]: cred["username"],
                                        s["login"]["password_field"]: cred["password"]})
    if page.is_login:
        return ToolResult(False, "Sign-in failed: " + "; ".join(page.alerts or ["unknown reason"]), {}, "tool_error")
    return ToolResult(True, f"Signed in to {a['system']} as {cred['username']}. Now at {page.url}.",
                      {"status": page.status, "url": page.url})


def build_tools() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("kb_search", "Search company policies and procedures (knowledge base). Use this first to "
                 "learn the rules and procedure for a task.", _obj({"query": S}, ["query"]), _kb_search_desc, _kb_search),
        ToolSpec("fs_list", "List a folder on the company file share (folders: inbox/, knowledge/, directory/, out/).",
                 _obj({"path": S}), _fs_list_desc, _fs_list),
        ToolSpec("fs_read", "Read a text file from the company file share.", _obj({"path": S}, ["path"]),
                 _fs_read_desc, _fs_read),
        ToolSpec("fs_write", "Write a text file (only allowed under out/).",
                 _obj({"path": S, "content": S}, ["path", "content"]), _fs_write_desc, _fs_write),
        ToolSpec("list_systems", "List the company web systems you can use, with their URLs.", _obj({}),
                 _list_systems_desc, _list_systems),
        ToolSpec("browser_login", "Sign in to a company system. Credentials are injected by the platform; you never "
                 "see or type them. Call again if your session expires.", _obj({"system": S}, ["system"]),
                 _login_desc, _login),
        ToolSpec("browser_goto", "Open a URL in the browser. Returns the page text, [L#] links and [F#] forms.",
                 _obj({"url": S}, ["url"]), _goto_desc, _goto),
        ToolSpec("browser_click", "Follow a link [L#] on the current page.", _obj({"link_id": S}, ["link_id"]),
                 _click_desc, _click),
        ToolSpec("browser_submit_form", "Fill in and submit a form [F#] on the CURRENT page. Provide values for "
                 "the editable fields; hidden fields (CSRF) are added for you.",
                 _obj({"form_id": S, "values": {"type": "object"}}, ["form_id"]), _submit_desc, _submit),
        ToolSpec("ask_human", "Ask the requesting employee a question when you are missing information or a "
                 "decision that you cannot get from the documents or systems.",
                 _obj({"question": S, "options": {"type": "array", "items": S}}, ["question"]), _ask_desc, _ask),
        ToolSpec("update_plan", "Record or revise your plan. Each step has text and status "
                 "(pending|in_progress|done|blocked). Keep it current.",
                 _obj({"steps": {"type": "array", "items": _obj({"text": S, "status": S}, ["text"])}}, ["steps"]),
                 _plan_desc, _plan),
        ToolSpec("remember", "Store a short fact you will need later (amounts, ids, decisions).",
                 _obj({"key": S, "value": S}, ["key", "value"]), _remember_desc, _remember),
        ToolSpec("finish", "End the task. status is done|blocked|failed. For 'done' you MUST include claims: "
                 "checks that an independent verifier will run against the real system (fresh session) to confirm "
                 "the outcome. Types: web_contains{system,path,contains[],not_contains[]}; "
                 "json_check{system,path,expect{dotted.path:value}} (reads the system's audit API); "
                 "file_contains{path,contains[]}. Claims must cover every resource you changed.",
                 _obj({"status": S, "summary": S,
                       "claims": {"type": "array", "items": _obj({
                           "description": S, "type": S, "system": S, "path": S,
                           "contains": {"type": "array", "items": S}, "not_contains": {"type": "array", "items": S},
                           "expect": {"type": "object"}}, ["type"])}}, ["status", "summary"]),
                 _finish_desc, _finish),
    ]
    return {s.name: s for s in specs}
