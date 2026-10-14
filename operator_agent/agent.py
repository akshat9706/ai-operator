"""The agent loop:  Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete.

The model does the understanding, planning and adapting (one tool-use loop). The platform around it
provides what a model cannot guarantee by itself: permissions, approvals, safe retries, working
memory, independent verification, checkpoint/resume and a final evidence report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from requests.cookies import create_cookie

from .browser import Browser
from .executor import Executor
from .llm import AssistantTurn, LLMBackend, LLMError, ToolCall
from .prompt import SYSTEM_PROMPT
from .state import RunState, ToolResult
from .tools import ToolSpec
from .verifier import LLMAuditor, VerificationResult, Verifier

KEEP_FULL_RESULTS = 6          # most recent tool results kept verbatim in the model's view
ELIDE_ABOVE = 500              # older results longer than this are shortened
MAX_CONSECUTIVE_ERRORS = 8
MAX_VERIFICATION_ATTEMPTS = 3
MAX_NUDGES = 3


class Agent:
    def __init__(self, backend: LLMBackend, executor: Executor, verifier: Verifier, state: RunState,
                 tools: dict[str, ToolSpec], system_prompt: str = SYSTEM_PROMPT, browser: Browser | None = None,
                 auditor: LLMAuditor | None = None):
        self.backend, self.executor, self.verifier, self.state = backend, executor, verifier, state
        self.browser = browser
        self.auditor = auditor
        self.tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.schema}
                          for t in tools.values()]
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []
        self.remaining: list[ToolCall] = []                 # calls of the current turn not yet executed
        self.collected: list[tuple[str, ToolResult]] = []   # results of the current turn so far
        self.verification: VerificationResult | None = None
        self.nudges = 0

    # ------------------------------------------------------------------ run
    def run(self) -> RunState:
        st = self.state
        if not self.messages:
            u = st.user
            self.messages.append({"role": "user", "content": [{"type": "text", "text":
                f"Request from {u.get('name')} ({u.get('username')}, {u.get('email')}):\n{st.goal}"}]})
        st.status = "running"
        st.log("run_start", goal=st.goal, backend=getattr(self.backend, "name", "?"))
        while st.status == "running":
            if not self.remaining:
                if st.step >= st.max_steps:
                    self._stop("failed", f"Step budget of {st.max_steps} exhausted before the task was verified complete.")
                    break
                if self.executor.stuck:
                    self._stop("failed", self.executor.stuck)
                    break
                if st.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self._stop("failed", f"Stopped after {st.consecutive_errors} consecutive failed actions; "
                               "a human should look at this.")
                    break
                if not self._next_turn():
                    break
            self._drain()
            if not self.remaining and self.collected:   # also on finish, so the transcript stays valid
                self._close_turn()
            self._checkpoint()
        self._checkpoint()
        st.log("run_end", status=st.status)
        return st

    # ------------------------------------------------------------- one turn
    def _next_turn(self) -> bool:
        st = self.state
        st.step += 1
        try:
            turn = self.backend.generate(self.system_prompt, self._view(), self.tool_defs, st)
        except LLMError as exc:
            self._stop("failed", f"The language model could not be reached: {exc}")
            return False
        self.messages.append({"role": "assistant", "content": turn.content or [{"type": "text", "text": "(no output)"}]})
        st.log("assistant", text=turn.text[:600], calls=[{"tool": c.name} for c in turn.tool_calls],
               usage=turn.usage)
        if not turn.tool_calls:
            self.nudges += 1
            if self.nudges > MAX_NUDGES:
                self._stop("stalled", "The model stopped acting without calling finish.")
                return False
            self.messages.append({"role": "user", "content": [{"type": "text", "text":
                "You did not call a tool. Continue the task with a tool call, or call finish if you are done "
                "or blocked."}]})
            return True
        self.remaining, self.collected = list(turn.tool_calls), []
        return True

    def _drain(self) -> None:
        st = self.state
        while self.remaining and st.status == "running":
            call = self.remaining[0]
            if call.name == "finish":
                result = self._finish(call.input)
                self.collected.append((call.id, result))
                self.remaining.pop(0)
                # Anything the model batched after finish is not executed, but every tool_use block must
                # still be answered or the next API request is rejected as an invalid transcript.
                for extra in self.remaining:
                    self.collected.append((extra.id, ToolResult(
                        False, "Not executed: finish must be the last call of a turn.", {}, "invalid")))
                self.remaining = []
                break
            was_paused = next((p for p in st.pending if p.get("tool") == call.name and p.get("input") == call.input), None)
            result = self.executor.execute(call.name, call.input, expect_action=(was_paused or {}).get("action"))
            if result.error_kind == "deferred":
                st.pending = [{"tool": call.name, "input": call.input, "why": result.data.get("summary", ""),
                               "action": result.data.get("action")}]
                st.status = "waiting_for_input" if call.name == "ask_human" else "waiting_for_approval"
                st.log("paused", status=st.status, pending=st.pending)
                return
            self.collected.append((call.id, result))
            self.remaining.pop(0)
        if st.status == "running":
            st.pending = []

    def _close_turn(self) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": cid, "content": r.for_model(), "is_error": not r.ok}
                  for cid, r in self.collected]
        self.messages.append({"role": "user", "content": blocks})
        self.state.last_results = [r for _, r in self.collected]
        self.collected = []
        self.nudges = 0

    # --------------------------------------------------------------- finish
    def _finish(self, args: dict[str, Any]) -> ToolResult:
        st = self.state
        status = args.get("status")
        summary = str(args.get("summary", ""))
        st.final = {"status": status, "summary": summary, "claims": args.get("claims", [])}
        if status in ("blocked", "failed"):
            st.status = status
            return ToolResult(True, "Recorded. The employee will be told what is missing.")
        if status != "done":
            return ToolResult(False, "status must be one of done | blocked | failed.", {}, "invalid")
        st.verification_attempts += 1
        vr = self.verifier.verify(args.get("claims"))
        if vr.passed and self.auditor is not None:
            sufficient, gaps = self.auditor.audit(st.goal, summary, vr, self.verifier.dir, st)
            if not sufficient:
                vr.passed = False
                vr.problems += [f"Independent auditor: {g}" for g in gaps] or ["Independent auditor judged the evidence insufficient."]
        self.verification = vr
        st.log("verification", attempt=st.verification_attempts, passed=vr.passed, problems=vr.problems,
               checks=[{"description": c.description, "passed": c.passed, "detail": c.detail} for c in vr.checks])
        if vr.passed:
            st.status = "completed"
            return ToolResult(True, "Independent verification passed. Task complete.")
        if st.verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
            st.status = "failed"
            st.final["summary"] = (f"Verification failed {st.verification_attempts} times; the outcome could not be "
                                   "confirmed. Original operator summary: " + summary)
        return ToolResult(False, vr.feedback(), {}, "verification_failed")

    def _stop(self, status: str, reason: str) -> None:
        self.state.status = status
        self.state.final = {**self.state.final, "status": status, "summary": reason + (
            "  Operator's last summary: " + self.state.final["summary"] if self.state.final.get("summary") else "")}
        self.state.log("stopped", status=status, reason=reason)

    # ---------------------------------------------------- context handling
    def _view(self) -> list[dict[str, Any]]:
        """What the model sees: full recent observations, shortened older ones, plus working memory.
        The full transcript stays in self.messages and in events.jsonl."""
        idx = [(i, j) for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], list)
               for j, b in enumerate(m["content"]) if b.get("type") == "tool_result"]
        old = set(idx[:-KEEP_FULL_RESULTS]) if len(idx) > KEEP_FULL_RESULTS else set()
        view: list[dict[str, Any]] = []
        for i, m in enumerate(self.messages):
            if isinstance(m["content"], list):
                blocks = []
                for j, b in enumerate(m["content"]):
                    if (i, j) in old and len(str(b.get("content", ""))) > ELIDE_ABOVE:
                        b = {**b, "content": str(b["content"])[:300] + " ...[older observation shortened; "
                             "rely on working memory or re-read if you need it]"}
                    blocks.append(b)
                view.append({"role": m["role"], "content": blocks})
            else:
                view.append(m)
        last = view[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            last["content"] = last["content"] + [{"type": "text", "text": self.state.working_memory()}]
        return view

    # ---------------------------------------------------------- persistence
    def _checkpoint(self) -> None:
        browser = None
        if self.browser is not None:
            browser = {"url": self.browser.page.url if self.browser.page else None,
                       "cookies": [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                                   for c in self.browser.session.cookies]}
        data = {"state": self.state.to_dict(), "messages": self.messages, "browser": browser,
                "remaining": [{"id": c.id, "name": c.name, "input": c.input} for c in self.remaining],
                "collected": [{"id": cid, "ok": r.ok, "content": r.content, "data": r.data, "error_kind": r.error_kind}
                              for cid, r in self.collected]}
        (self.state.run_dir / "checkpoint.json").write_text(json.dumps(data, default=str), encoding="utf-8")

    @staticmethod
    def read_checkpoint(run_dir: Path) -> dict[str, Any]:
        return json.loads((Path(run_dir) / "checkpoint.json").read_text(encoding="utf-8"))

    def restore(self, data: dict[str, Any]) -> None:
        """Restore the conversation and the in-flight turn. The RunState itself is rebuilt by the caller
        (RunState.from_dict(data['state'])) *before* the executor/tools are created, so they share it."""
        self.messages = data["messages"]
        b = data.get("browser")
        if b and self.browser is not None:
            for c in b["cookies"]:
                self.browser.session.cookies.set_cookie(create_cookie(c["name"], c["value"], domain=c["domain"], path=c["path"]))
            if b["url"]:
                try:
                    self.browser.goto(b["url"])      # rebuild the page the operator was looking at
                except ConnectionError:
                    pass
        self.remaining = [ToolCall(c["id"], c["name"], c["input"]) for c in data["remaining"]]
        self.collected = [(c["id"], ToolResult(c["ok"], c["content"], c["data"], c["error_kind"]))
                          for c in data["collected"]]
