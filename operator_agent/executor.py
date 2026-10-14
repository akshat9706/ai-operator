"""Executor: the generic reliability and safety layer around every tool call.

Pipeline per call:
  validate -> describe real effect -> unsafe-retry guard -> permission gate (+ human approval)
  -> run (retrying only operations that are safe to repeat) -> classify outcome -> ledger/log.

Nothing here knows what task is being performed.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .context import ToolError, resource_key
from .human import ApprovalRequest
from .permissions import PermissionPolicy
from .state import ApprovalRecord, Mutation, ToolResult
from .tools import Action, ToolContext, ToolSpec

MAX_MODEL_CONTENT = 6000
_TYPES = {"string": str, "array": list, "object": dict}


def validate_args(spec: ToolSpec, args: Any) -> str | None:
    if not isinstance(args, dict):
        return "Arguments must be a JSON object."
    for key in spec.schema.get("required", []):
        if key not in args:
            return f"Missing required argument '{key}'."
    for key, val in args.items():
        want = spec.schema["properties"].get(key, {}).get("type")
        if want in _TYPES and not isinstance(val, _TYPES[want]):
            return f"Argument '{key}' must be of type {want}."
    return None


REPEAT_HINT_AT = 3     # same failing action this many times in a row -> tell the model to stop and rethink
REPEAT_STOP_AT = 6     # ...and this many -> stop the run, a human should look


class Executor:
    def __init__(self, tools: dict[str, ToolSpec], ctx: ToolContext, policy: PermissionPolicy,
                 read_retries: int = 2, backoff: float = 0.3, sleep: Callable[[float], None] = time.sleep):
        self.tools, self.ctx, self.policy = tools, ctx, policy
        self.read_retries, self.backoff, self.sleep = read_retries, backoff, sleep
        self.fail_streak: dict[str, int] = {}
        self.stuck: str | None = None

    # ------------------------------------------------------------------ main
    def execute(self, name: str, args: Any, expect_action: str | None = None) -> ToolResult:
        """`expect_action` is set when re-running a call that was paused for a human: the effect the human was
        shown. If the call now resolves to something else (the page it referred to is gone), do not run it."""
        st = self.ctx.state
        seq = st.next_seq()
        spec = self.tools.get(name)
        if spec is None:
            return self._done(name, seq, None, ToolResult(
                False, f"Unknown tool '{name}'. Available tools: {sorted(self.tools)}", {}, "invalid"))
        bad = validate_args(spec, args)
        if bad:
            return self._done(name, seq, None, ToolResult(False, bad, {}, "invalid"))
        try:
            action = spec.describe(self.ctx, args)
        except ToolError as exc:
            return self._done(name, seq, None, ToolResult(False, str(exc), {}, exc.kind))
        except Exception as exc:        # odd argument shapes must come back to the model as errors, not crash the run
            return self._done(name, seq, None, ToolResult(
                False, f"Could not understand this call ({type(exc).__name__}: {exc}). Check the argument "
                "names and types in the tool description.", {}, "invalid"))
        st.log("tool_call", seq=seq, tool=name, args=self._redact(args), action=action.string)

        if expect_action is not None and action.string != expect_action:
            return self._done(name, seq, action, ToolResult(
                False, "This call was paused earlier, but the page it referred to has changed since (for example the "
                f"session expired while paused). It would now do: {action.string}, not what was approved/shown: "
                f"{expect_action}. Nothing was done. Re-open the page and repeat what you intended.", {}, "stale_action"))

        # 1. never blindly repeat a state-changing action whose outcome is unknown
        if action.mutating:
            unresolved = [m for m in st.unresolved_ambiguous() if m.fingerprint == action.fingerprint]
            if unresolved:
                return self._done(name, seq, action, ToolResult(
                    False, "Blocked: the identical previous attempt failed with an unknown outcome (server error or "
                    f"timeout) and you have not re-checked {unresolved[0].resource} since. Re-open it and look at its "
                    "current state first; only repeat the action if it clearly did not take effect.",
                    {}, "unsafe_retry"))

        # 2. permission gate
        decision = self.policy.evaluate(action.string)
        key = action.fingerprint or action.string
        if decision.verdict == "deny":
            st.log("permission", seq=seq, action=action.string, verdict="deny", rule=decision.rule)
            return self._done(name, seq, action, ToolResult(
                False, f"Blocked by company policy: {decision.reason} Do not look for another way around this. "
                "If it prevents the task, tell the employee via finish(status='blocked') or ask_human.",
                {}, "denied"))
        if decision.verdict == "ask" and key not in st.approved:
            resp = self.ctx.human.approve(ApprovalRequest(action.string, decision.reason, action.summary, action.context))
            st.approvals.append(ApprovalRecord(seq, action.string, decision.reason, resp.decision, resp.by, resp.note))
            st.log("approval", seq=seq, action=action.string, decision=resp.decision, by=resp.by, note=resp.note)
            if resp.decision == "defer":
                return self._done(name, seq, action, ToolResult(
                    False, "Waiting for the employee to approve this action. The run will pause.",
                    {"action": action.string, "summary": action.summary}, "deferred"))
            if resp.decision != "approve":
                return self._done(name, seq, action, ToolResult(
                    False, "The employee declined this action" + (f": {resp.note}" if resp.note else "") +
                    ". Do not attempt it again; adjust the plan or report back.", {}, "not_approved"))
            st.approved.add(key)

        # 3. run (reads are retried automatically; state-changing calls never are)
        result = self._run(spec, action, args)

        # 4. classify and record
        self._bookkeep(name, seq, action, result, key)
        return self._done(name, seq, action, result)

    # -------------------------------------------------------------- internals
    def _run(self, spec: ToolSpec, action: Action, args: dict) -> ToolResult:
        attempt = 0
        while True:
            try:
                result = spec.handler(self.ctx, args, action)
            except ToolError as exc:
                result = ToolResult(False, str(exc), {}, exc.kind)
            except ConnectionError as exc:
                result = ToolResult(False, f"Connection problem: {exc}. " + (
                    "The request may or may not have reached the server; re-check the resource's state before "
                    "repeating it." if action.mutating else "Retrying a read is safe."), {}, "transient")
            except Exception as exc:        # a bug or odd argument shape must not crash the whole run
                self.ctx.state.log("tool_exception", tool=spec.name, error=f"{type(exc).__name__}: {exc}")
                result = ToolResult(False, f"The tool failed unexpectedly ({type(exc).__name__}: {exc}). Check your "
                                    "arguments against the tool description and try again."
                                    + (" This state-changing call may or may not have taken effect: re-check the "
                                       "resource before repeating it." if action.mutating else ""),
                                    {}, "transient" if action.mutating else "tool_error")
            transient = result.error_kind == "transient" or result.data.get("status", 0) >= 500
            if transient and not action.mutating and attempt < self.read_retries:
                attempt += 1
                self.ctx.state.log("retry", tool=spec.name, action=action.string, attempt=attempt)
                self.sleep(self.backoff * (2 ** (attempt - 1)))
                continue
            if attempt and not transient:
                result.content = f"(succeeded after {attempt} automatic retr{'y' if attempt == 1 else 'ies'})\n" + result.content
                result.data["retries"] = attempt
            return result

    def _bookkeep(self, name: str, seq: int, action: Action, result: ToolResult, key: str) -> None:
        st = self.ctx.state
        status = result.data.get("status")
        expired = result.data.get("session_expired", False)
        if action.mutating:
            if result.error_kind == "transient" or (status and status >= 500):
                outcome = "ambiguous"
            elif not result.ok or expired or (status and status >= 400):
                outcome = "rejected"
            else:
                outcome = "ok"
            resource = action.resource or name
            final = result.data.get("url")
            if outcome == "ok" and final and action.resource:
                better = resource_key(self.ctx.systems, final)
                if better != action.resource and better.rsplit("/", 1)[-1].isdigit():
                    resource = better   # e.g. POST /reports -> redirected to /reports/7
            st.mutations.append(Mutation(seq, action.fingerprint, resource, name, action.summary, outcome))
            # An approval is spent once the system has actually evaluated the action. A request bounced to the
            # sign-in page (expired session) never reached the business logic, so the human's "yes" still stands.
            if outcome == "ok" or (outcome == "rejected" and not expired):
                st.approved.discard(key)
            result.data["mutation_outcome"] = outcome
        elif result.ok and not expired and (status is None or status < 400):
            res = resource_key(self.ctx.systems, result.data["url"]) if result.data.get("url") else action.resource
            if res:
                st.record_read(res)

    def _done(self, name: str, seq: int, action: Action | None, result: ToolResult) -> ToolResult:
        st = self.ctx.state
        status = result.data.get("status")
        problem = (not result.ok) or result.data.get("session_expired", False) or bool(status and status >= 400)
        if name not in ("update_plan", "remember"):
            st.consecutive_errors = st.consecutive_errors + 1 if problem else 0
        if action is not None and action.mutating:
            if problem:
                n = self.fail_streak[action.string] = self.fail_streak.get(action.string, 0) + 1
                if n >= REPEAT_HINT_AT:
                    result.content += (f"\n\nSTOP: this same action has now failed {n} times in a row. Do not send it "
                                       "again unchanged. Read the error messages on the page, change the values to "
                                       "match them exactly, or use ask_human, or finish with status 'blocked'.")
                if n >= REPEAT_STOP_AT:
                    self.stuck = (f"Stopped: the same action ({action.summary}) failed {n} times in a row and the "
                                  "model was not able to fix it. A human should look at this.")
            else:
                self.fail_streak.pop(action.string, None)
        if len(result.content) > MAX_MODEL_CONTENT:
            result.content = result.content[:MAX_MODEL_CONTENT] + "\n...[truncated]"
        st.log("tool_result", seq=seq, tool=name, ok=result.ok, error_kind=result.error_kind, status=status,
               outcome=result.data.get("mutation_outcome"), problem=problem,
               session_expired=bool(result.data.get("session_expired")), preview=result.content[:240])
        return result

    @staticmethod
    def _redact(args: Any) -> Any:
        if isinstance(args, dict) and "content" in args and isinstance(args["content"], str):
            return {**args, "content": args["content"][:200] + ("..." if len(args["content"]) > 200 else "")}
        return args
