"""Platform guarantees that hold regardless of how clever (or dishonest) the model is.

* the permission gate and file sandbox cannot be talked or tricked around
* an operator that claims success without doing the work is caught by the verifier
* the Claude API client speaks the wire protocol correctly (tested against a local fake API that enforces the
  same transcript rules as the real one) and survives overload / hard failures
* the optional independent auditor can veto a weak proof
"""
import json
import os
import threading
import unittest
from pathlib import Path

from werkzeug.serving import make_server
from flask import Flask, jsonify, request

from tests.helpers import (TODAY, assert_valid_transcript, fresh_company, new_runs_dir, run_expense)
from company_env.launch import start_portal
from operator_agent.agent import Agent
from operator_agent.context import KnowledgeBase, Sandbox
from operator_agent.human import NoHuman, ScriptedHuman
from operator_agent.llm import AnthropicBackend, AssistantTurn, LLMError, ToolCall
from operator_agent.offline import GeneratorBackend, call
from operator_agent.permissions import PermissionPolicy
from operator_agent.runtime import build_runtime
from operator_agent.state import RunState
from tasks.expense_oracle import grade


class PortalCase(unittest.TestCase):
    def setUp(self):
        self.portal = start_portal()
        self.addCleanup(self.portal.stop)

    def runtime(self, policy=None, human=None, goal="test", company=None, backend=None, **kw):
        backend = backend or GeneratorBackend(policy or (lambda: iter(())))
        return build_runtime(company or fresh_company(), new_runs_dir(), goal, "priya.sharma", backend,
                             human or NoHuman(), self.portal.url, read_backoff=0, today=TODAY, **kw)


# --------------------------------------------------------------------------------- permission gate & sandbox
class TestRepeatedFailureGuard(PortalCase):
    def test_same_failing_form_resubmitted_over_and_over_is_stopped_and_hinted(self):
        def policy():
            yield call("browser_login", system="expense_portal")
            r = yield call("browser_goto", url=f"{self.portal.url}/reports/new")
            yield call("browser_submit_form", form_id="F1", values={"title": "Loop test", "cost_center": "CC-1042"})
            for _ in range(12):                       # a read in between each try, as real models do
                r = yield call("browser_goto", url=f"{self.portal.url}/reports/1/lines/new")
                r = yield call("browser_submit_form", form_id="F1", values={"date": "bad", "category": "Nope",
                                                                           "merchant": "x", "amount": "1.5",
                                                                           "receipt_ref": ""})
                if "STOP" in str(r):
                    self.hint_seen = True
        self.hint_seen = False
        rt = self.runtime(policy, goal="loop")
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertIn("same action", rt.state.final["summary"])
        self.assertTrue(self.hint_seen, "the model must be told to stop repeating before the run is killed")
        bad = [m for m in rt.state.mutations if m.outcome == "rejected"]
        self.assertEqual(len(bad), 6)


class TestGateAndSandbox(PortalCase):
    def ex(self, rt, tool, **args):
        return rt.agent.executor.execute(tool, args)

    def test_operator_cannot_reach_audit_ops_or_foreign_hosts(self):
        rt = self.runtime()
        for url in (f"{self.portal.url}/audit/reports.json", f"{self.portal.url}/__faults",
                    "http://evil.example/steal", f"{self.portal.url.replace('127.0.0.1', 'localhost')}/reports"):
            r = self.ex(rt, "browser_goto", url=url)
            self.assertEqual(r.error_kind, "denied", url)
        self.assertEqual(rt.state.mutations, [])

    def test_file_sandbox_blocks_traversal_symlinks_and_writes_outside_out(self):
        company = fresh_company()
        (company / "fs" / "inbox" / "escape").symlink_to("/etc")
        rt = self.runtime(company=company)
        self.assertEqual(self.ex(rt, "fs_read", path="../config/vault.json").error_kind, "denied")
        self.assertEqual(self.ex(rt, "fs_read", path="inbox/escape/hostname").error_kind, "denied")
        self.assertEqual(self.ex(rt, "fs_write", path="knowledge/expense_policy.md", content="x").error_kind, "denied")
        self.assertEqual(self.ex(rt, "fs_write", path="../config/permissions.json", content="{}").error_kind, "denied")
        self.assertTrue(self.ex(rt, "fs_write", path="out/ok.md", content="fine").ok)
        self.assertEqual(self.ex(rt, "fs_write", path="out/../knowledge/x.md", content="x").error_kind, "denied")

    def test_login_never_exposes_credentials_to_the_model(self):
        rt = self.runtime()
        r = self.ex(rt, "browser_login", system="expense_portal")
        self.assertTrue(r.ok, r.content)
        blob = json.dumps([r.content, r.data])
        self.assertNotIn("pw-priya-demo", blob)
        self.assertNotIn("audit-token-demo-only", blob)

    def test_unknown_tools_and_bad_arguments_are_reported_not_raised(self):
        rt = self.runtime()
        self.assertEqual(self.ex(rt, "rm_rf", path="/").error_kind, "invalid")
        self.assertEqual(self.ex(rt, "fs_read").error_kind, "invalid")
        self.assertEqual(self.ex(rt, "fs_read", path=5).error_kind, "invalid")
        self.assertEqual(self.ex(rt, "browser_click", link_id="L99").error_kind, "invalid")

    def test_submit_is_gated_and_a_declined_approval_is_not_retried_around(self):
        rt = self.runtime(human=ScriptedHuman(approve_all=False))
        self.ex(rt, "browser_login", system="expense_portal")
        # build a real draft report as the operator would
        from tests.helpers import make_draft_report
        rid = make_draft_report(self.portal, title="Gate test")
        self.ex(rt, "browser_goto", url=f"{self.portal.url}/reports/{rid}")
        r = self.ex(rt, "browser_submit_form", form_id="F1", values={})
        self.assertEqual(r.error_kind, "not_approved")
        con = __import__("sqlite3").connect(self.portal.db_path)
        self.assertEqual(con.execute("SELECT status FROM reports WHERE id=?", (rid,)).fetchone()[0], "draft")
        con.close()


class TestApprovalSemantics(PortalCase):
    def ex(self, rt, tool, **args):
        return rt.agent.executor.execute(tool, args)

    def test_expired_session_bounce_does_not_spend_the_humans_approval(self):
        from tests.helpers import make_draft_report
        human = ScriptedHuman(approve_all=True)
        rid = make_draft_report(self.portal, title="Approval semantics")
        self.portal.set_faults(session_expire_after=2)      # login(/reports)=1, open report=2, submit=3 -> bounced
        rt = self.runtime(human=human)
        self.ex(rt, "browser_login", system="expense_portal")
        self.ex(rt, "browser_goto", url=f"{self.portal.url}/reports/{rid}")
        r1 = self.ex(rt, "browser_submit_form", form_id="F1", values={})
        self.assertTrue(r1.data["session_expired"], r1.content)
        self.assertEqual(len(human.seen_requests), 1)
        con = __import__("sqlite3").connect(self.portal.db_path)
        self.assertEqual(con.execute("SELECT status FROM reports WHERE id=?", (rid,)).fetchone()[0], "draft")
        con.close()
        # sign in again and repeat: same action, same approval, so the human is not asked a second time
        self.ex(rt, "browser_login", system="expense_portal")
        self.ex(rt, "browser_goto", url=f"{self.portal.url}/reports/{rid}")
        r2 = self.ex(rt, "browser_submit_form", form_id="F1", values={})
        self.assertEqual(r2.data["mutation_outcome"], "ok", r2.content)
        self.assertEqual(len(human.seen_requests), 1, "approval must survive the bounce")
        self.assertEqual([m.outcome for m in rt.state.mutations], ["rejected", "ok"])

    def test_approval_is_spent_after_the_system_evaluates_the_action(self):
        from tests.helpers import make_draft_report
        human = ScriptedHuman(approve_all=True)
        rid = make_draft_report(self.portal, title="Spent approval", with_line=False)   # no lines: portal rejects submit
        rt = self.runtime(human=human)
        self.ex(rt, "browser_login", system="expense_portal")
        self.ex(rt, "browser_goto", url=f"{self.portal.url}/reports/{rid}")
        r1 = self.ex(rt, "browser_submit_form", form_id="F1", values={})
        self.assertEqual(r1.data["status"], 422)
        self.ex(rt, "browser_goto", url=f"{self.portal.url}/reports/{rid}")      # the error page has no form: reopen
        self.ex(rt, "browser_submit_form", form_id="F1", values={})
        self.assertEqual(len(human.seen_requests), 2, "a business rejection consumes the approval")


class TestPolicyRules(unittest.TestCase):
    def test_first_match_wins_and_default_is_ask(self):
        pol = PermissionPolicy([{"pattern": "a:x*", "decision": "deny"}, {"pattern": "a:*", "decision": "allow"}],
                               default="ask")
        self.assertEqual(pol.evaluate("a:xyz").verdict, "deny")
        self.assertEqual(pol.evaluate("a:other").verdict, "allow")
        self.assertEqual(pol.evaluate("b:thing").verdict, "ask")


# ------------------------------------------------------------------------------------------- verifier
class TestVerifierCatchesFalseCompletion(PortalCase):
    def test_claiming_success_without_doing_the_work_fails(self):
        def liar():
            for _ in range(5):
                r = yield call("finish", status="done", summary="All done, report submitted!", claims=[
                    {"description": "report submitted", "type": "json_check", "system": "expense_portal",
                     "path": "/audit/reports/1.json", "expect": {"status": "submitted"}}])
        rt = self.runtime(liar)
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertIn("could not be confirmed", rt.state.final["summary"])
        self.assertEqual(rt.state.verification_attempts, 3)
        self.assertIsNotNone(rt.agent.verification)
        self.assertFalse(rt.agent.verification.passed)
        report = (rt.state.run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("FAILED - not completed", report)

    def test_vague_claim_cannot_cover_a_resource_that_was_changed(self):
        def sneaky():
            yield call("browser_login", system="expense_portal")
            r = yield call("browser_goto", url=f"{self.portal.url}/reports/new")
            fid = r.data["forms"][0]["id"]
            yield call("browser_submit_form", form_id=fid, values={"title": "Sneaky", "cost_center": "CC-1042"})
            yield call("finish", status="done", summary="created a report", claims=[
                {"description": "portal reachable", "type": "web_contains", "system": "expense_portal",
                 "path": "/reports", "contains": ["My expense reports"]}])
        rt = self.runtime(sneaky)
        rt.run()
        vr = rt.agent.verification
        self.assertFalse(vr.passed)
        self.assertTrue(any("not covered" in p for p in vr.problems), vr.problems)
        self.assertNotEqual(rt.state.status, "completed")

    def test_claim_failure_feedback_reaches_the_model_and_it_can_recover(self):
        def recovers():
            yield call("fs_write", path="out/note.md", content="hello world")
            r = yield call("finish", status="done", summary="wrote a note", claims=[
                {"description": "note has the text", "type": "file_contains", "path": "out/note.md",
                 "contains": ["goodbye"]}])
            self.assertEqual(r.error_kind, "verification_failed")
            self.assertIn("missing text: ['goodbye']", r.content)
            yield call("finish", status="done", summary="wrote a note", claims=[
                {"description": "note has the text", "type": "file_contains", "path": "out/note.md",
                 "contains": ["hello"]}])
        rt = self.runtime(recovers)
        rt.run()
        self.assertEqual(rt.state.status, "completed", rt.state.final)
        self.assertEqual(rt.state.verification_attempts, 2)


class TestAuditor(PortalCase):
    class Fixed:
        name = "fixed"

        def __init__(self, text):
            self.text, self.calls = text, 0

        def generate(self, system, messages, tools, state):
            self.calls += 1
            return AssistantTurn(self.text, [], [{"type": "text", "text": self.text}])

    def policy(self):
        def p():
            yield call("fs_write", path="out/n.md", content="abc")
            yield call("finish", status="done", summary="wrote it", claims=[
                {"description": "file", "type": "file_contains", "path": "out/n.md", "contains": ["abc"]}])
        return p

    def test_auditor_veto_blocks_completion(self):
        aud = self.Fixed('{"sufficient": false, "gaps": ["proves a file exists, not that the request was met"]}')
        rt = self.runtime(self.policy(), auditor_backend=aud)
        rt.run()
        self.assertNotEqual(rt.state.status, "completed")
        self.assertTrue(any("Independent auditor" in p for p in rt.agent.verification.problems))

    def test_auditor_approval_allows_completion(self):
        rt = self.runtime(self.policy(), auditor_backend=self.Fixed('{"sufficient": true, "gaps": []}'))
        rt.run()
        self.assertEqual(rt.state.status, "completed")

    def test_unusable_auditor_reply_fails_open_but_is_logged(self):
        rt = self.runtime(self.policy(), auditor_backend=self.Fixed("I think it looks fine!"))
        rt.run()
        self.assertEqual(rt.state.status, "completed")
        self.assertTrue([e for e in rt.state.events() if e["type"] == "auditor_inconclusive"])


# -------------------------------------------------------------------------------- Claude API client
class FakeAnthropicAPI:
    """A local stand-in for POST /v1/messages that enforces the transcript rules of the real API and plays a
    scripted sequence of replies (or failures)."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        app = Flask(__name__)

        @app.post("/v1/messages")
        def messages():
            body = request.get_json()
            self.requests.append({"headers": dict(request.headers), "body": body})
            err = self.validate(body)
            if err:
                return jsonify(type="error", error={"type": "invalid_request_error", "message": err}), 400
            step = self.script.pop(0)
            if isinstance(step, int):
                return jsonify(type="error", error={"type": "overloaded_error", "message": "busy"}), step, \
                    {"retry-after": "0"}
            return jsonify(id="msg_1", type="message", role="assistant", model=body["model"], stop_reason="tool_use",
                           content=step, usage={"input_tokens": 10, "output_tokens": 5})

        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()

    @staticmethod
    def validate(body) -> str | None:
        for key in ("model", "max_tokens", "messages"):
            if key not in body:
                return f"missing {key}"
        msgs = body["messages"]
        if not msgs or msgs[0]["role"] != "user":
            return "first message must be from the user"
        for i, m in enumerate(msgs):
            if i and m["role"] == msgs[i - 1]["role"]:
                return f"messages {i - 1} and {i} have the same role"
            if m["role"] == "assistant":
                ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
                if ids:
                    nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                    got = [b["tool_use_id"] for b in (nxt or {}).get("content", []) if b.get("type") == "tool_result"]
                    if got != ids:
                        return f"tool_use ids {ids} were not answered by tool_result blocks {got}"
        for t in body.get("tools", []):
            sch = t.get("input_schema", {})
            if sch.get("type") != "object" or sch.get("required") == []:
                return f"bad input_schema for tool {t.get('name')}"
        return None


def tu(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


class TestAnthropicBackend(PortalCase):
    def backend(self, api, **kw):
        return AnthropicBackend(api_key="test-key", base_url=api.url, sleep=lambda s: None, **kw)

    def test_full_run_over_the_wire_including_overload_retry_and_batched_calls_after_finish(self):
        api = FakeAnthropicAPI([
            529,                                              # overloaded -> client must retry transparently
            [{"type": "text", "text": "Let me look at the policy."}, tu("t1", "kb_search", query="expense policy")],
            # the model batches calls AFTER finish in the same turn; finish(done) with no claims must be refused
            [tu("t2", "fs_list", path="inbox"), tu("t3", "finish", status="done", summary="done?"),
             tu("t4", "fs_list", path="knowledge")],
            [tu("t5", "finish", status="blocked", summary="Cannot proceed in this test.")],
        ])
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="Do the thing")
        rt.run()
        self.assertEqual(rt.state.status, "blocked", rt.state.final)
        self.assertEqual(len(api.requests), 4, "1 overloaded + 3 real turns")
        first = api.requests[1]
        self.assertEqual(first["headers"]["X-Api-Key"], "test-key")
        self.assertEqual(first["headers"]["Anthropic-Version"], "2023-06-01")
        self.assertEqual({t["name"] for t in first["body"]["tools"]} >= {"finish", "fs_read", "browser_goto"}, True)
        self.assertIn("autonomous operator", first["body"]["system"])
        self.assertIn("Do the thing", json.dumps(first["body"]["messages"][0]))
        # last request shows all three results of the batched turn (the API would have returned 400 otherwise)
        last_user = api.requests[3]["body"]["messages"][-1]["content"]
        results = [b for b in last_user if b["type"] == "tool_result"]
        self.assertEqual([b["tool_use_id"] for b in results], ["t2", "t3", "t4"])
        self.assertTrue(results[1]["is_error"])
        self.assertIn("No claims", results[1]["content"])
        self.assertIn("Not executed", results[2]["content"])
        # the platform's working memory (plan, today's date, ledger) is attached to what the model sees
        self.assertIn("WORKING MEMORY", json.dumps(api.requests[3]["body"]["messages"][-1]["content"]))
        self.assertIn(f"Today's date: {TODAY}", json.dumps(api.requests[3]["body"]["messages"][-1]["content"]))
        assert_valid_transcript(rt.agent.messages)

    def test_hard_api_error_stops_the_run_cleanly_and_says_why(self):
        api = FakeAnthropicAPI([])       # empty script + a body that fails validation would 400; force with bad key use
        self.addCleanup(api.stop)
        b = self.backend(api)
        # sabotage: an invalid request (first message from assistant) is a non-retryable 400
        with self.assertRaises(LLMError) as cm:
            b.generate("sys", [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}], [], RunState(
                "g", {}, new_runs_dir()))
        self.assertIn("HTTP 400", str(cm.exception))
        self.assertEqual(len(api.requests), 1, "a 400 must not be retried")

    def test_persistent_overload_gives_up_after_bounded_retries(self):
        api = FakeAnthropicAPI([529] * 10)
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api, max_retries=2), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertIn("could not be reached", rt.state.final["summary"])
        self.assertEqual(len(api.requests), 3)
        self.assertEqual(len([e for e in rt.state.events() if e["type"] == "llm_retry"]), 2)

    def test_missing_api_key_is_a_clear_error(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(LLMError) as cm:
                AnthropicBackend()
            self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


# --------------------------------------------------------------------------- context handling & knowledge
class TestContext(PortalCase):
    def test_old_observations_are_shortened_but_recent_ones_and_memory_survive(self):
        rt = self.runtime()
        a = rt.agent
        a.messages = [{"role": "user", "content": [{"type": "text", "text": "goal"}]}]
        for i in range(10):
            a.messages.append({"role": "assistant", "content": [tu(f"t{i}", "fs_read", path="x")]})
            a.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"RESULT{i} " + "x" * 2000}]})
        view = a._view()
        blob = [b for m in view if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
        self.assertIn("older observation shortened", blob[0]["content"])
        self.assertNotIn("older observation shortened", blob[-1]["content"])
        self.assertEqual(len(a.messages[2]["content"][0]["content"]), len("RESULT0 ") + 2000, "full transcript kept")
        self.assertIn("WORKING MEMORY", json.dumps(view[-1]))

    def test_knowledge_search_finds_the_procedure_for_a_vague_request(self):
        kb = KnowledgeBase(Sandbox(fresh_company() / "fs"))
        top = kb.search("how do I get reimbursed for a business trip")[0]
        self.assertIn("expense", top["path"])
        self.assertEqual(kb.search("zzzz qqqq"), [])


class TestOracleIsNotATautology(PortalCase):
    def test_grader_rejects_the_wrong_task_and_the_wrong_status(self):
        rt = run_expense(self.portal)
        self.assertEqual(rt.state.status, "completed")
        self.assertTrue(grade(self.portal.db_path, "expense_mumbai").passed)
        self.assertFalse(grade(self.portal.db_path, "expense_pune").passed)
        self.assertFalse(grade(self.portal.db_path, "expense_mumbai", expect_status="draft").passed)


if __name__ == "__main__":
    unittest.main()
