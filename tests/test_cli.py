"""The CLI wiring, end to end, with the Claude API replaced by a local fake: run -> pause for approval (exit 3)
-> resume in a fresh portal/process-equivalent -> completes the gated action. Real model quality is NOT tested here."""
import contextlib
import io
import json
import os
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path

from operator_agent.cli import main
from tests.test_platform import FakeAnthropicAPI, tu


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestCLI(unittest.TestCase):
    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_run_pauses_for_approval_then_resume_completes_it(self):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        api = FakeAnthropicAPI([
            [tu("a1", "browser_login", system="expense_portal")],
            [tu("a2", "browser_goto", url=f"{base}/reports/new")],
            [tu("a3", "browser_submit_form", form_id="F1", values={"title": "CLI test", "cost_center": "CC-1042"})],
            [tu("a4", "browser_goto", url=f"{base}/reports/1/lines/new")],
            [tu("a5", "browser_submit_form", form_id="F1", values={
                "date": "2026-09-14", "category": "Transport", "merchant": "Cab", "amount": "100",
                "receipt_ref": "r1", "nights": "1", "note": ""})],
            [tu("a6", "browser_goto", url=f"{base}/reports/1")],
            [tu("a7", "browser_submit_form", form_id="F1", values={})],          # gated -> pause
            # --- after resume: the portal restarted (old cookies invalid), so the page the paused call referred to
            # is gone. The platform refuses to run it against the login form; the model signs in, reopens the
            # report and repeats the action.
            [tu("a8", "browser_login", system="expense_portal")],
            [tu("a9", "browser_goto", url=f"{base}/reports/1")],
            [tu("a10", "browser_submit_form", form_id="F1", values={})],
            [tu("a11", "finish", status="blocked", summary="Stopping after the test action.")],
        ])
        self.addCleanup(api.stop)
        runs = tempfile.mkdtemp(prefix="cli_runs_")
        env = {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_BASE_URL": api.url}
        with unittest.mock.patch.dict(os.environ, env):
            code, out = self.cli("run", "Submit a report", "--runs-dir", runs, "--run-id", "t1", "--human", "defer",
                                 "--port", str(port))
            self.assertEqual(code, 3, out)
            self.assertIn("PAUSED", out)
            self.assertIn("resume t1", out)
            run_dir = Path(runs) / "t1"
            for f in ("run_meta.json", "checkpoint.json", "events.jsonl", "report.md", "portal.db"):
                self.assertTrue((run_dir / f).exists(), f)
            con = sqlite3.connect(run_dir / "portal.db")
            self.assertEqual(con.execute("SELECT status FROM reports").fetchone()[0], "draft", "nothing irreversible yet")
            con.close()

            code, out = self.cli("resume", "t1", "--runs-dir", runs, "--human", "auto")
            self.assertEqual(code, 2, out)               # the scripted model then reports 'blocked'
            self.assertIn("BLOCKED", out)
            con = sqlite3.connect(run_dir / "portal.db")
            self.assertEqual(con.execute("SELECT status FROM reports").fetchone()[0], "submitted")
            con.close()
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual([a["decision"] for a in result["approvals"]], ["defer", "approve"],
                             "audit trail: nobody was there at first, then exactly one human approval; the action "
                             "was attempted twice but never re-approved")
            self.assertEqual([m["outcome"] for m in result["mutations"] if "/submit" in m["summary"]], ["ok"])
            events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            stale = [e for e in events if e["type"] == "tool_result" and e.get("error_kind") == "stale_action"]
            self.assertEqual(len(stale), 1, "the paused call must not silently re-bind to the login form")
            self.assertEqual(len(api.requests), 11)

    def test_run_without_api_key_fails_with_a_clear_message(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY", "GROQ_API_KEY",
                      "OPENROUTER_API_KEY", "NVIDIA_API_KEY"):
                os.environ.pop(k, None)
            code, out = self.cli("run", "x", "--runs-dir", tempfile.mkdtemp())
        self.assertEqual(code, 2)
        self.assertIn("ANTHROPIC_API_KEY", out)

    def test_offline_demo_runs_every_scenario_and_matches_ground_truth(self):
        code, out = self.cli("demo", "--runs-dir", tempfile.mkdtemp())
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("PASS "), 4, out)
        self.assertIn("test double", out)


import unittest.mock  # noqa: E402  (kept at the bottom so the module docstring stays first)

if __name__ == "__main__":
    unittest.main()
