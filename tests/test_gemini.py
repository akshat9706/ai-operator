"""The Gemini backend, tested against a local fake of the generateContent API that enforces the rules the real
one enforces (function calls must be answered by the same number of function responses, Gemini 3 thought
signatures must be echoed back, tool schemas must be well formed). Real model quality is NOT tested here."""
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from operator_agent.cli import main
from operator_agent.gemini import GeminiBackend, list_models, probe_model
from operator_agent.llm import LLMError
from operator_agent.state import RunState
from tests.helpers import new_runs_dir
from tests.test_platform import PortalCase


def fc(name, **args):
    return {"functionCall": {"name": name, "args": args}}


def txt(s):
    return {"text": s}


class FakeGeminiAPI:
    """script items: list of parts (a normal reply), or an int / (int, body-dict) for an error status."""

    def __init__(self, script):
        self.script, self.requests, self.sigs = list(script), [], 0
        app = Flask(__name__)

        @app.post("/v1beta/models/<path:rest>")
        def generate(rest):
            body = request.get_json()
            self.requests.append({"headers": dict(request.headers), "body": body, "path": rest})
            err = self.validate(body)
            if err:
                return jsonify(error={"code": 400, "message": err, "status": "INVALID_ARGUMENT"}), 400
            step = self.script.pop(0)
            if isinstance(step, tuple):
                status, extra = step
                return jsonify(error={"code": status, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                                      **extra}), status
            if isinstance(step, int):
                return jsonify(error={"code": step, "message": "busy", "status": "UNAVAILABLE"}), step
            parts, signed = [], False
            for p in step:
                p = json.loads(json.dumps(p))
                if "functionCall" in p and not signed:     # Gemini 3: first call of a step carries a signature
                    self.sigs += 1
                    p["thoughtSignature"] = f"SIG-{self.sigs}"
                    signed = True
                parts.append(p)
            return jsonify(candidates=[{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                           usageMetadata={"promptTokenCount": 10, "candidatesTokenCount": 5})

        @app.get("/v1beta/models")
        def models():
            return jsonify(models=[
                {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/gemma-3-27b", "supportedGenerationMethods": ["generateContent"]},
            ])

        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()

    @staticmethod
    def validate(body):
        if "x-goog-api-key" in body:
            return "api key must be a header"
        if not body.get("systemInstruction", {}).get("parts"):
            return "missing systemInstruction"
        contents = body.get("contents") or []
        if not contents or contents[0]["role"] != "user":
            return "first content must be from the user"
        for i, c in enumerate(contents):
            if not c.get("parts"):
                return f"content {i} has no parts"
            for p in c["parts"]:
                if "text" in p and not p["text"].strip():
                    return f"content {i} has an empty text part"
            calls = [p for p in c["parts"] if "functionCall" in p]
            if c["role"] == "model" and calls:
                if "thoughtSignature" not in calls[0]:
                    return f"function call in content {i} is missing its thought signature"
                nxt = contents[i + 1] if i + 1 < len(contents) else None
                resp = [p["functionResponse"]["name"] for p in (nxt or {}).get("parts", []) if "functionResponse" in p]
                if resp != [p["functionCall"]["name"] for p in calls]:
                    return f"function calls {[p['functionCall']['name'] for p in calls]} answered by {resp}"
                for p in (nxt or {}).get("parts", []):
                    if "functionResponse" in p and not isinstance(p["functionResponse"]["response"], dict):
                        return "functionResponse.response must be an object"
        for t in body.get("tools", []):
            for d in t.get("functionDeclarations", []):
                sch = d.get("parametersJsonSchema")
                if "parameters" in d:
                    return f"{d['name']}: use parametersJsonSchema"
                if sch is not None and (sch.get("type") != "object" or not sch.get("properties")
                                        or sch.get("required") == []):
                    return f"bad schema for tool {d['name']}"
        return None


class TestGeminiBackend(PortalCase):
    def backend(self, api, **kw):
        return GeminiBackend(api_key="test-key", base_url=api.url, sleep=lambda s: None, **kw)

    def test_full_run_over_the_wire_including_retry_signatures_and_batched_calls_after_finish(self):
        api = FakeGeminiAPI([
            503,                                                          # transient -> retried transparently
            (429, {"details": [{"retryDelay": "0s"}]}),                   # per-minute limit -> retried
            [txt("Let me look at the policy."), fc("kb_search", query="expense policy")],
            [fc("fs_list", path="inbox"), fc("finish", status="done", summary="done?"), fc("fs_list", path="knowledge")],
            [fc("list_systems")],                                         # a tool with no parameters
            [fc("finish", status="blocked", summary="Cannot proceed in this test.")],
        ])
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api, model="gemini-3.5-flash"), goal="Do the thing")
        rt.run()
        self.assertEqual(rt.state.status, "blocked", rt.state.final)
        self.assertEqual(len(api.requests), 6, "2 failures + 4 real turns")
        first = api.requests[2]
        self.assertEqual(first["headers"]["X-Goog-Api-Key"], "test-key")
        self.assertTrue(first["path"].startswith("gemini-3.5-flash") or first["path"].startswith("models/gemini-3.5-flash"))
        decls = {d["name"]: d for d in first["body"]["tools"][0]["functionDeclarations"]}
        self.assertTrue({"finish", "fs_read", "browser_submit_form", "list_systems"} <= set(decls))
        self.assertNotIn("parametersJsonSchema", decls["list_systems"])
        self.assertIn("values", decls["browser_submit_form"]["parametersJsonSchema"]["properties"])
        self.assertIn("autonomous operator", first["body"]["systemInstruction"]["parts"][0]["text"])
        # the batched turn: finish without claims is refused, the call after finish is not executed, all three answered
        last = api.requests[4]["body"]["contents"][-1]["parts"]
        resp = [p["functionResponse"] for p in last if "functionResponse" in p]
        self.assertEqual([r["name"] for r in resp], ["fs_list", "finish", "fs_list"])
        self.assertIn("No claims", resp[1]["response"]["error"])
        self.assertIn("Not executed", resp[2]["response"]["error"])
        self.assertIn("WORKING MEMORY", json.dumps(last))
        # signatures issued by the API came back unchanged in every later request
        self.assertIn("SIG-1", json.dumps(api.requests[3]["body"]))
        self.assertIn("SIG-2", json.dumps(api.requests[5]["body"]))

    def test_plan_as_plain_strings_and_odd_arguments_do_not_crash_the_run(self):
        api = FakeGeminiAPI([
            [fc("update_plan", steps=["Read the policy", "Create the report"])],       # strings, not objects
            [fc("update_plan", steps=[{"text": "a"}, 7, None])],                       # mixed junk
            [fc("remember", key="k", value="v")],
            [fc("finish", status="blocked", summary="done testing")],
        ])
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "blocked", rt.state.final)
        self.assertEqual([p["text"] for p in rt.state.plan], ["a", "7", "None"])

    def test_throttle_spaces_calls_out(self):
        api = FakeGeminiAPI([[txt("a")], [txt("b")]])
        self.addCleanup(api.stop)
        waits, now = [], [100.0]
        b = GeminiBackend(api_key="k", base_url=api.url, min_interval=5.0, clock=lambda: now[0],
                          sleep=lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)))
        st = RunState("g", {}, new_runs_dir())
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        b.generate("sys", msgs, [], st)
        b.generate("sys", msgs, [], st)
        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], 5.0)

    def test_daily_quota_stops_immediately_with_a_clear_reason(self):
        api = FakeGeminiAPI([(429, {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]})] * 3)
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertIn("DAILY", rt.state.final["summary"])
        self.assertEqual(len(api.requests), 1, "a spent daily quota must not be retried")

    def test_unknown_model_gives_a_pointer_to_the_models_command(self):
        api = FakeGeminiAPI([(404, {})])
        self.addCleanup(api.stop)
        b = self.backend(api, model="gemini-does-not-exist")
        with self.assertRaises(LLMError) as cm:
            b.generate("sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], RunState("g", {}, new_runs_dir()))
        self.assertIn("models", str(cm.exception))

    def test_persistent_overload_gives_up_after_bounded_retries(self):
        api = FakeGeminiAPI([503] * 10)
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api, max_retries=2), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertEqual(len(api.requests), 3)

    def test_missing_api_key_is_a_clear_error(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            with self.assertRaises(LLMError) as cm:
                GeminiBackend()
        self.assertIn("GEMINI_API_KEY", str(cm.exception))

    def test_list_models_keeps_only_gemini_models_that_can_generate(self):
        api = FakeGeminiAPI([])
        self.addCleanup(api.stop)
        self.assertEqual(list_models(api_key="k", base_url=api.url), ["gemini-3.5-flash"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestGeminiCLI(unittest.TestCase):
    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_gemini_run_pauses_then_resume_rebuilds_the_same_provider_and_finishes(self):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        api = FakeGeminiAPI([
            [fc("browser_login", system="expense_portal")],
            [fc("browser_goto", url=f"{base}/reports/new")],
            [fc("browser_submit_form", form_id="F1", values={"title": "Gemini CLI test", "cost_center": "CC-1042"})],
            [fc("browser_submit_form", form_id="F1", values={})],                       # gated -> pause
            # after resume the portal restarted, so the old session is gone
            [fc("browser_login", system="expense_portal")],
            [fc("browser_goto", url=f"{base}/reports/1")],
            [fc("browser_submit_form", form_id="F1", values={})],
            [fc("finish", status="blocked", summary="Stopping after the test action.")],
        ])
        self.addCleanup(api.stop)
        runs = tempfile.mkdtemp(prefix="gem_cli_")
        env = {"GEMINI_API_KEY": "test-key", "GEMINI_BASE_URL": api.url}
        with unittest.mock.patch.dict(os.environ, env):
            os.environ.pop("ANTHROPIC_API_KEY", None)           # provider 'auto' must pick Gemini on its own
            code, out = self.cli("run", "Submit a report", "--runs-dir", runs, "--run-id", "g1", "--human", "defer",
                                 "--port", str(port), "--min-interval", "0")
            self.assertEqual(code, 3, out)
            self.assertIn("gemini/gemini-3.5-flash", out)
            meta = json.loads((Path(runs) / "g1" / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["provider"], meta["model"]), ("gemini", "gemini-3.5-flash"))
            code, out = self.cli("resume", "g1", "--runs-dir", runs, "--human", "auto")
            self.assertEqual(code, 2, out)
            self.assertIn("BLOCKED", out)
            self.assertEqual(len(api.requests), 8)
            sub = [json.loads(l) for l in (Path(runs) / "g1" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len([e for e in sub if e["type"] == "approval" and e["decision"] == "approve"]), 1)

    def test_models_test_flag_probes_text_models_only(self):
        api = FakeGeminiAPI([[txt("ok")]])
        self.addCleanup(api.stop)
        with unittest.mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            with unittest.mock.patch("operator_agent.cli.list_models", lambda: [
                    "gemini-3.5-flash", "gemini-3.1-flash-image", "gemini-3.8-flash-tts", "gemini-2.5-pro"]), \
                 unittest.mock.patch("operator_agent.cli.probe_model",
                                     lambda n: probe_model(n, api_key="k", base_url=api.url)), \
                 unittest.mock.patch("operator_agent.cli.time.sleep", lambda s: None):
                code, out = self.cli("models", "--test")
        self.assertEqual(code, 0)
        self.assertIn("gemini-3.5-flash", out)
        self.assertIn("WORKS", out)
        self.assertNotIn("image", out)
        self.assertNotIn("tts", out)
        self.assertEqual(len(api.requests), 1, "only the one real text model is probed (pro/image/tts skipped)")

    def test_models_command_lists_models(self):
        api = FakeGeminiAPI([])
        self.addCleanup(api.stop)
        with unittest.mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            with unittest.mock.patch("operator_agent.cli.list_models",
                                     lambda: list_models(api_key="k", base_url=api.url)):
                code, out = self.cli("models")
        self.assertEqual(code, 0)
        self.assertIn("gemini-3.5-flash", out)


if __name__ == "__main__":
    unittest.main()
