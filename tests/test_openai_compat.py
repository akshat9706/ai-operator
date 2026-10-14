"""The OpenAI-compatible backend (Groq, Cerebras, OpenRouter, Ollama, ...), tested against a local fake of the
chat-completions API that enforces the transcript rules real servers enforce. Real model quality is NOT tested."""
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from operator_agent.cli import main
from operator_agent.llm import LLMError
from operator_agent.openai_compat import OpenAICompatBackend, list_openai_models
from operator_agent.state import RunState
from tests.helpers import new_runs_dir
from tests.test_platform import PortalCase


def reply(text=None, calls=(), raw_args=None):
    """A normal assistant answer. raw_args overrides the JSON string of every call (to simulate bad JSON)."""
    tcs = [{"id": f"call_{i}", "type": "function",
            "function": {"name": n, "arguments": raw_args if raw_args is not None else json.dumps(a)}}
           for i, (n, a) in enumerate(calls)]
    msg = {"role": "assistant", "content": text}
    if tcs:
        msg["tool_calls"] = tcs
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tcs else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeChatAPI:
    """script items: a reply dict, or (status, body_text, headers) for an error."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        app = Flask(__name__)

        @app.post("/v1/chat/completions")
        def chat():
            body = request.get_json()
            self.requests.append({"headers": dict(request.headers), "body": body})
            err = self.validate(body)
            if err:
                return jsonify(error={"message": err}), 400
            step = self.script.pop(0)
            if isinstance(step, tuple):
                status, text, headers = step
                return text, status, {"content-type": "application/json", **headers}
            return jsonify(step)

        @app.get("/v1/models")
        def models():
            return jsonify(data=[{"id": "llama-test-70b"}, {"id": "qwen-test-32b"}])

        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def stop(self):
        self.server.shutdown()

    @staticmethod
    def validate(body):
        msgs = body.get("messages") or []
        if not msgs or msgs[0]["role"] != "system":
            return "first message must be the system prompt"
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls"):
                return f"assistant message {i} is empty"
            if m["role"] == "assistant" and m.get("tool_calls"):
                ids = [t["id"] for t in m["tool_calls"]]
                for t in m["tool_calls"]:
                    if not isinstance(t["function"]["arguments"], str):
                        return "tool call arguments must be a JSON string"
                    json.loads(t["function"]["arguments"])
                got = [x["tool_call_id"] for x in msgs[i + 1:i + 1 + len(ids)] if x["role"] == "tool"]
                if got != ids:
                    return f"tool calls {ids} answered by {got}"
            if m["role"] == "tool" and not isinstance(m["content"], str):
                return "tool content must be a string"
        for t in body.get("tools", []):
            if t.get("type") != "function" or not t["function"].get("name"):
                return "bad tool declaration"
        return None


class TestOpenAICompat(PortalCase):
    def backend(self, api, **kw):
        return OpenAICompatBackend("groq", api_key="test-key", model="test-model", base_url=api.url,
                                   sleep=lambda s: None, **kw)

    def test_full_run_over_the_wire(self):
        api = FakeChatAPI([
            (503, '{"error":{"message":"busy"}}', {}),
            (429, '{"error":{"message":"slow down"}}', {"retry-after": "0"}),
            reply("Let me check.", [("kb_search", {"query": "expense policy"})]),
            reply(None, [("fs_list", {"path": "inbox"}), ("finish", {"status": "done", "summary": "x"}),
                         ("fs_list", {"path": "knowledge"})]),
            reply(None, [("fs_read", {})], raw_args="{not json"),             # broken JSON from a weak model
            reply(None, [("finish", {"status": "blocked", "summary": "Cannot proceed in this test."})]),
        ])
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="Do the thing")
        rt.run()
        self.assertEqual(rt.state.status, "blocked", rt.state.final)
        self.assertEqual(len(api.requests), 6)
        first = api.requests[2]
        self.assertEqual(first["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(first["body"]["model"], "test-model")
        names = {t["function"]["name"] for t in first["body"]["tools"]}
        self.assertTrue({"finish", "fs_read", "browser_submit_form", "list_systems"} <= names)
        msgs = api.requests[4]["body"]["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        self.assertTrue(any("No claims" in m["content"] for m in tool_msgs))
        self.assertTrue(any("Not executed" in m["content"] for m in tool_msgs))
        self.assertTrue(any("WORKING MEMORY" in str(m.get("content")) for m in msgs if m["role"] == "user"))
        last = api.requests[5]["body"]["messages"]
        self.assertTrue(any("Missing required argument" in m["content"] for m in last if m["role"] == "tool"))

    def test_daily_quota_stops_immediately(self):
        api = FakeChatAPI([(429, '{"error":{"message":"Rate limit reached for tokens per day (TPD): Limit 200000"}}', {})] * 3)
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "failed")
        self.assertIn("DAILY", rt.state.final["summary"])
        self.assertEqual(len(api.requests), 1)

    def test_malformed_tool_calls_from_the_server_do_not_kill_the_run(self):
        bad = (400, '{"error":{"message":"Failed to call a function","code":"tool_use_failed"}}', {})
        api = FakeChatAPI([bad, bad, bad,
                           reply(None, [("finish", {"status": "blocked", "summary": "Stopping after the nudge."})])])
        self.addCleanup(api.stop)
        rt = self.runtime(backend=self.backend(api), goal="x")
        rt.run()
        self.assertEqual(rt.state.status, "blocked", rt.state.final)
        self.assertIn("You did not call a tool", json.dumps(api.requests[3]["body"]["messages"]))

    def test_wrong_key_and_unknown_model_have_clear_messages(self):
        api = FakeChatAPI([(401, '{"error":{"message":"invalid api key"}}', {}),
                           (404, '{"error":{"message":"model not found"}}', {})])
        self.addCleanup(api.stop)
        st = RunState("g", {}, new_runs_dir())
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        with self.assertRaises(LLMError) as cm:
            self.backend(api).generate("sys", msgs, [], st)
        self.assertIn(".env", str(cm.exception))
        with self.assertRaises(LLMError) as cm:
            self.backend(api).generate("sys", msgs, [], st)
        self.assertIn("models --provider groq", str(cm.exception))

    def test_construction_errors(self):
        with unittest.mock.patch.dict(os.environ, {}):
            os.environ.pop("CEREBRAS_API_KEY", None)
            with self.assertRaises(LLMError) as cm:
                OpenAICompatBackend("cerebras", model="m")
            self.assertIn("CEREBRAS_API_KEY", str(cm.exception))
            with self.assertRaises(LLMError) as cm:
                OpenAICompatBackend("cerebras", api_key="k")
            self.assertIn("--model", str(cm.exception))
            OpenAICompatBackend("ollama", model="m")                 # local servers need no key

    def test_list_models(self):
        api = FakeChatAPI([])
        self.addCleanup(api.stop)
        with unittest.mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            self.assertEqual(list_openai_models("groq", api.url), ["llama-test-70b", "qwen-test-32b"])


class TestOpenAICompatCLI(unittest.TestCase):
    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_run_and_models_through_the_cli_with_a_local_server(self):
        api = FakeChatAPI([reply(None, [("finish", {"status": "blocked", "summary": "Test stop."})])])
        self.addCleanup(api.stop)
        runs = tempfile.mkdtemp(prefix="oai_cli_")
        code, out = self.cli("run", "Do something", "--provider", "ollama", "--model", "qwen-test-32b",
                             "--base-url", api.url, "--runs-dir", runs, "--run-id", "o1", "--human", "defer",
                             "--min-interval", "0")
        self.assertEqual(code, 2, out)
        self.assertIn("ollama/qwen-test-32b", out)
        meta = json.loads((Path(runs) / "o1" / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["provider"], meta["model"], meta["base_url"]), ("ollama", "qwen-test-32b", api.url))
        code, out = self.cli("models", "--provider", "ollama", "--base-url", api.url)
        self.assertEqual(code, 0, out)
        self.assertIn("llama-test-70b", out)

    def test_missing_model_is_explained(self):
        with unittest.mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
            code, out = self.cli("run", "x", "--provider", "cerebras", "--runs-dir", tempfile.mkdtemp())
        self.assertEqual(code, 2)
        self.assertIn("--model", out)

    def test_auto_provider_prefers_what_you_have_a_key_for(self):
        from operator_agent.cli import resolve_provider
        keys = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY", "GROQ_API_KEY",
                "OPENROUTER_API_KEY", "NVIDIA_API_KEY")
        with unittest.mock.patch.dict(os.environ, {}):
            for k in keys:
                os.environ.pop(k, None)
            self.assertEqual(resolve_provider("auto"), "anthropic")
            os.environ["GROQ_API_KEY"] = "k"
            self.assertEqual(resolve_provider("auto"), "groq")
            os.environ["CEREBRAS_API_KEY"] = "k"
            self.assertEqual(resolve_provider("auto"), "cerebras")
            os.environ["GEMINI_API_KEY"] = "k"
            self.assertEqual(resolve_provider("auto"), "gemini")
            self.assertEqual(resolve_provider("groq"), "groq")


if __name__ == "__main__":
    unittest.main()
