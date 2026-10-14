from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from company_env.launch import copy_company, start_portal  # noqa: E402
from operator_agent.human import Human, ScriptedHuman  # noqa: E402
from operator_agent.offline import GeneratorBackend, expense_standin  # noqa: E402
from operator_agent.runtime import build_runtime  # noqa: E402

GOAL = "Submit my expenses for the Mumbai trip."
TODAY = "2026-09-20"


def new_runs_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="operator_runs_"))


def fresh_company() -> Path:
    """A private copy of the company file share + config, so runs never write into the repository."""
    return copy_company(Path(tempfile.mkdtemp(prefix="company_")))


def run_expense(portal, human: Human | None = None, goal: str = GOAL, backend=None, runs_dir: Path | None = None,
                max_steps: int = 120, auditor_backend=None, company: Path | None = None):
    backend = backend or GeneratorBackend(expense_standin(goal))
    human = human or ScriptedHuman(approve_all=True)
    rt = build_runtime(company or fresh_company(), runs_dir or new_runs_dir(), goal, "priya.sharma", backend, human,
                       portal.url, max_steps=max_steps, read_backoff=0, auditor_backend=auditor_backend, today=TODAY)
    rt.run()
    return rt


def portal_session(portal, username="priya.sharma", password="pw-priya-demo") -> requests.Session:
    s = requests.Session()
    s.post(portal.url + "/login", data={"username": username, "password": password})
    return s


def csrf(html: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def make_draft_report(portal, title="Scratch report", cost_center="CC-2207", with_line=True) -> int:
    """Create a draft report directly (as a user would), bypassing the operator."""
    s = portal_session(portal)
    t = csrf(s.get(portal.url + "/reports/new").text)
    r = s.post(portal.url + "/reports", data={"title": title, "cost_center": cost_center, "csrf_token": t})
    rid = int(r.url.split("/reports/")[1].split("?")[0])
    if with_line:
        t = csrf(s.get(f"{portal.url}/reports/{rid}/lines/new").text)
        s.post(f"{portal.url}/reports/{rid}/lines", data=dict(
            date="2026-09-14", category="Transport", merchant="Test Cab", amount="500",
            receipt_ref="scratch-1", nights="1", note="", csrf_token=t))
    return rid


def assert_valid_transcript(messages: list[dict]) -> None:
    """Every assistant tool_use must be answered by a tool_result with the same id in the next user message."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = messages[i + 1]
        assert nxt["role"] == "user", f"message {i + 1} should be a user message"
        got = [b["tool_use_id"] for b in nxt["content"] if b.get("type") == "tool_result"]
        assert got == ids, f"tool_use/tool_result mismatch at message {i}: {ids} vs {got}"
