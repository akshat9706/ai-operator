"""Wiring: build a ready-to-run agent for a company environment."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .agent import Agent
from .browser import Browser
from .context import KnowledgeBase, Sandbox, Systems
from .executor import Executor
from .human import Human
from .llm import LLMBackend
from .permissions import PermissionPolicy
from .report import write_report
from .state import RunState
from .tools import ToolContext, build_tools
from .verifier import LLMAuditor, Verifier


@dataclass
class Runtime:
    agent: Agent
    state: RunState
    ctx: ToolContext
    sandbox: Sandbox

    def run(self) -> RunState:
        st = self.agent.run()
        write_report(st, self.agent.verification)
        return st


def load_user(company_root: Path, username: str) -> dict:
    for e in json.loads((Path(company_root) / "fs" / "directory" / "employees.json").read_text(encoding="utf-8")):
        if e["username"] == username:
            return {"username": e["username"], "name": e["name"], "email": e["email"],
                    "employee_id": e["employee_id"]}
    raise SystemExit(f"Unknown employee '{username}'.")


def build_runtime(company_root: Path, runs_dir: Path, goal: str, username: str, backend: LLMBackend,
                  human: Human, portal_url: str, max_steps: int = 80, run_id: str | None = None,
                  resume_data: dict | None = None, read_backoff: float = 0.3,
                  auditor_backend: LLMBackend | None = None, today: str | None = None) -> Runtime:
    company_root = Path(company_root)
    placeholders = {"PORTAL": portal_url.rstrip("/")}
    if resume_data:
        run_dir = Path(runs_dir) / run_id  # type: ignore[operator]
        state = RunState.from_dict(resume_data["state"], run_dir)
    else:
        run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(runs_dir) / run_id
        state = RunState(goal, load_user(company_root, username), run_dir, max_steps, today)
    sandbox = Sandbox(company_root / "fs")
    systems = Systems.load(company_root / "config", placeholders)
    policy = PermissionPolicy.load(company_root / "config" / "permissions.json", placeholders)
    browser = Browser()
    ctx = ToolContext(sandbox, KnowledgeBase(sandbox), systems, browser, state, human)
    tools = build_tools()
    executor = Executor(tools, ctx, policy, backoff=read_backoff)
    verifier = Verifier(systems, sandbox, state, run_dir / "evidence")
    auditor = LLMAuditor(auditor_backend) if auditor_backend is not None else None
    agent = Agent(backend, executor, verifier, state, tools, browser=browser, auditor=auditor)
    if resume_data:
        agent.restore(resume_data)
    return Runtime(agent, state, ctx, sandbox)
