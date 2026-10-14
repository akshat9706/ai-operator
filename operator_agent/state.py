"""Run state: what the operator knows, what it changed, and an append-only event log.

State is deliberately plain data so a run can be checkpointed to disk, paused while it
waits for a human, and resumed later (possibly by another process).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    # denied | not_approved | deferred | invalid | unsafe_retry | stale_action | transient | tool_error
    error_kind: str | None = None

    def for_model(self) -> str:
        if self.ok:
            return self.content
        return f"ERROR[{self.error_kind or 'error'}]: {self.content}"


@dataclass
class Mutation:
    seq: int
    fingerprint: str
    resource: str          # e.g. "expense_portal:/reports/3"
    tool: str
    summary: str
    outcome: str           # ok | rejected | ambiguous
    resolved: bool = False  # True once a later read of the same resource re-established the facts


@dataclass
class ApprovalRecord:
    seq: int
    action: str
    reason: str
    decision: str          # approve | deny | defer
    by: str
    note: str = ""


class RunState:
    def __init__(self, goal: str, user: dict[str, Any], run_dir: Path, max_steps: int = 80,
                 today: str | None = None):
        self.goal = goal
        self.today = today          # the harness's clock (ISO date); a model has no reliable idea of "today"
        self.user = user
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.step = 0
        self.seq = 0
        self.status = "running"   # running|completed|blocked|failed|waiting_for_approval|waiting_for_input|stalled
        self.plan: list[dict[str, str]] = []
        self.facts: dict[str, str] = {}
        self.mutations: list[Mutation] = []
        self.reads: list[tuple[int, str]] = []          # (seq, resource) for every successful read
        self.approved: set[str] = set()                  # fingerprints approved and not yet consumed
        self.approvals: list[ApprovalRecord] = []
        self.pending: list[dict[str, Any]] = []          # tool calls waiting on a human
        self.consecutive_errors = 0
        self.verification_attempts = 0
        self.last_results: list[ToolResult] = []         # results of the previous turn (offline policies use this)
        self.final: dict[str, Any] = {}
        self._events_path = self.run_dir / "events.jsonl"
        self.listener = None        # optional callback(record) for live progress display

    # ------------------------------------------------------------------ log
    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def log(self, type_: str, **data: Any) -> None:
        rec = {"ts": round(time.time(), 3), "step": self.step, "type": type_, **data}
        with self._events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if self.listener:
            self.listener(rec)

    def events(self) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        return [json.loads(l) for l in self._events_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ------------------------------------------------- mutation bookkeeping
    def record_read(self, resource: str) -> None:
        self.reads.append((self.seq, resource))
        for m in self.mutations:
            if m.outcome == "ambiguous" and not m.resolved and m.resource == resource:
                m.resolved = True

    def unresolved_ambiguous(self) -> list[Mutation]:
        return [m for m in self.mutations if m.outcome == "ambiguous" and not m.resolved]

    # ------------------------------------------------------ working memory
    def working_memory(self) -> str:
        lines = ["=== WORKING MEMORY (kept by the platform; trust it over older messages) ===",
                 f"Goal: {self.goal}",
                 f"Requesting employee: {self.user.get('name')} ({self.user.get('username')})",
                 f"Step {self.step}/{self.max_steps}"]
        if self.today:
            lines.insert(2, f"Today's date: {self.today} (use it for any deadline or time-window rule)")
        if self.plan:
            lines.append("Plan:")
            mark = {"done": "x", "in_progress": ">", "pending": " ", "blocked": "!"}
            lines += [f"  [{mark.get(p.get('status', 'pending'), ' ')}] {p.get('text', '')}" for p in self.plan]
        else:
            lines.append("Plan: (none yet - call update_plan first)")
        if self.facts:
            lines.append("Facts you recorded:")
            lines += [f"  {k}: {v}" for k, v in self.facts.items()]
        if self.mutations:
            lines.append("State-changing actions so far (platform ledger):")
            for m in self.mutations[-12:]:
                flag = ""
                if m.outcome == "ambiguous":
                    flag = " (RE-READ the resource to learn whether it took effect)" if not m.resolved else " (resolved)"
                lines.append(f"  #{m.seq} {m.summary} -> {m.outcome}{flag}")
        return "\n".join(lines)

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "user": self.user, "max_steps": self.max_steps, "today": self.today,
            "step": self.step,
            "seq": self.seq, "status": self.status, "plan": self.plan, "facts": self.facts,
            "mutations": [asdict(m) for m in self.mutations], "reads": self.reads,
            "approved": sorted(self.approved), "approvals": [asdict(a) for a in self.approvals],
            "pending": self.pending, "consecutive_errors": self.consecutive_errors,
            "verification_attempts": self.verification_attempts, "final": self.final,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], run_dir: Path) -> "RunState":
        s = cls(d["goal"], d["user"], run_dir, d.get("max_steps", 80), d.get("today"))
        for k in ("step", "seq", "status", "plan", "facts", "pending", "consecutive_errors",
                  "verification_attempts", "final"):
            setattr(s, k, d[k])
        s.mutations = [Mutation(**m) for m in d["mutations"]]
        s.reads = [tuple(r) for r in d["reads"]]
        s.approved = set(d["approved"])
        s.approvals = [ApprovalRecord(**a) for a in d["approvals"]]
        return s
