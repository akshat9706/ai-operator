"""Final evidence report: what was asked, what happened, what verified it, what a human decided."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .state import RunState
from .verifier import VerificationResult

HEADLINE = {
    "completed": "COMPLETED and independently verified",
    "blocked": "BLOCKED - needs a human",
    "failed": "FAILED - not completed",
    "stalled": "STALLED - the operator stopped without finishing",
    "waiting_for_approval": "PAUSED - waiting for your approval",
    "waiting_for_input": "PAUSED - waiting for your answer",
    "running": "INCOMPLETE",
}


def build_report(st: RunState, verification: VerificationResult | None) -> str:
    ev = st.events()
    calls = [e for e in ev if e["type"] == "tool_call"]
    results = {e["seq"]: e for e in ev if e["type"] == "tool_result"}
    t0 = ev[0]["ts"] if ev else 0
    t1 = ev[-1]["ts"] if ev else 0
    L: list[str] = []
    L.append(f"# Operator run report\n\n**Request:** {st.goal}\n\n**Requested by:** {st.user.get('name')} "
             f"({st.user.get('username')})\n\n**Outcome:** {HEADLINE.get(st.status, st.status)}\n")
    L.append(f"*{st.step} model turns, {len(calls)} tool calls, {t1 - t0:.1f}s, generated "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
    if st.final.get("summary"):
        L.append("## Summary from the operator\n\n" + st.final["summary"] + "\n")
    if st.pending:
        L.append("## Waiting on you\n\n" + json.dumps(st.pending, indent=2) + "\n\nResume with `resume` once ready.\n")

    L.append("## Independent verification\n")
    if verification is None:
        L.append("Not run (the operator did not reach a `done` finish).\n")
    else:
        L.append(f"Result: **{'PASSED' if verification.passed else 'FAILED'}**. Checks ran from a fresh session"
                 " (and the audit API with a separate credential), not from the operator's own memory.\n")
        L.append("| Check | Result | Observation | Evidence |\n|---|---|---|---|")
        for c in verification.checks:
            L.append(f"| {c.description} | {'PASS' if c.passed else 'FAIL'} | {c.detail} | {c.evidence_file or ''} |")
        for p in verification.problems:
            L.append(f"\n- Platform check failed: {p}")
        L.append("")

    L.append("## Changes made in company systems\n")
    if st.mutations:
        for m in st.mutations:
            L.append(f"- #{m.seq} {m.summary} -> **{m.outcome}**" + (" (re-checked)" if m.outcome == "ambiguous" and m.resolved else ""))
    else:
        L.append("None.")
    L.append("")

    L.append("## Human approvals\n")
    if st.approvals:
        for a in st.approvals:
            L.append(f"- #{a.seq} `{a.action}` - {a.decision} by {a.by}. Reason it was gated: {a.reason}")
    else:
        L.append("No approval was needed.")
    L.append("")

    notable = []
    for e in ev:
        if e["type"] == "retry":
            notable.append(f"Automatic retry #{e['attempt']} of a read ({e['action']})")
        if e["type"] == "tool_result" and e.get("error_kind") in ("unsafe_retry", "denied", "not_approved"):
            notable.append(f"Platform blocked action #{e['seq']} ({e['error_kind']}): {e['preview'][:140]}")
        if e["type"] == "tool_result" and e.get("outcome") == "ambiguous":
            notable.append(f"Action #{e['seq']} failed with an unknown outcome; the operator had to re-check state")
        if e["type"] == "tool_result" and e.get("outcome") == "rejected":
            notable.append(f"Action #{e['seq']} was rejected by the system (nothing changed); the operator adapted")
        if e["type"] == "tool_result" and e.get("session_expired"):
            notable.append(f"Action #{e['seq']} hit an expired session; the operator signed in again")
        if e["type"] == "tool_result" and e.get("error_kind") == "stale_action":
            notable.append(f"Action #{e['seq']} was paused earlier and no longer matched the page; the platform refused to run it")
    L.append("## Problems encountered and how they were handled\n")
    L += [f"- {n}" for n in notable] or ["None."]
    L.append("")

    L.append("## Timeline\n")
    for c in calls:
        r = results.get(c["seq"], {})
        flag = "problem" if r.get("problem") else "ok"
        L.append(f"{c['seq']:>3}. `{c['action']}` - {flag}" + (f" ({r['error_kind']})" if r.get("error_kind") else "")
                 + (f" HTTP {r['status']}" if r.get("status") else ""))
    L.append("")
    if st.plan:
        L.append("## Final plan state\n")
        L += [f"- [{p['status']}] {p['text']}" for p in st.plan]
    return "\n".join(L) + "\n"


def write_report(st: RunState, verification: VerificationResult | None) -> Path:
    path = st.run_dir / "report.md"
    path.write_text(build_report(st, verification), encoding="utf-8")
    (st.run_dir / "result.json").write_text(json.dumps({
        "status": st.status, "goal": st.goal, "final": st.final, "steps": st.step,
        "approvals": [a.__dict__ for a in st.approvals],
        "mutations": [m.__dict__ for m in st.mutations],
        "verification": verification.to_dict() if verification else None}, indent=2, default=str))
    return path
