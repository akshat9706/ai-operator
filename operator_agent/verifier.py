"""Independent verification.

The operator does not get to declare victory. When it calls finish(status="done") it must attach
*claims*: concrete checks against the real systems. The verifier then runs them itself, from a
**fresh browser session** (no cookies or memory from the run) and, for JSON checks, with a
**different credential** (the system's read-only audit token that the operator cannot use).

Beyond the claims the operator chose, the platform adds checks the operator cannot skip:
  * every resource the run changed must be covered by at least one claim (so a vague claim such as
    "page contains 'Expense'" cannot pass for a changed report);
  * no state-changing action may remain with an unknown outcome;
  * at least one claim must exist.
Each check stores the raw observation as evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .browser import Browser
from .context import Sandbox, Systems, resource_key
from .state import RunState


@dataclass
class CheckResult:
    description: str
    kind: str
    passed: bool
    detail: str
    evidence_file: str | None = None


@dataclass
class VerificationResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        lines = ["VERIFICATION FAILED - the task is NOT considered complete."]
        lines += [f"- {p}" for p in self.problems]
        lines += [f"- check '{c.description}': {c.detail}" for c in self.checks if not c.passed]
        lines.append("Investigate the real state, fix what is wrong (or tell the employee what is blocking), "
                     "then call finish again with corrected claims.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "problems": self.problems,
                "checks": [c.__dict__ for c in self.checks]}


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(dotted)
    return obj


def _same(actual: Any, expected: Any) -> bool:
    try:
        return float(actual) == float(expected) and not isinstance(actual, bool)
    except (TypeError, ValueError):
        return str(actual) == str(expected)


class Verifier:
    def __init__(self, systems: Systems, sandbox: Sandbox, state: RunState, evidence_dir: Path,
                 timeout: float = 10.0, retries: int = 2, sleep=time.sleep):
        self.systems, self.sandbox, self.state = systems, sandbox, state
        self.dir = Path(evidence_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.sleep = timeout, retries, sleep
        self._n = 0

    # ------------------------------------------------------------------ entry
    def verify(self, claims: Any) -> VerificationResult:
        res = VerificationResult(True)
        if not isinstance(claims, list) or not claims:
            res.problems.append("No claims were provided. Provide checks that confirm the outcome in the real system.")
        for m in self.state.unresolved_ambiguous():
            res.problems.append(f"Action #{m.seq} ({m.summary}) has an unknown outcome and was never re-checked.")
        claims = claims if isinstance(claims, list) else []
        covered: set[str] = set()
        for c in claims:
            cr = self._run_claim(c)
            res.checks.append(cr)
            if cr.passed:
                covered.add(self._claim_resource(c) or "")
        for m in self.state.mutations:
            if m.outcome in ("ok", "ambiguous") and m.resource not in covered:
                res.problems.append(f"Changed resource {m.resource} ({m.summary}) is not covered by any passing claim.")
        res.problems = list(dict.fromkeys(res.problems))
        res.passed = not res.problems and all(c.passed for c in res.checks) and bool(res.checks)
        return res

    # ----------------------------------------------------------------- claims
    def _claim_resource(self, c: dict) -> str | None:
        try:
            if c.get("type") == "file_contains":
                return f"fs:{self.sandbox.rel(self.sandbox.resolve(c['path']))}"
            system = self.systems.systems[c["system"]]
            path = c["path"]
            prefix = system.get("audit", {}).get("path_prefix", "")
            if c["type"] == "json_check" and prefix and path.startswith(prefix):
                path = path[len(prefix):]
            if path.endswith(".json"):
                path = path[:-5]
            return resource_key(self.systems, system["base_url"].rstrip("/") + path)
        except Exception:
            return None

    def _run_claim(self, c: Any) -> CheckResult:
        self._n += 1
        if not isinstance(c, dict) or "type" not in c:
            return CheckResult(str(c)[:80], "invalid", False, "Malformed claim: expected an object with a 'type'.")
        desc, kind = c.get("description") or c["type"], c["type"]
        try:
            if kind == "web_contains":
                return self._web(c, desc)
            if kind == "json_check":
                return self._json(c, desc)
            if kind == "file_contains":
                return self._file(c, desc)
            return CheckResult(desc, kind, False, f"Unknown claim type '{kind}'.")
        except KeyError as exc:
            return CheckResult(desc, kind, False, f"Claim is missing or has an invalid field: {exc}.")
        except Exception as exc:  # verification must never crash the run
            return CheckResult(desc, kind, False, f"Could not run check: {type(exc).__name__}: {exc}")

    def _evidence(self, kind: str, body: str) -> str:
        name = f"check_{self._n:02d}_{kind}.txt"
        (self.dir / name).write_text(body, encoding="utf-8")
        return str(Path(self.dir.name) / name)

    def _with_retries(self, fn):
        last = None
        for i in range(self.retries + 1):
            try:
                return fn()
            except (requests.RequestException, ConnectionError) as exc:
                last = exc
                self.sleep(0.3 * (i + 1))
        raise last  # type: ignore[misc]

    def _web(self, c: dict, desc: str) -> CheckResult:
        system = self.systems.systems[c["system"]]
        base = system["base_url"].rstrip("/")
        def run():
            b = Browser(self.timeout)
            lg = system["login"]
            cred = self.systems.credential(lg["credential"])
            page = b.goto(base + lg["path"])
            form = next(f for f in page.forms.values() if any(x.type == "password" for x in x_fields(f)))
            b.submit(form.id, {lg["username_field"]: cred["username"], lg["password_field"]: cred["password"]})
            return b.goto(base + c["path"])
        page = self._with_retries(run)
        text = page.text
        problems = []
        if page.status != 200 or page.is_login:
            problems.append(f"page returned HTTP {page.status}{' (sign-in page)' if page.is_login else ''}")
        low = text.lower()
        problems += [f"expected text not found: {s!r}" for s in c.get("contains", []) if str(s).lower() not in low]
        problems += [f"unexpected text present: {s!r}" for s in c.get("not_contains", []) if str(s).lower() in low]
        ev = self._evidence("web", f"GET {base + c['path']} (fresh session)\nHTTP {page.status}\n\n{text}")
        return CheckResult(desc, "web_contains", not problems,
                           "; ".join(problems) if problems else f"observed on {c['path']} in a fresh session", ev)

    def _json(self, c: dict, desc: str) -> CheckResult:
        system = self.systems.systems[c["system"]]
        audit = system.get("audit")
        if not audit:
            return CheckResult(desc, "json_check", False, f"System {c['system']} has no audit API.")
        if not c["path"].startswith(audit["path_prefix"]):
            return CheckResult(desc, "json_check", False,
                               f"json_check must target the audit API ({audit['path_prefix']}/...).")
        token = self.systems.credential(audit["credential"])["token"]
        url = system["base_url"].rstrip("/") + c["path"]
        resp = self._with_retries(lambda: requests.get(url, headers={audit["header"]: token}, timeout=self.timeout))
        problems = []
        data: Any = None
        if resp.status_code != 200:
            problems.append(f"audit API returned HTTP {resp.status_code}")
        else:
            data = resp.json()
            for dotted, expected in (c.get("expect") or {}).items():
                try:
                    actual = _dig(data, dotted)
                except (KeyError, IndexError, ValueError):
                    problems.append(f"{dotted}: not present in system of record")
                    continue
                if not _same(actual, expected):
                    problems.append(f"{dotted}: expected {expected!r} but the system of record has {actual!r}")
            if not c.get("expect"):
                problems.append("json_check needs a non-empty 'expect'.")
        ev = self._evidence("json", f"GET {url} (audit credential)\nHTTP {resp.status_code}\n\n" +
                            (json.dumps(data, indent=2) if data is not None else resp.text))
        return CheckResult(desc, "json_check", not problems,
                           "; ".join(problems) if problems else "system of record matches: " + json.dumps(c["expect"]), ev)

    def _file(self, c: dict, desc: str) -> CheckResult:
        p = self.sandbox.resolve(c["path"])
        if not p.is_file():
            return CheckResult(desc, "file_contains", False, f"file {c['path']} does not exist")
        text = p.read_text(encoding="utf-8")
        missing = [s for s in c.get("contains", []) if s not in text]
        ev = self._evidence("file", f"{self.sandbox.rel(p)}\n\n{text}")
        return CheckResult(desc, "file_contains", not missing,
                           f"missing text: {missing}" if missing else f"{self.sandbox.rel(p)} contains expected text", ev)


AUDIT_SYSTEM = """You are an independent auditor. You did not do the work. You are given an employee's request, \
the operator's own summary, and EVIDENCE the platform collected itself from the real systems in a fresh session. \
Decide whether the evidence demonstrates that the requested outcome was actually achieved and that nothing the \
request implies was missed (for example: right amounts per policy, right status, nothing wrongly included or \
omitted). Rely on the evidence, not on the operator's summary. Reply with ONLY a JSON object: \
{"sufficient": true|false, "gaps": ["...specific problem or missing proof..."]}."""


class LLMAuditor:
    """Optional second opinion on whether the evidence proves the goal (catches weak or irrelevant claims).
    Fails open on an unusable reply (the mechanical checks already passed) but records that it was inconclusive."""

    def __init__(self, backend):
        self.backend = backend

    def audit(self, goal: str, summary: str, vr: "VerificationResult", evidence_dir: Path, state: RunState) -> tuple[bool, list[str]]:
        ev = []
        for c in vr.checks:
            if c.evidence_file:
                f = evidence_dir.parent / c.evidence_file
                ev.append(f"--- {c.description} ({c.evidence_file}) ---\n" + (f.read_text(encoding="utf-8")[:3000] if f.exists() else ""))
        msg = f"REQUEST:\n{goal}\n\nOPERATOR SUMMARY:\n{summary}\n\nEVIDENCE:\n" + "\n".join(ev)
        try:
            turn = self.backend.generate(AUDIT_SYSTEM, [{"role": "user", "content": [{"type": "text", "text": msg}]}], [], state)
            raw = turn.text[turn.text.index("{"): turn.text.rindex("}") + 1]
            out = json.loads(raw)
            return bool(out.get("sufficient")), [str(g) for g in out.get("gaps", [])]
        except Exception as exc:
            state.log("auditor_inconclusive", error=f"{type(exc).__name__}: {exc}")
            return True, []


def x_fields(form):  # small helper so the generator expression above stays readable
    return form.fields
