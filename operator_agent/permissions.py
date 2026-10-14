"""Permission gate: every tool call is reduced to an *action string* describing its real effect
(e.g. ``http:POST http://host/reports/3/submit`` or ``fs_write:out/summary.md``) and matched
against an ordered rule list. First match wins. The model's own description of what it is doing
never enters the decision, so it cannot talk its way past the gate.

The rules live in company config (permissions.json), not in code or in the prompt.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Decision:
    verdict: str   # allow | ask | deny
    reason: str
    rule: str


def render_placeholders(text: str, placeholders: dict[str, str]) -> str:
    for k, v in placeholders.items():
        text = text.replace("{" + k + "}", v)
    return text


class PermissionPolicy:
    def __init__(self, rules: list[dict], default: str = "ask"):
        self.rules = rules
        self.default = default

    @classmethod
    def load(cls, path: Path, placeholders: dict[str, str]) -> "PermissionPolicy":
        raw = render_placeholders(Path(path).read_text(encoding="utf-8"), placeholders)
        cfg = json.loads(raw)
        return cls(cfg["rules"], cfg.get("default", "ask"))

    def evaluate(self, action: str) -> Decision:
        for rule in self.rules:
            if fnmatch.fnmatchcase(action, rule["pattern"]):
                return Decision(rule["decision"], rule.get("reason", ""), rule["pattern"])
        return Decision(self.default, "No rule matched; defaulting to asking a human.", "<default>")
