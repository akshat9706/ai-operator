"""The human in the loop. Three implementations:

* CLIHuman      - asks on the terminal (interactive use)
* ScriptedHuman - answers from a predefined policy (demos and tests)
* NoHuman       - nobody is watching: every approval/question is *deferred*, the run pauses and
                  can be resumed later with the answer. Failing closed is the safe default.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ApprovalRequest:
    action: str
    reason: str
    summary: str
    context: str = ""


@dataclass
class ApprovalResponse:
    decision: str      # approve | deny | defer
    by: str = "unknown"
    note: str = ""


class Human:
    def approve(self, req: ApprovalRequest) -> ApprovalResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def answer(self, question: str, options: list[str] | None) -> str | None:  # pragma: no cover
        raise NotImplementedError


class NoHuman(Human):
    def approve(self, req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse("defer", by="nobody-available")

    def answer(self, question: str, options: list[str] | None) -> str | None:
        return None


@dataclass
class ScriptedHuman(Human):
    approve_all: bool = True
    deny_patterns: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    name: str = "scripted-human"
    seen_requests: list[ApprovalRequest] = field(default_factory=list)
    seen_questions: list[str] = field(default_factory=list)

    def approve(self, req: ApprovalRequest) -> ApprovalResponse:
        self.seen_requests.append(req)
        if any(p in req.action for p in self.deny_patterns) or not self.approve_all:
            return ApprovalResponse("deny", by=self.name, note="denied by script")
        return ApprovalResponse("approve", by=self.name)

    def answer(self, question: str, options: list[str] | None) -> str | None:
        self.seen_questions.append(question)
        return self.answers.pop(0) if self.answers else None


class CLIHuman(Human):
    def approve(self, req: ApprovalRequest) -> ApprovalResponse:
        print("\n" + "=" * 70)
        print("APPROVAL NEEDED")
        print(f"  Action : {req.summary}")
        print(f"  Why    : {req.reason}")
        if req.context:
            print("  Context:\n    " + req.context.replace("\n", "\n    "))
        ans = input("  Approve? [y]es / [n]o / [l]ater: ").strip().lower()
        if ans.startswith("y"):
            return ApprovalResponse("approve", by="cli-user")
        if ans.startswith("l"):
            return ApprovalResponse("defer", by="cli-user")
        return ApprovalResponse("deny", by="cli-user", note=input("  Reason (optional): "))

    def answer(self, question: str, options: list[str] | None) -> str | None:
        print("\n" + "=" * 70)
        print("THE OPERATOR NEEDS YOUR INPUT\n  " + question)
        if options:
            print("  Options: " + " / ".join(options))
        ans = input("  Your answer (blank = answer later): ").strip()
        return ans or None
