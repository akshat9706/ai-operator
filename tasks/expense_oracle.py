"""External grader for the expense tasks. NOT part of the operator: it reads the portal's database
directly and compares it with hand-written ground truth (tasks/<task>/expected.json)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


@dataclass
class Grade:
    passed: bool
    problems: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def load_expected(task: str) -> dict:
    return json.loads((HERE / task / "expected.json").read_text(encoding="utf-8"))


def grade(db_path: str, task: str = "expense_mumbai", expect_status: str = "submitted") -> Grade:
    con = sqlite3.connect(db_path)
    try:
        return _grade(con, load_expected(task), expect_status)
    finally:
        con.close()


def _grade(con: sqlite3.Connection, exp: dict, expect_status: str) -> Grade:
    con.row_factory = sqlite3.Row
    reports = con.execute("SELECT * FROM reports WHERE owner=?", (exp["owner"],)).fetchall()
    problems: list[str] = []
    if len(reports) != 1:
        problems.append(f"expected exactly 1 report for {exp['owner']}, found {len(reports)}")
        return Grade(False, problems, {"reports": len(reports)})
    rep = reports[0]
    lines = [dict(r) for r in con.execute("SELECT * FROM lines WHERE report_id=? ORDER BY id", (rep["id"],))]
    if rep["title"] != exp["title"]:
        problems.append(f"title {rep['title']!r} != {exp['title']!r}")
    if rep["cost_center"] != exp["cost_center"]:
        problems.append(f"cost center {rep['cost_center']} != {exp['cost_center']}")
    if rep["status"] != expect_status:
        problems.append(f"status {rep['status']} != {expect_status}")
    got = {l["receipt_ref"]: (l["category"], l["amount"], l["nights"]) for l in lines}
    want = {l["receipt_ref"]: (l["category"], l["amount"], l["nights"]) for l in exp["lines"]}
    if len(got) != len(lines):
        problems.append("duplicate receipt refs on the report")
    for ref, w in want.items():
        if ref not in got:
            problems.append(f"missing claimable line {ref}")
        elif got[ref] != w:
            problems.append(f"line {ref}: got {got[ref]}, expected {w}")
    for ref in got:
        if ref not in want:
            problems.append(f"line {ref} should NOT be claimed ({exp['why'].get(ref, 'not claimable')})")
    total = sum(l["amount"] for l in lines)
    if total != exp["total"]:
        problems.append(f"total {total} != {exp['total']}")
    return Grade(not problems, problems, {"report_id": rep["id"], "status": rep["status"], "total": total,
                                          "lines": len(lines)})
