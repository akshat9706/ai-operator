"""OFFLINE STAND-IN POLICIES - test doubles for the language model.

These exist so the *platform* (permissions, approvals, retry safety, verification, pause/resume,
reporting) can be exercised end to end in CI and in a sandbox with no API key. They are NOT the
product and are not intelligent: ExpenseStandIn is a hand-written procedure that reads the same tool
outputs a model would and reacts to them. The real autonomy comes from AnthropicBackend.

Both policies are written as generators: `result = yield ("tool_name", {args})`.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Generator, Iterator

from .llm import AssistantTurn, ToolCall
from .state import RunState, ToolResult


def call(name: str, **args: Any) -> tuple[str, dict]:
    return name, args


class GeneratorBackend:
    """Adapts a generator-based policy to the LLMBackend interface (one tool call per turn)."""
    name = "offline-standin"

    def __init__(self, factory: Callable[[], Generator]):
        self.factory, self.gen, self.n = factory, None, 0

    def generate(self, system: str, messages: list[dict], tools: list[dict], state: RunState) -> AssistantTurn:
        try:
            if self.gen is None:
                self.gen = self.factory()
                name, args = next(self.gen)
            else:
                last = state.last_results[0] if state.last_results else None
                name, args = self.gen.send(last)
        except StopIteration:
            return AssistantTurn("(policy ended without finishing)", [], [{"type": "text", "text": "done"}], "end_turn")
        self.n += 1
        cid = f"toolu_offline_{self.n:04d}"
        return AssistantTurn("", [ToolCall(cid, name, args)],
                             [{"type": "tool_use", "id": cid, "name": name, "input": args}], "tool_use")


class Blocked(Exception):
    pass


# ------------------------------------------------------------------ helpers (shared)

def _login() -> Iterator:
    r = yield call("browser_login", system="expense_portal")
    if not r.ok:
        raise Blocked(f"Could not sign in: {r.content}")


def _open(url: str) -> Iterator:
    """Open a page robustly: sign in again if the session expired."""
    for _ in range(4):
        r = yield call("browser_goto", url=url)
        if r.ok and r.data.get("session_expired"):
            yield from _login()
            continue
        if r.ok and r.data.get("status", 200) < 400:
            return r
        if not r.ok and r.error_kind in ("denied", "not_approved"):
            raise Blocked(r.content)
    raise Blocked(f"Could not open {url}")


def _form(r: ToolResult, contains: str) -> dict:
    for f in r.data["forms"]:
        if contains in f["action"]:
            return f
    raise Blocked(f"Expected a form for '{contains}' on {r.data.get('url')}")


def _submit(page_url: str, action_part: str, values: dict, verify: Callable[[], Iterator],
            attempts: int = 4) -> Iterator:
    """Submit a form and handle what can go wrong: expired session, stale page, server errors with an
    unknown outcome (verify before repeating), validation rejections (returned to the caller)."""
    for _ in range(attempts):
        r = yield from _open(page_url)
        form = _form(r, action_part)
        r = yield call("browser_submit_form", form_id=form["id"], values=values)
        status = r.data.get("status", 0)
        if r.error_kind == "invalid":
            continue
        if r.error_kind in ("denied", "not_approved"):
            return ("declined", r)
        if r.data.get("session_expired"):
            yield from _login()
            continue
        if r.error_kind in ("transient", "unsafe_retry") or status >= 500:
            if (yield from verify()):
                return ("ok", r)
            continue
        if status >= 400:
            return ("rejected", r)
        return ("ok", r)
    raise Blocked("Gave up after repeated failures.")


# --------------------------------------------------------------------- expense task

def _parse_receipt(name: str, text: str) -> dict:
    g = lambda pat: (re.search(pat, text, re.M) or [None, ""])[1].strip()
    items = [(d.strip(), int(a)) for d, a in re.findall(r"^\s*-\s*(.+?)\s+(\d+)\s*$", text, re.M)]
    return {"ref": name.rsplit(".", 1)[0], "merchant": g(r"^Merchant:\s*(.+)"), "date": g(r"^Date:\s*(.+)"),
            "bill": g(r"^Bill no:\s*(.+)"), "desc": g(r"^Description:\s*(.+)"), "items": items,
            "total": int(g(r"^TOTAL:\s*(\d+)") or 0), "covers": "Covers:" in text,
            "folio": "HOTEL FOLIO" in text, "text": text}


def expense_standin(request_goal: str, username: str = "priya.sharma") -> Callable[[], Generator]:
    def procedure() -> Generator:
        try:
            result = yield from _expense_procedure(request_goal, username)
        except Blocked as exc:
            yield call("finish", status="blocked", summary=f"I could not complete this: {exc}")
            return
        if result is not None:
            yield result
    return procedure


def _expense_procedure(goal: str, username: str) -> Iterator:
    yield call("update_plan", steps=[
        {"text": "Find the procedure and policy", "status": "in_progress"},
        {"text": "Work out which trip, read the documents and decide what is claimable", "status": "pending"},
        {"text": "Create the draft report and add lines in the Expense Portal", "status": "pending"},
        {"text": "Check the draft, get approval, submit", "status": "pending"},
        {"text": "Confirm the outcome and write a summary", "status": "pending"}])
    # ---- discover company context
    r = yield call("kb_search", query="expense reimbursement procedure")
    proc = (yield call("fs_read", path=r.data["hits"][0]["path"])).content
    r = yield call("kb_search", query="expense policy caps not reimbursable duplicates cost center")
    pol_path = next(h["path"] for h in r.data["hits"] if "policy" in h["path"])
    policy = (yield call("fs_read", path=pol_path)).content
    meal_cap = int(re.search(r"at most ([\d,]+) INR per day", policy).group(1).replace(",", ""))
    lodging_cap = int(re.search(r"at most ([\d,]+) INR per night", policy).group(1).replace(",", ""))
    title_fmt = re.search(r"Title format: `(.+?)`", proc).group(1)

    # ---- which trip? ask only if the request does not settle it
    r = yield call("fs_list", path="inbox")
    dirs = r.data["dirs"]
    hits = [d for d in dirs if d.split("_")[0].lower() in goal.lower()]
    if len(hits) == 1 or (not hits and len(dirs) == 1):
        folder = (hits or dirs)[0]
    else:
        r = yield call("ask_human", question="Which trip do you want me to submit expenses for?",
                       options=[d.split("_")[0].title() for d in dirs])
        if not r.ok:
            raise Blocked("I need to know which trip to submit.")
        folder = next((d for d in dirs if d.split("_")[0].lower() in r.data["answer"].lower()), None)
        if folder is None:
            raise Blocked(f"I could not match '{r.data['answer']}' to a trip folder in your inbox.")
    city = folder.split("_")[0].title()
    r = yield call("fs_list", path=f"inbox/{folder}")
    files = [e for e in r.data["entries"] if e.endswith(".txt")]
    docs = {}
    for f in files:
        docs[f] = (yield call("fs_read", path=f"inbox/{folder}/{f}")).content

    # ---- decide what is claimable (policy applied to the documents)
    req_name = next((f for f in files if "travel_request" in f), None)
    req = docs.get(req_name, "")
    receipts = [_parse_receipt(f, docs[f]) for f in files if f != req_name]
    cc = (re.search(r"Cost center:\s*(CC-\d+)", req) or [None, None])[1]
    if cc is None:   # policy: no travel request -> the employee's default cost center
        emp = (yield call("fs_read", path="directory/employees.json")).content
        cc = re.search(rf'"username": "{username}".*?"default_cost_center": "(CC-\d+)"', emp, re.S).group(1)
    purpose = (re.search(r"^Purpose:\s*(.+)", req, re.M) or [None, "Business travel"])[1]
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", req) or sorted(rc["date"] for rc in receipts)
    mon = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul",
           "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}[dates[0][5:7]] + " " + dates[0][:4]
    title = title_fmt.replace("<City>", city).replace("<Mon YYYY>", mon)

    seen, lines, left_out = set(), [], []
    for rc in receipts:
        key = (rc["merchant"], rc["bill"])
        if key in seen:
            left_out.append(f"{rc['ref']}: duplicate of an already-claimed bill ({rc['bill']})")
            continue
        seen.add(key)
        if re.search(r"gift|personal", rc["desc"], re.I):
            left_out.append(f"{rc['ref']}: personal purchase/gift (not reimbursable)")
            continue
        if re.search(r"\bfine\b|penalt|challan", rc["desc"], re.I):
            left_out.append(f"{rc['ref']}: fine/penalty (not reimbursable)")
            continue
        ok_items = [(d, a) for d, a in rc["items"] if not re.search(r"alcohol|minibar|beer|wine|laundry", d, re.I)]
        skipped = [(d, a) for d, a in rc["items"] if (d, a) not in ok_items]
        amount = sum(a for _, a in ok_items)
        nights = 1
        if rc["folio"]:
            cat, nights = "Lodging", int(re.search(r"(\d+) nights", rc["text"]).group(1))
            if amount / nights > lodging_cap:
                left_out.append(f"{rc['ref']}: {amount - lodging_cap * nights} over the lodging cap of {lodging_cap} per night")
                amount = lodging_cap * nights
        elif rc["covers"]:
            cat = "Meals"
        elif re.search(r"registration|conference|summit", rc["desc"], re.I):
            cat = "Other"
        else:
            cat = "Transport"
        if skipped:
            left_out.append(f"{rc['ref']}: {', '.join(f'{d} ({a})' for d, a in skipped)} not reimbursable")
        lines.append({"date": rc["date"], "category": cat, "merchant": rc["merchant"], "amount": amount,
                      "receipt_ref": rc["ref"], "nights": nights,
                      "note": ("excluded: " + ", ".join(d for d, _ in skipped)) if skipped else ""})
    day_meals: dict[str, int] = {}
    for ln in lines:
        if ln["category"] == "Meals":
            used = day_meals.get(ln["date"], 0)
            allowed = max(0, meal_cap - used)
            if ln["amount"] > allowed:
                left_out.append(f"{ln['receipt_ref']}: {ln['amount'] - allowed} over the daily meal cap of {meal_cap}")
                ln["amount"] = allowed
            day_meals[ln["date"]] = used + ln["amount"]
    total = sum(ln["amount"] for ln in lines)
    yield call("remember", key="decision", value=f"{len(lines)} lines, total {total}, cost center {cc}, title {title}")

    # ---- act in the portal
    r = yield call("list_systems")
    base = r.data["systems"][0]["base_url"].rstrip("/")
    yield from _login()
    r = yield from _open(f"{base}/reports")
    row = re.search(rf"Report #(\d+) \[L\d+\] \| {re.escape(title)} \| (\w+)", r.content)
    if row:
        raise Blocked(f"A report titled '{title}' already exists (#{row.group(1)}, {row.group(2)}); I did not want to "
                      "create a duplicate. Tell me whether to reuse it.")

    def find_report() -> Iterator:
        rr = yield from _open(f"{base}/reports")
        m = re.search(rf"Report #(\d+) \[L\d+\] \| {re.escape(title)} \|", rr.content)
        return int(m.group(1)) if m else None

    def created() -> Iterator:
        return (yield from find_report()) is not None

    out, r = yield from _submit(f"{base}/reports/new", "/reports", {"title": title, "cost_center": cc, "purpose": purpose}, created)
    if out == "rejected":
        yield call("ask_human", question="The portal rejected the new report: " + "; ".join(r.data["alerts"]))
        raise Blocked("Portal rejected the draft report: " + "; ".join(r.data["alerts"]))
    rid = yield from find_report()
    if rid is None:
        raise Blocked("The draft report could not be found after creating it.")
    yield call("remember", key="report_id", value=str(rid))

    added = []
    for ln in lines:
        def has_line(ref=ln["receipt_ref"]) -> Iterator:
            rr = yield from _open(f"{base}/reports/{rid}")
            return ref in rr.content
        out, r = yield from _submit(f"{base}/reports/{rid}/lines/new", f"/reports/{rid}/lines",
                                    {**{k: str(v) for k, v in ln.items()}}, has_line)
        if out == "rejected":
            why = "; ".join(r.data["alerts"])
            if "older than" in why:     # adapt: the portal enforces a rule the documents did not reveal
                left_out.append(f"{ln['receipt_ref']}: {why} (rejected by the portal, so not claimed)")
                continue
            yield call("ask_human", question=f"The portal rejected the line for {ln['receipt_ref']}: {why}")
            raise Blocked(f"Portal rejected line {ln['receipt_ref']}: {why}")
        added.append(ln)
    if len(added) != len(lines):
        lines = added
        total = sum(ln["amount"] for ln in lines)
        yield call("remember", key="decision_revised", value=f"{len(lines)} lines, total {total} after portal rejections")

    r = yield from _open(f"{base}/reports/{rid}")
    if f"Total: {total}" not in r.content or f"Lines: {len(lines)}" not in r.content:
        raise Blocked("The draft in the portal does not match what I intended to claim; stopping before submission.")

    def submitted() -> Iterator:
        rr = yield from _open(f"{base}/reports/{rid}")
        return "Status: submitted" in rr.content

    out, r = yield from _submit(f"{base}/reports/{rid}", f"/reports/{rid}/submit", {}, submitted)
    if out == "declined":
        yield call("finish", status="blocked", summary=(
            f"Draft report #{rid} is ready with {len(lines)} lines totalling {total} INR but NOT submitted: the "
            "submission was not approved. Nothing was sent for reimbursement."))
        return
    if out == "rejected":
        raise Blocked("The portal refused the submission: " + "; ".join(r.data["alerts"]))
    r = yield from _open(f"{base}/reports/{rid}")
    if "Status: submitted" not in r.content:
        raise Blocked("Submission did not take effect.")

    summary = (f"Expense report #{rid} '{title}' submitted for {username}: {len(lines)} lines, total {total} INR, "
               f"cost center {cc}.\n\nClaimed:\n" + "\n".join(
                   f"- {l['date']} {l['category']} {l['merchant']}: {l['amount']} ({l['receipt_ref']})" for l in lines) +
               "\n\nLeft out:\n" + "\n".join(f"- {x}" for x in left_out))
    yield call("fs_write", path=f"out/expense_summary_{rid}.md", content=summary)
    yield call("update_plan", steps=[{"text": t, "status": "done"} for t in (
        "Find the procedure and policy", "Work out which trip, read the documents and decide what is claimable",
        "Create the draft report and add lines in the Expense Portal", "Check the draft, get approval, submit",
        "Confirm the outcome and write a summary")])
    return call("finish", status="done", summary=summary, claims=[
        {"description": "System of record shows the report submitted with the right totals", "type": "json_check",
         "system": "expense_portal", "path": f"/audit/reports/{rid}.json",
         "expect": {"status": "submitted", "cost_center": cc, "line_count": len(lines), "total": total, "owner": username}},
        {"description": "Portal page (fresh session) shows submitted report", "type": "web_contains",
         "system": "expense_portal", "path": f"/reports/{rid}", "contains": ["Status: submitted", f"Total: {total}"]},
        {"description": "Summary file saved for the employee", "type": "file_contains",
         "path": f"out/expense_summary_{rid}.md", "contains": [str(total), "Left out"]}])
