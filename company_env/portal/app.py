"""Acme Expense Portal - a small but real web application used as the simulated
company system. It is intentionally *not* friendly to scripts: it has session
cookies, CSRF tokens, server-side policy validation, and injectable faults, so an
operator has to behave like a careful user rather than replay a fixed request.

Faults (all optional, set through create_app(faults=...) or POST /__faults):
  submit_503_before_commit : int  - next N submits fail with 503 and change nothing
  submit_error_after_commit: int  - next N submits COMMIT, then answer 503 (ambiguous!)
  session_expire_after     : int  - once, after N authenticated requests, drop the session
  latency_ms               : int  - artificial delay on every request
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time
from datetime import date, datetime, timezone
from html import escape

from flask import Flask, Response, abort, g, jsonify, redirect, request, session

CATEGORIES = ["Transport", "Lodging", "Meals", "Other"]
MEAL_CAP_PER_DAY = 1500
LODGING_CAP_PER_NIGHT = 8000
MAX_RECEIPT_AGE_DAYS = 30
COST_CENTERS = {"CC-1042", "CC-2207", "CC-3310"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, title TEXT NOT NULL,
  cost_center TEXT NOT NULL, purpose TEXT, status TEXT NOT NULL DEFAULT 'draft',
  created_at TEXT NOT NULL, submitted_at TEXT);
CREATE TABLE IF NOT EXISTS lines (
  id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER NOT NULL, date TEXT NOT NULL,
  category TEXT NOT NULL, merchant TEXT NOT NULL, amount INTEGER NOT NULL,
  receipt_ref TEXT NOT NULL, nights INTEGER NOT NULL DEFAULT 1, note TEXT DEFAULT '');
"""


def create_app(db_path: str, users: dict[str, str], audit_token: str,
               faults: dict | None = None, today: str = "2026-09-20") -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)
    app.config.update(DB_PATH=db_path, USERS=users, AUDIT_TOKEN=audit_token,
                      FAULTS=dict(faults or {}), FIRED=set(), TODAY=today,
                      LOCK=threading.RLock())

    boot = sqlite3.connect(db_path)
    boot.executescript(SCHEMA)
    boot.close()

    # ------------------------------------------------------------ helpers
    def db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def _close(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    def page(title: str, body: str, status: int = 200, errors: list[str] | None = None,
             notice: str | None = None) -> Response:
        flash = "".join(f'<div class="error" role="alert">{escape(e)}</div>' for e in (errors or []))
        if notice:
            flash += f'<div class="notice" role="status">{escape(notice)}</div>'
        nav = ""
        if session.get("user"):
            nav = ('<nav><a href="/reports">My reports</a> <a href="/reports/new">New report</a> '
                   f'<span>Signed in as {escape(session["user"])}</span></nav>')
        html = (f"<!doctype html><html><head><title>{escape(title)} - Acme Expenses</title></head>"
                f"<body>{nav}<h1>{escape(title)}</h1>{flash}{body}</body></html>")
        return Response(html, status=status, mimetype="text/html")

    def csrf_field() -> str:
        return f'<input type="hidden" name="csrf_token" value="{escape(session.get("csrf", ""))}">'

    def faults() -> dict:
        return app.config["FAULTS"]

    def consume(fault: str) -> bool:
        with app.config["LOCK"]:
            n = int(faults().get(fault, 0) or 0)
            if n > 0:
                faults()[fault] = n - 1
                return True
        return False

    @app.before_request
    def _latency():
        ms = int(faults().get("latency_ms", 0) or 0)
        if ms:
            time.sleep(ms / 1000)

    def login_redirect(expired: bool = False):
        return redirect("/login?expired=1" if expired else "/login")

    def require_login():
        """Returns a redirect Response if the caller must sign in, else None."""
        if not session.get("user"):
            return login_redirect()
        session["n"] = session.get("n", 0) + 1
        limit = int(faults().get("session_expire_after", 0) or 0)
        with app.config["LOCK"]:
            if limit and "session_expire" not in app.config["FIRED"] and session["n"] > limit:
                app.config["FIRED"].add("session_expire")
                session.clear()
                return login_redirect(expired=True)
        return None

    def check_csrf():
        if request.form.get("csrf_token") != session.get("csrf"):
            abort(403)

    def get_report(rid: int):
        row = db().execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        if row is None or row["owner"] != session.get("user"):
            abort(404)
        return row

    def report_lines(rid: int):
        return db().execute("SELECT * FROM lines WHERE report_id=? ORDER BY id", (rid,)).fetchall()

    def total(lines) -> int:
        return sum(l["amount"] for l in lines)

    # -------------------------------------------------------------- auth
    @app.get("/")
    def index():
        return redirect("/reports")

    @app.get("/login")
    def login_form():
        notice = None
        errors = []
        if request.args.get("expired"):
            errors.append("Your session has expired. Please sign in again.")
        body = ('<form method="post" action="/login">'
                '<label>Username <input name="username" type="text" required></label>'
                '<label>Password <input name="password" type="password" required></label>'
                '<button type="submit">Sign in</button></form>')
        return page("Sign in", body, errors=errors, notice=notice)

    @app.post("/login")
    def login():
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if app.config["USERS"].get(u) != p or not p:
            return page("Sign in", '<form method="post" action="/login">'
                        '<input name="username" type="text"><input name="password" type="password">'
                        '<button type="submit">Sign in</button></form>',
                        status=401, errors=["Invalid username or password."])
        session.clear()
        session.update(user=u, csrf=secrets.token_hex(12), n=0)
        return redirect("/reports")

    # ----------------------------------------------------------- reports
    @app.get("/reports")
    def list_reports():
        r = require_login()
        if r:
            return r
        rows = db().execute("SELECT * FROM reports WHERE owner=? ORDER BY id", (session["user"],)).fetchall()
        trs = "".join(
            f'<tr><td><a href="/reports/{x["id"]}">Report #{x["id"]}</a></td><td>{escape(x["title"])}</td>'
            f'<td>{x["status"]}</td></tr>' for x in rows)
        body = ("<table><tr><th>Report</th><th>Title</th><th>Status</th></tr>" + trs + "</table>"
                if rows else "<p>No reports yet.</p>")
        return page("My expense reports", body)

    def report_form(values: dict, errors: list[str] | None = None, status: int = 200):
        body = (f'<form method="post" action="/reports">{csrf_field()}'
                f'<label>Title <input name="title" type="text" required value="{escape(values.get("title", ""))}"></label>'
                f'<label>Cost center <input name="cost_center" type="text" required value="{escape(values.get("cost_center", ""))}"></label>'
                f'<label>Purpose <textarea name="purpose">{escape(values.get("purpose", ""))}</textarea></label>'
                '<button type="submit">Create draft report</button></form>')
        return page("New expense report", body, status=status, errors=errors)

    @app.get("/reports/new")
    def new_report_form():
        r = require_login()
        return r or report_form({})

    @app.post("/reports")
    def create_report():
        r = require_login()
        if r:
            return r
        check_csrf()
        v = {k: request.form.get(k, "").strip() for k in ("title", "cost_center", "purpose")}
        errs = []
        if not v["title"]:
            errs.append("Title is required.")
        if not v["cost_center"]:
            errs.append("Cost center is required.")
        elif v["cost_center"] not in COST_CENTERS:
            errs.append(f"Unknown cost center '{v['cost_center']}'. Expected a code like CC-1042.")
        if errs:
            return report_form(v, errs, 422)
        with app.config["LOCK"]:
            cur = db().execute(
                "INSERT INTO reports(owner,title,cost_center,purpose,created_at) VALUES(?,?,?,?,?)",
                (session["user"], v["title"], v["cost_center"], v["purpose"], datetime.now(timezone.utc).isoformat()))
            db().commit()
        return redirect(f"/reports/{cur.lastrowid}?created=1")

    @app.get("/reports/<int:rid>")
    def show_report(rid: int):
        r = require_login()
        if r:
            return r
        rep = get_report(rid)
        lines = report_lines(rid)
        rows = "".join(
            f'<tr><td>{l["date"]}</td><td>{l["category"]}</td><td>{escape(l["merchant"])}</td>'
            f'<td>{l["amount"]}</td><td>{escape(l["receipt_ref"])}</td><td>{l["nights"]}</td>'
            f'<td>{escape(l["note"] or "")}</td></tr>' for l in lines)
        table = ("<table><tr><th>Date</th><th>Category</th><th>Merchant</th><th>Amount</th>"
                 "<th>Receipt</th><th>Nights</th><th>Note</th></tr>" + rows + "</table>"
                 if lines else "<p>No expense lines yet.</p>")
        actions = ""
        if rep["status"] == "draft":
            actions = (f'<p><a href="/reports/{rid}/lines/new">Add expense line</a></p>'
                       f'<form method="post" action="/reports/{rid}/submit">{csrf_field()}'
                       '<button type="submit">Submit report for approval</button></form>')
        body = (f'<p>Status: <strong>{rep["status"]}</strong></p><p>Owner: {escape(rep["owner"])}</p>'
                f'<p>Title: {escape(rep["title"])}</p><p>Cost center: {rep["cost_center"]}</p>'
                f'<p>Lines: {len(lines)}</p><p>Total: {total(lines)}</p>{table}{actions}')
        notice = "Report created." if request.args.get("created") else (
            "Expense line added." if request.args.get("added") else (
                "Report submitted." if request.args.get("submitted") else None))
        return page(f"Expense report #{rid}", body, notice=notice)

    def line_form(rid: int, v: dict, errors=None, status=200):
        opts = "".join(f'<option value="{c}"{" selected" if v.get("category") == c else ""}>{c}</option>'
                       for c in CATEGORIES)
        body = (f'<form method="post" action="/reports/{rid}/lines">{csrf_field()}'
                f'<label>Date (YYYY-MM-DD) <input name="date" type="text" required value="{escape(v.get("date", ""))}"></label>'
                f'<label>Category <select name="category">{opts}</select></label>'
                f'<label>Merchant <input name="merchant" type="text" required value="{escape(v.get("merchant", ""))}"></label>'
                f'<label>Amount (INR, whole rupees) <input name="amount" type="text" required value="{escape(v.get("amount", ""))}"></label>'
                f'<label>Receipt reference <input name="receipt_ref" type="text" required value="{escape(v.get("receipt_ref", ""))}"></label>'
                f'<label>Nights (lodging only) <input name="nights" type="text" value="{escape(v.get("nights", "1"))}"></label>'
                f'<label>Note <input name="note" type="text" value="{escape(v.get("note", ""))}"></label>'
                '<button type="submit">Add line</button></form>')
        return page(f"Add expense line to report #{rid}", body, status=status, errors=errors)

    @app.get("/reports/<int:rid>/lines/new")
    def new_line_form(rid: int):
        r = require_login()
        if r:
            return r
        rep = get_report(rid)
        if rep["status"] != "draft":
            return page("Not editable", "<p>Only draft reports can be edited.</p>", 409,
                        ["Report is already " + rep["status"] + "."])
        return line_form(rid, {})

    @app.post("/reports/<int:rid>/lines")
    def add_line(rid: int):
        r = require_login()
        if r:
            return r
        check_csrf()
        rep = get_report(rid)
        if rep["status"] != "draft":
            return page("Not editable", "<p>Only draft reports can be edited.</p>", 409,
                        ["Report is already " + rep["status"] + "."])
        v = {k: request.form.get(k, "").strip() for k in
             ("date", "category", "merchant", "amount", "receipt_ref", "nights", "note")}
        v["nights"] = v["nights"] or "1"
        errs: list[str] = []
        d = None
        try:
            d = date.fromisoformat(v["date"])
        except ValueError:
            errs.append("Date must be in YYYY-MM-DD format.")
        today = date.fromisoformat(app.config["TODAY"])
        if d and d > today:
            errs.append("Expense date cannot be in the future.")
        if d and (today - d).days > MAX_RECEIPT_AGE_DAYS:
            errs.append(f"Receipts older than {MAX_RECEIPT_AGE_DAYS} days cannot be claimed.")
        if v["category"] not in CATEGORIES:
            errs.append(f"Category must be one of: {', '.join(CATEGORIES)}.")
        if not v["merchant"]:
            errs.append("Merchant is required.")
        if not v["receipt_ref"]:
            errs.append("Receipt reference is required.")
        amount = nights = 0
        try:
            amount = int(v["amount"])
            if amount <= 0:
                raise ValueError
        except ValueError:
            errs.append("Amount must be a positive whole number of rupees.")
        try:
            nights = int(v["nights"])
            if nights < 1:
                raise ValueError
        except ValueError:
            errs.append("Nights must be a whole number >= 1.")
        if not errs:
            existing = report_lines(rid)
            if any(l["receipt_ref"] == v["receipt_ref"] for l in existing):
                errs.append(f"Receipt '{v['receipt_ref']}' is already on this report (duplicate).")
            if v["category"] == "Meals":
                day_total = sum(l["amount"] for l in existing
                                if l["category"] == "Meals" and l["date"] == v["date"]) + amount
                if day_total > MEAL_CAP_PER_DAY:
                    errs.append(f"Meals on {v['date']} would total {day_total}, exceeding the daily "
                                f"cap of {MEAL_CAP_PER_DAY} INR.")
            if v["category"] == "Lodging" and amount / nights > LODGING_CAP_PER_NIGHT:
                errs.append(f"Lodging of {amount} over {nights} night(s) exceeds the cap of "
                            f"{LODGING_CAP_PER_NIGHT} INR per night.")
        if errs:
            return line_form(rid, v, errs, 422)
        with app.config["LOCK"]:
            db().execute("INSERT INTO lines(report_id,date,category,merchant,amount,receipt_ref,nights,note)"
                         " VALUES(?,?,?,?,?,?,?,?)",
                         (rid, v["date"], v["category"], v["merchant"], amount, v["receipt_ref"], nights, v["note"]))
            db().commit()
        return redirect(f"/reports/{rid}?added=1")

    @app.post("/reports/<int:rid>/lines/<int:lid>/delete")
    def delete_line(rid: int, lid: int):
        r = require_login()
        if r:
            return r
        check_csrf()
        rep = get_report(rid)
        if rep["status"] != "draft":
            abort(409)
        with app.config["LOCK"]:
            db().execute("DELETE FROM lines WHERE id=? AND report_id=?", (lid, rid))
            db().commit()
        return redirect(f"/reports/{rid}")

    @app.post("/reports/<int:rid>/submit")
    def submit_report(rid: int):
        r = require_login()
        if r:
            return r
        check_csrf()
        rep = get_report(rid)
        if rep["status"] != "draft":
            return page("Cannot submit", "<p>This report was already submitted.</p>", 409,
                        [f"Report #{rid} is already {rep['status']}."])
        if not report_lines(rid):
            return page("Cannot submit", "<p>Add at least one line first.</p>", 422,
                        ["A report needs at least one expense line."])
        if consume("submit_503_before_commit"):
            return page("Service unavailable", "<p>Try again shortly.</p>", 503, ["503 Service Unavailable"])
        with app.config["LOCK"]:
            db().execute("UPDATE reports SET status='submitted', submitted_at=? WHERE id=?",
                         (datetime.now(timezone.utc).isoformat(), rid))
            db().commit()
        if consume("submit_error_after_commit"):
            return page("Service unavailable", "<p>Gateway timeout.</p>", 503, ["503 Service Unavailable"])
        return redirect(f"/reports/{rid}?submitted=1")

    # ------------------------------------------------- audit / ops (not for agents)
    def audit_ok() -> bool:
        return request.headers.get("X-Audit-Token") == app.config["AUDIT_TOKEN"]

    @app.get("/audit/reports/<int:rid>.json")
    def audit_report(rid: int):
        if not audit_ok():
            abort(401)
        rep = db().execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        if rep is None:
            abort(404)
        lines = [dict(l) for l in report_lines(rid)]
        return jsonify(id=rid, owner=rep["owner"], title=rep["title"], cost_center=rep["cost_center"],
                       status=rep["status"], line_count=len(lines), total=total(lines), lines=lines)

    @app.get("/audit/reports.json")
    def audit_reports():
        if not audit_ok():
            abort(401)
        rows = db().execute("SELECT id,owner,title,status FROM reports ORDER BY id").fetchall()
        return jsonify(reports=[dict(r) for r in rows])

    @app.post("/__faults")
    def set_faults():
        if not audit_ok():
            abort(401)
        with app.config["LOCK"]:
            faults().update(request.get_json(force=True) or {})
            app.config["FIRED"].clear()
        return jsonify(faults())

    @app.errorhandler(403)
    def _403(_e):
        return page("Forbidden", "<p>Invalid or missing form token.</p>", 403, ["403 Forbidden (CSRF)"])

    @app.errorhandler(404)
    def _404(_e):
        return page("Not found", "<p>No such page.</p>", 404, ["404 Not Found"])

    @app.errorhandler(401)
    def _401(_e):
        return Response(json.dumps({"error": "unauthorized"}), 401, mimetype="application/json")

    return app
