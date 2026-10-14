"""Command line interface.

  python -m operator_agent serve                      # the simulated Acme Expense Portal, to look at in a browser
  python -m operator_agent run --scenario pune        # a REAL run: Claude does the work (needs ANTHROPIC_API_KEY)
  python -m operator_agent run "Submit my expenses for the Mumbai trip."
  python -m operator_agent resume <run_id>            # continue a run that paused for approval / an answer
  python -m operator_agent grade <run_id> --task expense_pune   # external ground-truth check of a finished run
  python -m operator_agent demo [--scenario NAME|all] # OFFLINE test-double policy: exercises the platform, no model

Exit codes: 0 completed and verified, 2 blocked/failed/stalled, 3 paused waiting for a human.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from company_env.launch import ENV_ROOT, copy_company, start_portal
from tasks.expense_oracle import grade as grade_task
from tasks.scenarios import SCENARIOS

from .agent import Agent
from .human import CLIHuman, Human, NoHuman, ScriptedHuman
from .gemini import (DEFAULT_MODEL as GEMINI_DEFAULT_MODEL, GeminiBackend, list_models, probe_model,
                     usable_text_models)
from .llm import AnthropicBackend, LLMError
from .openai_compat import PRESETS, OpenAICompatBackend, list_openai_models, resolve_key
from .runtime import build_runtime
from .state import RunState

DEFAULT_TODAY = "2026-09-20"       # the simulated company's clock; fixture receipts are dated around it
EXIT = {"completed": 0, "waiting_for_approval": 3, "waiting_for_input": 3}
USE_COLOR = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


# ------------------------------------------------------------------------------------------ live progress
def progress_printer(verbose: bool = False):
    def show(e: dict) -> None:
        t = e["type"]
        if t == "assistant" and e.get("text"):
            print(c("36", f"\n[{e['step']}] " + e["text"].strip().replace("\n", " ")[:300]))
        elif t == "tool_call":
            print(c("2", f"    -> {e['tool']}  {e['action'][:110]}"))
        elif t == "tool_result":
            tag = c("32", "ok ") if not e.get("problem") else c("33", "!! ")
            extra = f" [{e['error_kind']}]" if e.get("error_kind") else ""
            extra += f" HTTP {e['status']}" if e.get("status") and e["status"] >= 400 else ""
            print(c("2", f"    <- ") + tag + e.get("preview", "").replace("\n", " ")[:110 if not verbose else 240] + extra)
        elif t == "approval":
            print(c("35", f"    ** approval: {e['action'][:90]} -> {e['decision']} (by {e['by']})"))
        elif t == "retry":
            print(c("33", f"    .. automatic retry #{e['attempt']} of a read"))
        elif t == "llm_retry":
            print(c("33", f"    .. model API busy, retrying in {e['delay']:.0f}s ({e['error'][:60]})"))
        elif t == "paused":
            print(c("35", f"\n== PAUSED ({e['status']}): " + json.dumps(e["pending"])[:300]))
        elif t == "verification":
            verdict = c("32", "PASSED") if e["passed"] else c("31", "FAILED")
            print(c("1", f"\n== independent verification attempt {e['attempt']}: ") + verdict)
            for ch in e["checks"]:
                print(f"     {'PASS' if ch['passed'] else 'FAIL'}  {ch['description']}: {ch['detail'][:100]}")
            for p in e["problems"]:
                print(c("31", f"     problem: {p[:140]}"))
    return show


# ------------------------------------------------------------------------------------------------ helpers
def make_human(mode: str) -> Human:
    if mode == "auto":
        return ScriptedHuman(approve_all=True, name="auto-approve")
    if mode == "defer" or (mode == "auto-detect" and not sys.stdin.isatty()):
        return NoHuman()
    return CLIHuman()


PROVIDERS = ["auto", "anthropic", "gemini", *PRESETS]


def resolve_provider(name: str | None) -> str:
    """'auto' looks at which API key you have set: Anthropic first, then Gemini, then the open-model hosts."""
    if name and name != "auto":
        return name
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    for p in ("cerebras", "groq", "openrouter", "nvidia"):
        if resolve_key(PRESETS[p], p):
            return p
    return "anthropic"


def make_backend(args):
    provider = resolve_provider(getattr(args, "provider", None))
    args.provider = provider
    interval = getattr(args, "min_interval", None)
    extra = {"max_retries": args.max_retries} if getattr(args, "max_retries", None) is not None else {}
    if provider == "gemini":
        args.model = args.model or GEMINI_DEFAULT_MODEL
        base = args.base_url or os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com"
        return GeminiBackend(model=args.model, base_url=base, max_tokens=args.max_tokens,
                             min_interval=4.0 if interval is None else interval, **extra)
    if provider in PRESETS:
        backend = OpenAICompatBackend(provider, model=args.model, base_url=args.base_url, max_tokens=args.max_tokens,
                                      min_interval=2.0 if interval is None else interval, **extra)
        args.model, args.base_url = backend.model, backend.base_url
        return backend
    args.model = args.model or "claude-sonnet-5-5"
    base = args.base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    return AnthropicBackend(model=args.model, base_url=base, max_tokens=args.max_tokens)


def finish_run(rt, run_dir: Path, scenario_task: str | None, db_path: str, assumed_answers=()) -> int:
    st = rt.state
    print("\n" + "=" * 72)
    print(f"STATUS: {st.status.upper()}   ({st.step} model turns)   report: {run_dir / 'report.md'}")
    if st.final.get("summary"):
        print("\n" + st.final["summary"])
    if scenario_task and st.status in ("completed", "blocked", "failed"):
        g = grade_task(db_path, scenario_task, "submitted" if st.status == "completed" else "draft")
        print("\n" + c("1", "Ground-truth check") + " (external oracle, never visible to the operator): " +
              (c("32", "MATCHES expected outcome") if g.passed else c("31", "DOES NOT MATCH: " + "; ".join(g.problems))))
        if assumed_answers:
            print(f"  (the expected outcome assumes you answered the question with: {', '.join(assumed_answers)})")
    if st.status in ("waiting_for_approval", "waiting_for_input"):
        print(f"\nPaused. Continue with:  python -m operator_agent resume {run_dir.name} --runs-dir {run_dir.parent}")
    return EXIT.get(st.status, 2)


# -------------------------------------------------------------------------------------------- subcommands
def cmd_serve(args) -> int:
    db = args.db or str(Path("runs") / "portal_serve.db")
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    faults = json.loads(args.faults) if args.faults else {}
    p = start_portal(faults=faults, port=args.port, db_path=db, today=args.today)
    print(f"Acme Expense Portal on {p.url}   (log in as priya.sharma / pw-priya-demo)   db: {db}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        p.stop()
    return 0


def cmd_run(args) -> int:
    sc = SCENARIOS.get(args.scenario) if args.scenario else None
    goal = args.goal or (sc.goal if sc else None)
    if not goal:
        print("Give a request, e.g.  run \"Submit my expenses for the Pune trip.\"  or  run --scenario pune")
        return 2
    try:
        backend = make_backend(args)
    except LLMError as exc:
        print(f"Cannot start: {exc}\nPut the key for your provider in the .env file (see .env.example), "
              "or use `demo` for the offline test-double run.")
        return 2
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    runs = Path(args.runs_dir)
    run_dir = runs / run_id
    if run_dir.exists():
        print(f"{run_dir} already exists. Pick another --run-id, or use `resume {run_id}` to continue it.")
        return 2
    run_dir.mkdir(parents=True)
    company = copy_company(run_dir / "company")
    faults = json.loads(args.faults) if args.faults else (sc.faults if sc else {})
    db = str(run_dir / "portal.db")
    portal = start_portal(faults=faults, port=args.port, db_path=db, today=args.today)
    meta = {"goal": goal, "user": args.user, "port": int(portal.url.rsplit(":", 1)[1]), "today": args.today,
            "faults": faults, "model": args.model, "provider": args.provider, "base_url": getattr(args, "base_url", None),
            "min_interval": getattr(args, "min_interval", None), "task": sc.task if sc else args.task, "audit": args.audit,
            "human": args.human, "max_steps": args.max_steps, "answers": list(sc.answers) if sc else []}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    human = make_human(args.human)
    if sc and sc.answers and args.human == "auto":
        human = ScriptedHuman(approve_all=True, answers=list(sc.answers), name="auto-approve")
    rt = build_runtime(run_dir / "company", runs, goal, args.user, backend, human, portal.url,
                       max_steps=args.max_steps, run_id=run_id, auditor_backend=backend if args.audit else None,
                       today=args.today)
    rt.state.listener = progress_printer(args.verbose)
    print(f"Request : {goal}\nPortal  : {portal.url}  (simulated company system, state in {db})\n"
          f"Model   : {args.provider}/{args.model}\nRun dir : {run_dir}")
    try:
        rt.run()
        return finish_run(rt, run_dir, meta["task"], db, meta["answers"])
    finally:
        portal.stop()


def cmd_resume(args) -> int:
    run_dir = Path(args.runs_dir) / args.run_id
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    try:
        backend = make_backend(argparse.Namespace(model=meta["model"], provider=meta.get("provider", "anthropic"),
                                                  base_url=args.base_url or meta.get("base_url"),
                                                  max_tokens=args.max_tokens,
                                                  min_interval=meta.get("min_interval")))
    except LLMError as exc:
        print(f"Cannot resume: {exc}")
        return 2
    data = Agent.read_checkpoint(run_dir)
    portal = start_portal(faults={}, port=meta["port"], db_path=str(run_dir / "portal.db"), today=meta["today"])
    rt = build_runtime(run_dir / "company", Path(args.runs_dir), meta["goal"], meta["user"], backend,
                       make_human(args.human), portal.url, max_steps=meta["max_steps"], run_id=args.run_id,
                       resume_data=data, auditor_backend=backend if meta.get("audit") else None, today=meta["today"])
    rt.state.listener = progress_printer(args.verbose)
    print(f"Resuming {args.run_id} (was: {rt.state.status}; pending: {json.dumps(rt.state.pending)[:200]})")
    try:
        rt.run()
        return finish_run(rt, run_dir, meta.get("task"), str(run_dir / "portal.db"), meta.get("answers", ()))
    finally:
        portal.stop()


def cmd_grade(args) -> int:
    run_dir = Path(args.runs_dir) / args.run_id
    task = args.task or json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")).get("task")
    if not task:
        print("No task known for this run; pass --task expense_mumbai|expense_pune")
        return 2
    g = grade_task(str(run_dir / "portal.db"), task, args.status)
    print(("MATCHES expected outcome" if g.passed else "DOES NOT MATCH:\n  " + "\n  ".join(g.problems)), g.detail)
    return 0 if g.passed else 2


def cmd_models(args) -> int:
    provider = resolve_provider(args.provider)
    if provider in PRESETS:
        try:
            names = list_openai_models(provider, args.base_url)
        except LLMError as exc:
            print(f"Could not list models: {exc}")
            return 2
        print(f"Your {provider} key works. Models you can pass to --model (pick a chat/instruct model that "
              "supports tools, such as a gpt-oss, Llama, Qwen or Mistral one):")
        for n in names:
            print("  " + n)
        return 0
    try:
        names = list_models()
    except LLMError as exc:
        print(f"Could not list models: {exc}")
        return 2
    if not args.test:
        print("Your key works. These names exist, but some may be retired or busy. "
              "Add --test to find out which ones really answer:")
        for n in names:
            print("  " + n)
        return 0
    cands = usable_text_models(names)
    print(f"Testing {len(cands)} text models with one tiny request each (this takes a minute)...")
    for n in cands:
        print(f"  {n:<34} {probe_model(n)}")
        time.sleep(1.0)
    print("Use a model marked WORKS:  python -m operator_agent run --scenario pune --model <name>")
    return 0


def cmd_demo(args) -> int:
    """Run scenarios with the OFFLINE stand-in policy. This is a hand-written script that reacts to tool
    output - NOT a model. It exists to exercise everything around the model end to end."""
    from .offline import GeneratorBackend, expense_standin
    names = list(SCENARIOS) if args.scenario in (None, "all") else [args.scenario]
    print(c("1", "OFFLINE DEMO: a scripted test double plays the model. It shows the platform (permissions, "
                 "approvals, retry safety, verification, evidence), not model intelligence.\n"))
    rc = 0
    for name in names:
        sc = SCENARIOS[name]
        run_id = f"demo_{name}"
        run_dir = Path(args.runs_dir) / run_id
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        company = copy_company(run_dir / "company")
        db = str(run_dir / "portal.db")
        portal = start_portal(faults=dict(sc.faults), db_path=db, today=args.today)
        rt = build_runtime(company, Path(args.runs_dir), sc.goal, sc.user, GeneratorBackend(expense_standin(sc.goal)),
                           ScriptedHuman(approve_all=True, answers=list(sc.answers)), portal.url, run_id=run_id,
                           today=args.today, read_backoff=0)
        if args.verbose:
            rt.state.listener = progress_printer()
        rt.run()
        portal.stop()
        g = grade_task(db, sc.task, "submitted") if sc.task else None
        ok = rt.state.status == "completed" and (g is None or g.passed)
        rc |= 0 if ok else 2
        print(f"{'PASS' if ok else 'FAIL'}  {name:<13} status={rt.state.status:<10} turns={rt.state.step:<3} "
              f"mutations={len(rt.state.mutations)} approvals={len(rt.state.approvals)}  report: {run_dir / 'report.md'}")
        print(f"      {sc.blurb}")
        if g and not g.passed:
            print("      oracle:", "; ".join(g.problems))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="operator_agent", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, runs=True):
        if runs:
            p.add_argument("--runs-dir", default="runs")
        p.add_argument("--today", default=DEFAULT_TODAY, help="the simulated company's date (default %(default)s)")
        p.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("serve", help="run the simulated Expense Portal for manual browsing")
    s.add_argument("--port", type=int, default=8000); s.add_argument("--db"); s.add_argument("--faults")
    common(s, runs=False); s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="give the operator a request (real model)")
    r.add_argument("goal", nargs="?"); r.add_argument("--scenario", choices=list(SCENARIOS))
    r.add_argument("--user", default="priya.sharma")
    r.add_argument("--provider", choices=PROVIDERS, default="auto",
                   help="which model API to use; auto = whichever API key you have set (default %(default)s)")
    r.add_argument("--model", default=None, help="model name (default: claude-sonnet-5-5 / %s for Gemini; other providers need one)" % GEMINI_DEFAULT_MODEL)
    r.add_argument("--min-interval", type=float, default=None,
                   help="not for Anthropic: minimum seconds between calls, to stay under free-tier limits (Gemini 4, others 2)")
    r.add_argument("--max-retries", type=int, default=None,
                   help="not for Anthropic: how many times to retry an overloaded/busy model")
    r.add_argument("--base-url")
    r.add_argument("--max-tokens", type=int, default=4096); r.add_argument("--max-steps", type=int, default=80)
    r.add_argument("--human", choices=["auto-detect", "cli", "auto", "defer"], default="auto-detect",
                   help="who answers approvals/questions: you at the terminal (cli), nobody - pause instead (defer), "
                        "or approve everything (auto; for unattended demos only)")
    r.add_argument("--audit", action="store_true", help="add an independent LLM auditor on top of the mechanical checks")
    r.add_argument("--faults", help='JSON, e.g. \'{"session_expire_after": 8, "submit_error_after_commit": 1}\'')
    r.add_argument("--port", type=int, default=0); r.add_argument("--run-id"); r.add_argument("--task")
    common(r); r.set_defaults(fn=cmd_run)

    z = sub.add_parser("resume", help="continue a paused run")
    z.add_argument("run_id"); z.add_argument("--base-url"); z.add_argument("--max-tokens", type=int, default=4096)
    z.add_argument("--human", choices=["auto-detect", "cli", "auto", "defer"], default="auto-detect")
    common(z); z.set_defaults(fn=cmd_resume)

    m = sub.add_parser("models", help="list the Gemini models your GEMINI_API_KEY can use (also checks the key works)")
    m.add_argument("--provider", choices=[p for p in PROVIDERS if p != "anthropic"], default="auto")
    m.add_argument("--base-url")
    m.add_argument("--test", action="store_true",
                   help="also send one tiny request to each text model to see which ones really work for you")
    m.set_defaults(fn=cmd_models)

    g = sub.add_parser("grade", help="check a finished run against the hand-written ground truth")
    g.add_argument("run_id"); g.add_argument("--task"); g.add_argument("--status", default="submitted")
    g.add_argument("--runs-dir", default="runs"); g.set_defaults(fn=cmd_grade)

    d = sub.add_parser("demo", help="OFFLINE scenarios with a scripted test double (no API key needed)")
    d.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    common(d); d.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
