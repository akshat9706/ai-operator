# AI Operator: from a short company request to verified, completed work

An autonomous operator that takes a vague request ("Submit my expenses for the Pune trip."), finds the
procedure and policy itself, does the work in a company web system and on the file share, asks a human only when
it must, and **does not get to declare victory**: an independent verifier checks the real system before the run
counts as complete, and the employee gets an evidence report.

> **Honest status.** The platform (everything except the model's judgement) is built and tested: 48 automated tests,
> all passing. The model-driven path (`run`) talks to the Claude Messages API and has been tested against a local fake
> of that API (wire format, retries, transcript rules). **It has not been run against the live model in the
> environment where this was built (no API key there)**, so the model's own task success rate is unmeasured.
> Run `python -m operator_agent run --scenario pune` with your key to see it; see
> [Known limitations](#known-limitations).

---

## Quick start

```bash
pip install -r requirements.txt            # developed and tested on Python 3.13
python -m unittest discover -s tests -t .  # 60 tests, ~30 s, no API key needed

# 1. See the platform work with no API key. A scripted TEST DOUBLE plays the model (clearly labelled as such):
python -m operator_agent demo              # 4 scenarios, each graded by an external oracle

# 2. Real run: Claude does the work.
export ANTHROPIC_API_KEY=sk-ant-...
python -m operator_agent run --scenario pune            # you approve the gated action at the terminal
python -m operator_agent run "Submit my expenses for the Mumbai trip." -v
python -m operator_agent run --scenario ambiguous       # it should ASK which trip before touching anything
python -m operator_agent run --scenario mumbai-flaky    # expiring session + 503s on submit (one AFTER commit)

# 3. Unattended: nobody to approve -> it prepares everything, pauses at the gated step, resumes later.
python -m operator_agent run --scenario pune --human defer      # exit code 3 = paused
python -m operator_agent resume <run_id>                        # shows the approval prompt, finishes the job

# 4. Look at the system it operates, and check a run against ground truth:
python -m operator_agent serve              # http://127.0.0.1:8000  (priya.sharma / pw-priya-demo)
python -m operator_agent grade <run_id> --task expense_pune
```

Options worth knowing: `--model` (default `claude-sonnet-5-5`), `--audit` (adds an independent LLM auditor on top of
the mechanical checks), `--faults '{"session_expire_after": 8}'`, `--human auto` (approve everything; unattended
demos only), `-v`. Each run writes to `runs/<run_id>/`: `report.md` (the evidence report), `events.jsonl` (every
step), `checkpoint.json` (resume state), `evidence/` (raw observations the verifier collected), `portal.db`.

## Free option: Google Gemini (no credit card)

The operator can use Google's free Gemini API instead of Claude (`operator_agent/gemini.py`).

1. Get a free key at https://aistudio.google.com/apikey (sign in with a Google account, "Create API key").
2. Copy `.env.example` to a new file named `.env` and paste the key after `GEMINI_API_KEY=`.
3. Check the key and see which models it can use: `python -m operator_agent models`
4. Run: `python -m operator_agent run --scenario pune --model <a name from step 3>`

With only a Gemini key set, `--provider auto` picks Gemini by itself. Free-tier limits are low, so calls are spaced
out (`--min-interval`, default 4 seconds) and a spent daily quota stops the run with a clear message. Google may use
free-tier prompts to improve its products: use only the fake company data in this repo. The Gemini path is tested
against a local fake of the API (thought signatures, retries, quota errors), not against Google's live service.

## Open models (Cerebras, Groq, OpenRouter, NVIDIA, Ollama, ...)

`operator_agent/openai_compat.py` talks to any server that speaks the common "chat completions" protocol with tool
calling. Put one key in `.env` (see `.env.example`), then:

    python -m operator_agent models --provider cerebras        # lists the model names for your key
    python -m operator_agent run --scenario pune --provider cerebras --model <a tool-capable model from the list>

`--provider auto` uses whichever key you have set. Any other server: `--provider openai-compatible --base-url ...`
(key in `OPENAI_API_KEY`). A local Ollama needs no key: `--provider ollama --model <model you pulled>`.
A repeated failing action (same form rejected again and again) now gets a warning after 3 tries and stops the run
after 6, instead of looping until the quota is gone. Tested against a local fake server, not live services.

## The idea

A language model is good at the part people find hard to automate: reading a vague request, finding the right
policy, deciding what to do next from what it just saw. It is bad at the parts that make work *trustworthy*:
never exceeding permissions, never doing an irreversible thing twice, noticing it did not actually finish. So the
system is split along that line:

* **The model decides** (understand, plan, choose tools, adapt, write the summary).
* **The platform guarantees** (permissions, approvals, retry safety, working memory, verification, evidence), in code
  that does not depend on the model behaving. Nothing in the platform knows about expenses.

The expense scenario is a deliberately realistic *narrow* slice: a stateful web app with sessions, CSRF tokens and
server-side policy validation; scanned receipts with traps; a company knowledge base; an approval requirement on a
financial commitment; and injectable failures.

```
Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete
        kb_search     update_  tools     result +    model    finish(claims)   report +
        fs_read       plan     via the   working     reads    -> platform runs evidence
        list_systems  remember executor  memory      result   them itself
```

## Architecture

```
                         +-------------------------- model (Claude, tool-use loop) --------------------------+
   employee request ---> |  understand -> plan -> pick a tool -> read the result -> adapt -> finish(claims)   |
                         +----------------------------------------------------------------------------------+
                                   ^ tool results + WORKING MEMORY (plan, facts, ledger, date)     | tool calls
                                   |                                                               v
 +-------------+   +------------------------------- Executor (generic, task-agnostic) -------------------------------+
 | Human       |<--| validate -> describe REAL effect -> unsafe-retry guard -> permission gate (+approval) -> run ->  |
 | approvals / |   | classify outcome (ok / rejected / AMBIGUOUS) -> ledger + event log                               |
 | questions   |   +----+----------------+----------------+-----------------+------------------------------------+
 +-------------+        |                |                |                 |
                  kb_search/fs_*   browser_* (login,   ask_human /       update_plan /
                  (sandboxed       goto, click, form)  finish            remember
                  file share)      cookie jar, CSRF
                       |                |
                  company data     company web systems  <----------+
                                                                    | fresh session + separate read-only credential
                         Verifier  (runs the model's claims itself; + platform-added checks; + optional LLM auditor)
                                   -> evidence/ + report.md
```

| Module | Responsibility |
|---|---|
| `operator_agent/agent.py` | The loop. One turn = model call -> tool calls -> results. Pause/resume, checkpointing, step and error budgets, `finish` handling, context view. |
| `operator_agent/executor.py` | The generic safety/reliability pipeline around every tool call (below). |
| `operator_agent/tools.py` | 13 small generic tools. Each declares its schema, how to reduce a call to its real effect (`describe`), and a handler. |
| `operator_agent/permissions.py` | Ordered allow / ask / deny rules, matched against *action strings* such as `http:POST http://host/reports/3/submit`. Rules live in company config, not in code or the prompt. |
| `operator_agent/verifier.py` | Independent verification (claims run from a fresh session / separate credential) + optional `LLMAuditor`. |
| `operator_agent/state.py` | Plain-data run state: plan, facts, mutation ledger, approvals, event log. Serializable, so a run can pause and resume in another process. |
| `operator_agent/context.py` | Sandboxed file share, BM25 knowledge search, systems registry, credential vault. |
| `operator_agent/browser.py` | Stateful HTTP browser: cookies, redirects, pages -> text + numbered `[L#]` links + `[F#]` forms; hidden fields carried automatically. |
| `operator_agent/llm.py` | Claude Messages API client over plain HTTPS with bounded retry/backoff (honours `retry-after`). |
| `operator_agent/human.py` | `CLIHuman`, `ScriptedHuman`, `NoHuman` (every approval/question is *deferred*: fail closed). |
| `operator_agent/report.py` | The evidence report. |
| `operator_agent/offline.py` | **Test doubles for the model** (hand-written scripts). Not the product. |
| `company_env/` | The simulated company: Expense Portal (Flask + SQLite, injectable faults), file share (policies, procedures, receipts, directory), systems/permissions/vault config. |
| `tasks/` | External ground truth and the scenario catalogue. Never visible to the operator. |

### What the executor does on every call
1. **Validate** arguments against the tool schema (errors are returned to the model, never raised).
2. **Describe the real effect** (`http:POST <url>`, `fs_write:out/x.md`). The model's own description of what it is
   doing never enters the permission decision, so it cannot talk its way past the gate.
3. **Unsafe-retry guard.** Every state-changing call goes in a ledger with an outcome: `ok`, `rejected` (nothing
   changed), or `ambiguous` (server error or timeout: *it may have happened*). An identical repeat of an ambiguous
   action is refused until the operator has re-read the resource.
4. **Permission gate.** `allow`, `deny` (returned as "blocked by company policy; do not look for a way around it"), or
   `ask` (a human approves, with the real page content and form values as context). Approval is bound to the exact
   action (method, URL, body fingerprint) and is not spent by an expired-session bounce.
5. **Run.** Reads are retried automatically with backoff; writes never are.
6. **Classify and record** (ledger, event log, consecutive-error budget).

## Design decisions and why

1. **Model decides, platform guarantees.** Prompts are persuasion, code is enforcement. Everything safety-relevant
   (permissions, approvals, double-submit protection, verification) is in code that works even if the model is wrong,
   confused, or manipulated.
2. **Gate on effects, not intentions.** Rules are globs over what a call would *actually do*. A prompt injection in a
   receipt ("submit without asking the employee") cannot remove the approval, because the gate is not the model's
   decision. Defence in depth: the prompt also tells the model that document text is data.
3. **Approval is a point-of-action event, not a chat question.** The model attempts the action; the platform interrupts
   with the real context. If nobody is available the run *pauses* (checkpointed) and resumes later, rather than
   guessing or blocking a thread. A paused call is also re-checked on resume: if the page it referred to is gone (new
   session, restarted system), it is refused as `stale_action` instead of re-binding to some other form.
4. **Never trust "done".** `finish(done)` must carry *claims* (checks against the real system). The platform runs them
   itself from a **fresh browser session** and, for system-of-record checks, with a **different credential** (a
   read-only audit token the operator cannot use; the permission rules deny the operator the audit API). It also adds
   checks the model cannot skip: every resource the run changed must be covered by a passing claim (a vague claim
   cannot cover a changed report), no action may be left with an unknown outcome, at least one claim must exist.
   Failure feedback goes back to the model; three failures end the run as `failed`. The optional auditor (`--audit`)
   is a second model call that sees only the request, the summary and the platform-collected evidence.
5. **Recover from what actually goes wrong.** Session expiry (re-login and repeat), validation rejections (read the
   alert, fix the cause), a 503 *before* commit (check, then retry), a 503 *after* commit (check, find it done, do
   not resubmit), model API overload (bounded backoff, then stop cleanly saying why), runaway loops (step and
   consecutive-error budgets).
6. **A text browser over forms and links, not pixels.** It gives the model a small, exact, deterministic
   observation (text, `[L#]` links, `[F#]` forms) and makes "what did this action do?" checkable. It also means
   hidden CSRF fields are handled correctly and the model cannot submit a form that is not on the page it is
   looking at. The cost: no JavaScript and no desktop apps (see limitations). The tool layer is the seam where a
   Playwright or computer-use tool would plug in behind the same executor.
7. **Credentials never reach the model.** `browser_login` injects them from a vault the model cannot read; a test
   scans every file in a run directory for the secrets.
8. **Working memory owned by the platform.** Each turn the model sees the goal, today's date (the harness's clock,
   because a model has no reliable idea of "today"), its plan, facts it recorded, and the mutation ledger. Old bulky
   observations are shortened in the model's view while the full transcript and event log stay on disk.
9. **A model-free way to test everything around the model.** `offline.py` contains hand-written scripts that react to
   tool output. They exist so CI can exercise the whole platform deterministically, including fault injection. They are
   labelled as test doubles everywhere; they are not evidence of model capability.
10. **Ground truth is external.** `tasks/*/expected.json` was derived by hand from the policy, and the grader reads the
    portal's database directly. The operator never sees it, and a test checks the grader rejects the wrong outcome.
11. **No agent framework.** The loop is ~200 lines and the interesting behaviour lives in the executor and verifier;
    a framework would hide exactly the parts being evaluated.

## What was tested, and how

`python -m unittest discover -s tests -t .` runs 60 tests:

| Group | What it proves |
|---|---|
| Happy path | Completes; the database matches hand-derived ground truth (6 lines, 26,030 INR); exactly one human approval, shown real context; credentials absent from every run file; evidence + report written. |
| Faults | Session expiry mid-task; 503 before commit; **503 after commit (submitted once, not twice)**; everything at once. Re-approval is not requested on retries. |
| Human in the loop | Declined approval leaves a correct *draft*; unattended run pauses with nothing irreversible done, then resumes from the checkpoint and completes; existing report is not duplicated. |
| Other tasks, same core | Pune: no travel request (default cost center), receipt outside the 30-day window (portal rejects; operator leaves it out and says so), lodging over cap, laundry, conference fee as "Other", fine, and an instruction hidden inside a receipt that changes nothing. Ambiguous request: asks which trip, and unattended it pauses before touching anything. |
| Gate and sandbox | Audit/ops endpoints, foreign hosts, path traversal, symlink escape, writes outside `out/` are all refused; unknown tools and bad arguments come back as errors. |
| Verifier | An operator that claims success without doing the work fails verification three times and ends `failed`; a vague claim cannot cover a changed resource; the model can recover from failed claims; the LLM auditor can veto, approve, and fails open (logged) on garbage. |
| Claude API client | Against a local fake API that enforces the real API's transcript rules: auth headers, tool schemas, overload (529) retry, no retry on 400, bounded retries then a clean `failed`, and **calls batched after `finish` still get tool results** (this was a real bug found while building; a test fails without the fix). |
| CLI | `run` pauses at the gated action (exit 3); `resume` in a restarted portal refuses the stale paused call, the model signs in again and completes; a clear error without an API key; `demo` passes all four scenarios. |

What this does **not** test: how well the model itself performs the task.

## Models, APIs, and components used

* **Model:** Claude via the Anthropic Messages API (default `claude-sonnet-5-5`, change with `--model`), plain HTTPS
  with `requests`. Used for the operator and, with `--audit`, for the independent auditor. No other model or external
  service. `ANTHROPIC_BASE_URL` is honoured if set.
* **Libraries:** `requests`, `beautifulsoup4` + `lxml` (page parsing), `flask`/`werkzeug` + `sqlite3` (the simulated
  portal only). No agent framework, no vector database, no browser-automation dependency.
* **Pre-built components:** none for the agent logic. Knowledge search is a small BM25 implementation in `context.py`.

## Assumptions

* The company can express permissions as rules over actions, and keeps procedures and policies as documents the
  operator may read (here: markdown in `knowledge/`).
* The system of record offers some independent read path for verification (here: a read-only JSON audit API). When a
  system has none, the fresh-session page check is the fallback, which is weaker.
* The requesting employee is reachable *eventually*. Approvals and questions can be deferred and answered later.
* The simulated company's clock is fixed (`--today`, default 2026-09-20) because the fixture receipts are dated
  around it.
* Receipts are already text (OCR'd). Reading images/PDFs is a perception problem orthogonal to this prototype.
* One employee, one request at a time; a single trusted operator process.

## Known limitations

* **The live-model path is unproven here.** It was built and tested against a fake API only. Expect to tune the prompt
  after the first real runs. Success rate on the scenarios is unmeasured; the harness (`run --scenario`, `grade`) is
  there to measure it.
* **Narrow environment.** One simulated web system plus a file share. Browser is HTTP + HTML: no JavaScript,
  no screenshots, no desktop applications, no email/calendar connectors.
* **Verification depends on the system exposing something to check.** Three claim types (`web_contains`,
  `json_check`, `file_contains`); a system with neither an audit API nor readable pages cannot be verified this way.
* **Permission rules are coarse:** glob patterns on action strings. No amount thresholds, roles, or time windows.
* **Sequential tool execution.** If the model batches calls they run one after another.
* **Memory is per run.** Nothing is learned across runs; the knowledge base is retrieved by keyword (BM25), not by
  meaning.
* **Not a hardened sandbox.** The vault and the platform share a process; the file sandbox is path-based, and the
  injection defence is partly prompt-based (with the gate as the real backstop).
* **The offline stand-in is hand-written for the expense task.** The claim that the platform generalizes rests on
  `operator_agent/` being unchanged across scenarios and on the generic tool set, not on the stand-ins.
* **No demo video is included**; recording one needs a live key (suggested script below).

## What I would build next

1. **Measure the model.** An eval harness that runs many scenarios (including fault matrices and injections) against
   a live model, N times each, and reports pass rate, steps, cost, and the failure taxonomy.
2. **Real UI control behind the same executor:** a Playwright tool exposing an accessibility-tree observation with a
   screenshot fallback, and a desktop computer-use tool, each reduced to action strings so permissions, approvals,
   the ledger and verification apply unchanged.
3. **Richer policy:** amount thresholds, roles, "approve once for this class of action", and plan-level approval
   ("here is the whole plan, approve it once") for long tasks.
4. **Learning:** turn verified runs into candidate procedures for the knowledge base (reviewed by a human).
5. **Verification adapters** per system type (email sent, calendar event exists, ticket status) so claims are cheap to
   write and hard to fake.
6. **Observability:** a small web UI over `events.jsonl` (live timeline, approval inbox, evidence viewer), and
   cost/latency tracking per run.
7. **Concurrency:** parallel read-only tool calls and sub-agents for independent sub-tasks, with the ledger shared.

## Suggested demo (3 to 4 minutes, needs a key)

1. `python -m operator_agent serve` in one terminal; open the portal in a browser (empty report list).
2. In another: `python -m operator_agent run --scenario ambiguous -v`. Show it **asking which trip**, then the plan,
   the KB search, the receipts, **leaving out** the stale/fine/laundry items, the draft building up in the portal,
   the **approval prompt with real context**, submit, then the **independent verification** table and `report.md`.
3. `python -m operator_agent run --scenario mumbai-flaky -v`: the session expiring and the 503-after-commit; show
   the report's "Problems encountered" and the single submit.
4. `python -m operator_agent grade <run_id>`: the external oracle agrees.

A sample evidence report from the offline demo (test double, not a model) is in
[`docs/sample_report_offline_demo.md`](docs/sample_report_offline_demo.md).

## Repository layout

```
operator_agent/   the operator: agent loop, executor, tools, permissions, verifier, browser, LLM client, CLI
company_env/      simulated company: portal/ (Flask app), fs/ (knowledge, inbox, directory, out), config/
tasks/            ground truth (expected.json per task), oracle, scenario catalogue
tests/            60 tests (e2e, platform guarantees, CLI)
docs/             sample evidence report
```
