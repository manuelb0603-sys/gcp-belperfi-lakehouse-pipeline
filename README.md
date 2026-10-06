# GCP PersonalFi BigQuery Data Lakehouse Pipeline

An automated, modern ELT (Extract, Load, Transform) data pipeline engineered inside Google Cloud Platform utilizing **BigQuery** and **Dataform** to build a structured, production-grade Data Lakehouse architecture.

## 🏗️ Architecture Overview
This pipeline implements a multi-layer Lakehouse medallion architecture designed to process, clean, and standardize financial transaction data:
* **Bronze Layer (Raw):** Raw data ingested from external sources and staged dynamically in BigQuery.
* **Silver Layer (Cleaned):** Dataform models (SQLX) enforce strict schema constraints, apply high-precision financial data typing (`NUMERIC`), handle time-zone standardization, and inject metadata processing logs (`processed_at`).
* **Gold Layer (Curated):** Business-ready views optimized for analytics, BI tools, and stakeholder dashboards.

## 🛠️ Tech Stack & Tooling
* **Cloud Provider:** Google Cloud Platform (GCP)
* **Data Warehouse:** Google BigQuery
* **Data Transformation:** Dataform (SQLX)
* **File Uploads:** Python, pandas, and Google Cloud Storage
* **Version Control:** Git & GitHub Integration
* **Security:** GCP authentication

## 📂 Project Structure
```text
├── Scripts/
│   ├── finance/                   # Shared report package (the real implementation)
│   │   ├── __init__.py            # Package marker
│   │   ├── agent_harness.py       # Compatibility re-exports of finance/harness/
│   │   ├── harness/               # Bounded author/validator harness
│   │   │   ├── harness.py         # ReportHarness: author loop, validation cycles, revisions, run stops
│   │   │   ├── protocol.py        # Report/rebuttal markers, author reply parsing, validator JSON schema
│   │   │   ├── prompts.py         # Instruction text sent to the author and validator
│   │   │   ├── feedback.py        # Prioritized revision feedback, resolved findings, cycle progress
│   │   │   ├── html_safety.py     # Mechanical HTML gate and plain-text derivation
│   │   │   ├── trace.py           # RunTrace, HarnessLimits, ToolSpec
│   │   │   ├── errors.py          # Run-stop errors and model-call failure classification
│   │   │   ├── transport.py       # Legacy query_ollama and the query-callable adapter used by tests
│   │   │   └── fallback.py        # Deterministic fallback report
│   │   ├── bigquery_tools.py      # Read-only tools for gold-layer spending and transaction evidence
│   │   ├── chat/                  # Finance chat: deterministic data, analysis, charts, tools, and the chat engine
│   │   │   ├── data.py            # Gold transactions query, TTL cache, refresh(), the data catalog, load_chat_data
│   │   │   ├── profile.py         # Data profile: schema, date coverage, usable/limited dimensions, example questions
│   │   │   ├── merchants.py       # normalize_merchant: descriptor cleanup, payments/credits, aliases
│   │   │   ├── analysis_spec.py   # AnalysisSpec (pydantic), period resolution, validate_and_repair
│   │   │   ├── executor.py        # run_analysis -> table, summary, caption; display_table; narrative number check
│   │   │   ├── charts.py          # build_figure: plotly line/bar/heatmap/pie/treemap with escaped labels
│   │   │   ├── tools.py           # Model tools (analyze_spending, list_values, get_month_snapshot), dispatch, compact results
│   │   │   ├── prompts.py         # Chat system prompt, catalog block, tool descriptions, user-facing messages
│   │   │   ├── session.py         # ChatSession: turns, rolling summary, last spec, history truncation
│   │   │   └── engine.py          # run_turn: model/tool loop, limits, narrative number guard, UI events
│   │   ├── budget.py              # Claude spend backstop: cost ledger, BudgetGuard (daily/run caps)
│   │   ├── email_delivery.py      # Weekly email: build_email_message (HTML + plain part) and send_email (Gmail SMTP)
│   │   ├── html_text.py           # Shared html_to_text (report plain text, history excerpts, Q&A)
│   │   ├── legacy_report.py       # Legacy report mode: BigQuery summary, data payload, one Ollama prompt, query_ollama
│   │   ├── memory.py              # MemoryStore: SQLite short-term history and long-term facts (open_memory)
│   │   ├── memory_search.py       # Hybrid (embedding + BM25) long-fact retrieval
│   │   ├── model_clients.py       # ModelClient layer: OllamaClient, AnthropicClient, model_roles config
│   │   ├── ollama.py              # Compatibility generate_text/embed_text/embed_texts over OllamaClient
│   │   ├── pricing.py             # The one Claude price table: cost_usd, worst_case_call_usd
│   │   ├── qa.py                  # Finance Q&A over the selected report and long-term facts
│   │   ├── report_metrics.py      # Deterministic snapshot: totals, MoM, disposable cash, goal caps, bar widths
│   │   ├── report_models.py       # Typed snapshot, validation, and finding models
│   │   ├── report_requirements.py # Month-specific report requirements checklist built from the snapshot
│   │   ├── report_memory.py       # After a send: open_report_memory, remember_report (short-term row + extracted facts)
│   │   ├── report_runner.py       # Agent-mode email run: progress, generate, archive (_failed rename), budget/dry-run/send, memory
│   │   ├── replay.py              # Replays an archived validator call against another model (validator experiments)
│   │   ├── reporting.py           # Builds the snapshot, tools, and harness for one report
│   │   ├── run_archive.py         # Run archive, live progress files, run metadata (git version), metrics_from_trace, the run index
│   │   ├── run_compare.py         # LLM comparison of archived runs (eval role) and Markdown-to-HTML
│   │   ├── settings.py            # Config loading, validated ReportSettings and code defaults, money parsing
│   │   ├── log.py                 # get_logger: the one console handler ([YYYY-MM-DD HH:MM:SS] message, UTF-8 safe)
│   │   └── status_log.py          # log_status: timestamped console status lines (through finance.log)
│   ├── apps/
│   │   └── finance_app.py         # Gradio UI: Monthly report tab (report and Q&A) and Chat tab
│   ├── reports/
│   │   ├── weekly_report.py       # Entry point (scheduled task): loads config, runs agent mode or legacy mode; re-exports the moved names
│   │   ├── compare_runs.py        # CLI: compare selected runs with the eval model
│   │   └── replay_validator.py    # CLI: replay one run's validator call with another model
│   ├── pipeline/
│   │   ├── upload_csv.py          # Uploads issuer-specific CSV files to GCS and archives them
│   │   ├── run_all_issuers.py     # Runs the uploader for every issuer in config.json
│   │   └── generate_mock_cd.py    # Generates mock Chase and Amex CSVs with recurring merchants (--records, --seed)
│   ├── tools/
│   │   └── model_benchmark.py     # Quick Ollama model speed benchmark
│   ├── tests/                     # Unit tests (package; run with discover -s Scripts/tests -t Scripts)
│   │   ├── chat_test_fixtures.py  # Hand-built transaction fixture plus sparse- and repeating-merchant profile fixtures
│   │   ├── test_chat_*.py         # Finance chat: merchants, spec/periods, executor golden tables, charts, data/cache/catalog,
│   │   │                          #   data profile, tools, engine (scripted fake chat client, follow-up gating), app import smoke test
│   │   ├── test_agent_harness.py  # Harness protocol, parsing, sanitization, retries, tool argument schemas, and feedback tests
│   │   ├── test_budget.py         # Claude pricing, cost ledger, budget guard, harness/app/CLI refusals, no unguarded calls
│   │   ├── test_finance_services.py # Shared reporting/tool wiring tests (including the author tools' argument schemas)
│   │   ├── test_guidance.py       # guidance strict/guided: strict golden prompts, guided checklist/addenda/revisions/limits, settings
│   │   ├── guidance_test_fixtures.py # Scripted harness runs for test_guidance.py (write_golden() regenerates the golden)
│   │   ├── testdata/strict_prompts_golden.json # Strict prompts captured before the guidance setting existed
│   │   ├── test_log.py            # finance.log console format, preformatted lines, encoding fallback
│   │   ├── test_memory.py         # SQLite memory and retrieval tests
│   │   ├── test_model_clients.py  # Model clients (fake Ollama/Anthropic), model_roles, html_to_text parity
│   │   ├── test_email_weekly_report.py # Email workflow, archiving, and memory persistence tests
│   │   ├── test_report_metrics.py # Snapshot goal-cap and bar-width math tests
│   │   ├── test_report_requirements.py # Requirements checklist content and prompt wiring tests
│   │   ├── test_replay.py         # Validator replay: prompt rebuild, role overrides, tally, saved results, cost
│   │   ├── test_run_compare.py    # Run index times/comparisons, run metadata, run gathering, compare CLI, Markdown escaping
│   │   ├── test_run_index_models.py # Run index model columns and ledger costs
│   │   ├── test_settings.py       # ReportSettings parsing/validation, README defaults table, example config
│   │   └── test_run_progress.py   # Live progress files and run-index tests
│   └── local_experiments/         # Local-only experiments (ignored by Git)
├── agent/
│   ├── prompts/
│   │   ├── personality.system.md     # Author persona and grounding instructions
│   │   ├── ui.system.md              # HTML email and presentation constraints
│   │   ├── financial-settings.md     # Financial policy and configuration context
│   │   └── validator.system.md       # Validator-only output and review contract
│   ├── memory/                       # Local SQLite memory files (ignored by Git)
│   └── runs/                         # One folder per run plus index.html (ignored by Git):
│                                     #   run.json, metrics.json, trace.jsonl, progress.json,
│                                     #   progress.log, final_report.html, final_report.txt;
│                                     #   comparisons/ holds reports/compare_runs.py pages (.html + .json);
│                                     #   replays/ holds reports/replay_validator.py results (.json);
│                                     #   cost_ledger.jsonl records every Claude call's cost
├── definitions/
│   ├── bronze/                # Raw landing dependency declarations
│   ├── silver/                # Cleaned SQLX models (e.g., chase_transactions.sqlx)
│   └── gold/                  # Curated analytical views
├── config.json                # Runtime configuration (secrets; ignored by Git)
├── config.example.json        # Example template for creating config.json
├── workflow_settings.yaml     # Dataform project configuration
├── package.json               # Dataform core dependency (`npm run compile` runs `dataform compile`)
├── requirements.txt           # Pinned direct Python dependencies of Scripts/
├── .gitignore                 # Keeps config.json, CSVs, memory databases, runs, and .vscode/ out of Git
├── AGENTS.md                  # Shared project instructions for coding agents
├── CLAUDE.md                  # Claude Code entry point (points to AGENTS.md)
└── README.md
```

## 📤 CSV Upload Workflow
The repository also includes a small CSV upload helper for moving local transaction files into Google Cloud Storage.

Each issuer can be configured independently in the config file with its own:
* local input directory
* backup directory
* GCS bucket and prefix
* GCP project name
* PII columns to remove before upload

The upload script reads the configured issuer, removes any listed PII columns, uploads the cleaned CSV to the target bucket/prefix, and then moves the processed file into the backup folder.

## ▶️ How to Run the CSV Upload Script
1. Copy config.example.json to config.json and update the values for your environment.
2. Make sure your Google Cloud credentials are available. A typical setup is:
   ```bash
   gcloud auth application-default login
   ```
3. Use the workspace virtual environment one level above the project root (`..\.venv`), which already has the required packages. To install them into a fresh environment, from the project root (`requirements.txt` pins the direct dependencies: pandas, numpy, Google Cloud Storage/BigQuery, requests, gradio, plotly, pydantic, anthropic, and Faker):
   ```powershell
   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Upload files for one issuer (run from the project root; the issuer key comes from `issuers` in config.json):
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\pipeline\upload_csv.py amex
   ```
   You can also specify the config file explicitly (a relative path is resolved from the project root):
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\pipeline\upload_csv.py --config config.json --issuer amex
   ```
5. Upload files for every issuer defined in config.json:
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\pipeline\run_all_issuers.py
   ```

This workflow is useful for automating local CSV ingestion into GCS while keeping a backup copy of each processed file.

## 📧 Weekly Financial Report Automation

### Agent harness mode

The report can run in two modes. `report_mode: "legacy"` preserves the original single-prompt flow. `report_mode: "agent"` uses bounded tools, an author model, and an independent validator model. The author keeps the consistent Mogumogu layout and section order defined in `agent/prompts/ui.system.md`. The validator checks current financial values, goal math, evidence, history labels, policy, required content, design, and HTML safety. Failed validation feeds structured feedback back to the author for a limited number of revisions. Email is sent only after validation passes.

How the agent run works:
* **Deterministic snapshot.** `report_metrics.py` computes every number before any model runs: monthly totals, MoM change, disposable cash, and the goal and chart fields (savings shortfall, eligible discretionary total, suggested caps, percent of cap, bar widths, MoM direction, and the top three discretionary categories). The author copies these values; the validator checks the report against them and spot-checks the arithmetic.
* **Requirements checklist.** `report_requirements.py` turns the snapshot into a short, month-specific checklist (about 10–20 lines): the exact required section headings in order, every Monthly Snapshot bar amount (refunds labeled), the four Sushi Stack values with month labels, the goal-method disclosures (for example that discretionary cuts alone cannot reach the target, with the remaining gap, or that no numeric savings target is configured), and each Top 3 row's spend, cap label, and percent of cap. `reporting.py` passes it to the harness, which adds it once as a "REPORT REQUIREMENTS CHECKLIST" block after the snapshot in the author base prompt (so revisions inherit it) and in every validator prompt, where the validator must check each item and report failures as error findings. It is content guidance only: no Python code inspects the report HTML against it. `ReportHarness.run` still works without it (`report_requirements=None`).
* **Author protocol.** To request evidence, the author replies with a small JSON object, `{"tool_calls": [...]}`. When the report is ready, it replies with the raw HTML between `<<<REPORT_HTML>>>` and `<<<END_REPORT_HTML>>>`, so the HTML never has to be escaped inside JSON. The plain-text email part is derived from the HTML. The validator still replies with schema-constrained JSON. Each tool declares a JSON schema for its arguments (`ToolSpec.parameters`; `reporting.tool_parameter_schemas`): `category` (string), `month` (a month-start date, `YYYY-MM-DD`, limited to the selected report month and its predecessor), `limit` (integer, 1-100 for transaction rows), and `query` (string, required for semantic memory). The author's tool list shows these schemas, and the harness checks each call's arguments against them before running the tool; invalid arguments come back to the author as tool-error evidence, like any other tool error (unknown extra keys are ignored).
* **Recovery instead of aborting.** Malformed or truncated author output gets a format retry. Rejected HTML (unsafe markup, oversize, or a missing history label), over-budget or unknown tool calls, and tool errors go back to the author as feedback. A model call that times out, cannot connect, returns HTTP 5xx, or comes back empty is retried once. If it fails again, the run ends with a fail decision: the deterministic fallback report if the author never produced one, otherwise the last sanitized report. Running out of `max_agent_rounds` ends the run the same way (an `agent_rounds_exhausted` trace event; a revision keeps the last sanitized report and the last review's findings). So does the run time limit `max_run_seconds`: before every author or validator call (retries included) the harness checks the deadline set when the run started, and each call's HTTP timeout is capped at the time remaining (at least 30 seconds, at most `ollama_timeout`). A call that times out because of that cap is not retried. Once the deadline passes, the harness records a `run_deadline_exceeded` trace event (elapsed and limit seconds) and stops: the fallback report in the initial phase, otherwise the last sanitized report with the last review's findings. Nothing is sent on a fail.
* **Focused revisions.** Each revision gets the previous HTML and only the top three errors (arithmetic first, then completeness, evidence, history, policy, html, usability, visual design), with the exact snapshot values they concern. The author is told to fix only those with minimal edits. (This is `strict` guidance; `guided` sends every error and allows restructuring, see Guidance modes.) The harness also keeps a short running list (the last 10) of findings that were errors in an earlier cycle and are absent from the latest cycle's errors, matched by the same normalized key as stall detection (category plus message prefix) and shown as "category: message" truncated to about 100 characters. A finding that comes back leaves the list, and a failed review without itemized findings (for example malformed validator JSON twice) leaves it unchanged. When the list is non-empty it appears in the revision request as "ALREADY FIXED IN EARLIER CYCLES (do not reintroduce)" and in the next validator prompt as "FINDINGS RESOLVED IN EARLIER CYCLES", which the validator re-flags only with new verbatim evidence from the current report. Each `validation_progress` trace event records it as `resolved_findings`, and the progress log line for a validation result shows the resolved count. The revision request is still replaced each cycle, not appended.
* **Evidence-grounded findings and rebuttals.** The validator prompt requires every error finding to quote the wrong report text verbatim and cite the conflicting snapshot field, to search the whole report for paraphrases before calling something missing, and not to repeat a prior finding without new evidence. During a revision the author may dispute a finding the previous report already satisfied by adding, after `<<<END_REPORT_HTML>>>`, a block `<<<REBUTTALS>>>[{"finding": "<short finding message>", "quote": "<verbatim report text>"}]<<<END_REBUTTALS>>>` (optionally wrapped in `<!-- -->`). The block is always stripped from the report HTML, malformed blocks are ignored (`author_rebuttals_ignored` trace event), and valid ones are traced (`author_rebuttals`) and passed to the next validator prompt as "AUTHOR REBUTTALS (verify each quote against the report)". The harness does not check quotes or filter findings; the validator decides. Genuine findings must still be fixed.
* **Stall detection.** When a cycle's error findings match the previous cycle's (normalized category plus message prefix), a `validation_stalled` trace event and progress line are recorded. Each `validation_progress` event carries `repeated_error_count`. This is informational and does not change the decision.

Set `dry_run` to `true` to execute the BigQuery, memory, author, validator, and rendering flow without sending email. Each run is saved under `agent/runs/<run_id>/` with its HTML report, plain text, trace, manifest, and metrics. At the start of each run the workflow records run metadata in `progress.json`, and the same keys go into `run.json`: `code_version` (`git rev-parse --short HEAD` plus a `dirty` flag from `git status --porcelain`; if git is missing or fails, `commit` and `dirty` are `null` with an `error` note), `author_role` and `validator_role` (resolved provider, model, options, `think`, and timeout or Claude `max_tokens`/`effort`; never server URLs or credentials), and `harness_limits` (the `max_*` limits), plus `guidance` (the effective mode) and, for guided runs, `effective_harness_limits`. Run comparisons use them to tell code changes from model changes. The run directory is created when the run starts so it can be watched live: `trace.jsonl` is appended as events happen (and rewritten once with the complete trace at the end), `progress.json` holds the current status, phase, round, validation cycle, tool-call count, latest error count, repeated-error count, stalled-cycle count, and author-rebuttal count, and `progress.log` has one timestamped line per model call (with elapsed seconds), tool call, and validation result (the start line shows the run time limit, and a `Run time limit reached` line marks a deadline stop). The same lines print to the console (through `finance/log.py`, the package's single console log handler, in the same `[YYYY-MM-DD HH:MM:SS] message` format; library diagnostics such as memory or progress-file warnings use it too). Follow a run with `Get-Content agent\runs\<run_id>\progress.log -Wait`. A run that raises is renamed with a `_failed` suffix. Failures after the agent harness starts are also archived with the available trace and traceback before control returns to the email workflow. The JSONL trace records model prompts/raw responses (with elapsed seconds, the provider, the normalized `done_reason`, prompt/output token counts, and for Claude the token `usage` including cache reads), tool calls and errors, model-call failures and retries, recovery actions, and each validation cycle's outcome. A `prompt_eval_count` close to `num_ctx` means the prompt was likely truncated; a `done_reason` of `length` means the output hit `num_predict`. For revisions, the validator receives the prior review as comparison context and must independently reassess the new report. Metrics summarize model calls, rounds, tools, retries, report size, and initial-to-final validator-reported error counts. These progress counts help spot resolved or recurring findings; they are based on model findings, not a deterministic correctness score, and do not train or permanently change the model. Traces can contain financial details and transaction evidence, so keep the run archive private and do not commit or share it indiscriminately. Open `agent/runs/index.html` in a browser to view reports and compare objective metrics across models and runs. The index is regenerated automatically after each run. Each row has a number (newest first) and local start and end times (from `progress.json`, or the first and last `trace.jsonl` timestamps for older runs; `running` while a run is live), author and validator models as `provider:model` (from `run.json`/`progress.json` roles, else `metrics.json`, else the `run_started` trace event), and the run's `guidance` when recorded. Compare selected runs shows the metrics side by side in a table and a ready-to-copy `compare_runs.py` command for those runs, and the Comparisons section lists the saved LLM comparisons (see Comparing runs).

#### Guidance modes

`guidance` (top level, default `"strict"`; `model_roles.author.guidance` overrides it) sets how tightly the agent harness prescribes the report. In both modes the facts come from code: the snapshot values (totals, MoM, disposable cash, shortfall, eligible total, remaining gap, every category's current and previous spend) must be correct wherever they are stated, and the code-computed caps stay in the Top 3 table as the baseline plan. Guided mode only relaxes wording, structure, and plans.

| | `strict` | `guided` |
|---|---|---|
| Requirements checklist | Exact headings and exact disclosure sentences | Required sections (recognizable alternative headings allowed, same order) and required facts in any wording; a value may be rounded only where it is labeled approximate |
| Cuts-insufficient disclosure | Required sentence | Required fact |
| Alternative plan | Not offered | Optional "Mogumogu's alternative plan" (different caps, fastest-growing categories, Needs to review), labeled as an advisor suggestion with its own arithmetic: cuts sum to the claimed savings and claimed gap = shortfall - cuts; never a configured goal |
| Author instructions | Unchanged | Plus a guided addendum (`GUIDED_AUTHOR_ADDENDUM` in `harness/prompts.py`) |
| Validator instructions | Unchanged | Plus a guided addendum (`GUIDED_VALIDATOR_ADDENDUM`) and checklist instruction: alternative headings, wording, and plans are allowed; it fails only on wrong or unsupported facts, a missing section or fact, an unreconciled or mislabeled alternative plan, unsupported merchant/behavior claims, and the HTML, safety, and visual-design requirements; findings still quote the report verbatim |
| Revisions | Top three errors; "fix only the listed issues with minimal edits" | Every error (still prioritized, with snapshot values); "fix every listed issue; you may restructure sections" |
| Tool limits | `max_tool_calls`, `max_calls_per_tool` as configured | Both doubled (for example 12/3 becomes 24/6); rounds, cycles, `max_run_seconds`, and the Claude cost guard are unchanged |

Use `strict` for small local authors (for example `gemma4:e4b`), which do best with exact instructions and minimal edits. Use `guided` for strong authors such as Claude Opus, which can write a better-structured report and a smarter savings plan when they are not held to fixed sentences. Strict prompts are byte-identical to the prompts before the setting existed. The effective guidance is recorded in the `run_started` trace event (with the effective limits, and `configured_limits` when guided), `progress.json` and `run.json` (`guidance`, plus `effective_harness_limits` when guided), and `metrics.json`, and run comparisons show it, so strict and guided runs can be told apart.

#### Comparing runs

`Scripts\reports\compare_runs.py` asks the `eval` model role to compare two or more archived runs (at most 10) and explain the results: which setup did best, why each run passed or failed, slow phases and token-limit hits, whether validator findings look genuine or false positives, author versus validator models, and concrete model, option, and prompt recommendations. From the project root:

```powershell
..\.venv\Scripts\python.exe Scripts\reports\compare_runs.py <run_id> <run_id> [...] [--config PATH] [--dry-run-prompt]
```

For each run it gathers a bounded summary (about 60k characters of prompt for five runs): start, end, and duration; the code version (`code_version` commit and dirty flag, or `not recorded` for older runs); author and validator provider/model, the recorded role settings (`author_role`/`validator_role`), and run limits; the run's `guidance` (`not recorded` for older runs, which behaved as strict); status, rounds, tool calls, validation cycles, retries, error counts, and report size; each validation cycle's decision and error findings (category, message, evidence, truncated; repeats collapsed); stall, rebuttal, retry, deadline, and exception events; per-role token counts, call times, and `length` stops from the trace; and a head-and-tail excerpt (about 1,500 characters) of `final_report.txt`. The eval instructions tell the model to compare code versions before attributing outcome differences to the models, since runs on different commits or with uncommitted changes may differ because of code or prompt changes. `--dry-run-prompt` prints the prompt, its size, and (for Claude) a maximum cost estimate without calling the model. The answer is Markdown, converted to escaped HTML with no external resources, and saved as `agent/runs/comparisons/<timestamp>_<n>runs.html` with a `.json` sibling (created time, run IDs, eval provider/model, prompt size, token usage, `done_reason`, estimated cost, and the Markdown); the index is then regenerated. Unknown run IDs, a failed model call, a refusal, or an empty answer print an error, save nothing, and exit non-zero. With an Anthropic eval model the summaries (findings, metrics, and report excerpts) are sent to the Claude API, and the estimated cost (input and output tokens at the prices under Model providers) is printed and saved.

#### Runtime config knobs

The live project configuration is read from `config.json` (not `config.example.json`). Every key the report code reads is loaded once into a frozen `ReportSettings` (`Scripts/finance/settings.py`, `ReportSettings.from_config(config)`), which applies the defaults below and validates the values: a bad value stops the run with an error that names the key (for example a non-positive limit, an unknown `report_mode`, a `memory_schema` with other than letters, digits, and underscores, or a non-numeric `Savings_goal_target`). The email workflow prints it as `Failed to load config: ...` and stops. `ReportSettings` keeps `Gmail_app_credentials` as a plain field but masks it in its printed form. The agent mode adds a few important runtime controls that keep the report process bounded, inspectable, and safe:

* `report_mode`: set to `"agent"` to use the structured author/validator flow. `"legacy"` (the default when the key is omitted) keeps the original one-shot report generation flow. Other values are rejected (case is ignored).
* `guidance`: `"strict"` (default) or `"guided"`; how tightly agent mode prescribes the report's wording and structure (see Guidance modes). `model_roles.author.guidance` overrides it for the author (it is not accepted on other roles). Other values are rejected (case is ignored).
* `dry_run`: when `true`, the project runs the full BigQuery + prompt + validation + archive flow without sending email. This is recommended while tuning the model or validating report quality. Default `false`. Only agent mode honors it; legacy mode currently sends regardless.
* `model_roles`: the primary way to configure the models. Each role (`author`, `validator`, `eval`, `chat`) names its provider (`"ollama"` or `"anthropic"`; any role may use either) and model, plus its own options; see Model providers below. `eval` is optional and falls back to the validator; `chat` (the app's finance chat) is optional and falls back to `model_name` (see Finance chat).
* `model_server_url`: the shared Ollama server for every Ollama role (and embeddings), either the base URL (`http://your-host:11434`) or the full endpoint (`http://your-host:11434/api/generate`); the generate and embed endpoints are derived from it. A role may override it with its own `base_url`.
* `model_name`: an Ollama model, still required: legacy mode, long-term fact extraction (unless `fact_model_name` is set), and the finance app (`Scripts/apps/finance_app.py`)'s Q&A use it, and it is the author model when `model_roles.author` is absent. Keep it set to an Ollama model even when the author role runs on Claude.
* Legacy fallbacks, still read when `model_roles` (or one of its roles) is absent: `validator_model_name` (validator model; defaults to `model_name`) and `ollama_options` (per-role Ollama settings, `{"author": {...}, "validator": {...}}`, with the same fields as a role's `options` plus `think`). A role present in `model_roles` ignores both.
* Ollama `options` (inside a `model_roles` entry, or `ollama_options[role]`): request generation settings such as `temperature`, `num_predict`, `repeat_penalty`, and `num_ctx`. A role may also set `"think": true|false`, sent as Ollama's top-level `think` flag for reasoning models. Hidden thinking counts toward `num_predict`: a reasoning validator can spend its whole budget thinking and return no JSON, so the recommended setup sets `"think": false` for the validator (or raise its `num_predict` substantially if you enable thinking). Unset keys are not sent, so the server's defaults apply. `num_predict` caps output so a model stuck in a loop stops quickly instead of running until `ollama_timeout`. About `32768` for `num_ctx` covers the largest validator prompts; a much larger context on a 16 GB server wastes memory and can push the model partly onto the CPU.
* `max_agent_rounds`: how many author rounds (tool requests and format retries) are allowed per phase (initial report, then each revision). Running out ends the run with a fail decision (the fallback report in the initial phase, the last sanitized report in a revision), which is archived normally and never sent.
* `max_tool_calls`: total number of tool calls permitted across the run.
* `max_calls_per_tool`: per-tool call ceiling to prevent one tool from being overused.
* `max_validation_cycles`: total number of validator reviews (validation cycles, including cycles where the HTML gate rejects the report) before the report is rejected, so at most this value − 1 revisions. Must be a positive integer (the config is rejected otherwise). Default `3` (three reviews, two revisions); the recommended setup uses `5`. Before an earlier change the value counted revisions after the first review (reviews = value + 1); an existing config with `4` now means 4 reviews and 3 revisions, so raise it by one to keep the previous behavior.
* `max_run_seconds`: wall-clock limit for the whole agent run in seconds (default `2700`, 45 minutes; `0` or `null` disables it; negative values are rejected). When it is reached before a model call, or a call's deadline-capped timeout expires, the run stops with a fail decision (fallback report before the first author report, otherwise the last sanitized report) and nothing is sent. Each call's timeout is the smaller of `ollama_timeout` and the time remaining (at least 30 seconds).
* `max_report_bytes`: maximum generated HTML size allowed before the report is rejected as too large.
* The `max_*` limits other than `max_run_seconds` must be positive integers.
* `days_back`: legacy mode only; how many days of BigQuery history the summary query looks back over. Agent mode reads the latest month and its predecessor.
* `ollama_stream`: legacy mode only; keep `false` so the model returns a complete response.
* `ollama_timeout`: the per-call request timeout for model calls (Ollama and Claude) in seconds (default `600` when omitted); a role may override it with its own `timeout`. In agent mode a call is also capped at the time left under `max_run_seconds`. Increase this for larger or slower models that need more time to reason. A typical starting point is `1800` (30 minutes), with larger models sometimes needing more.
* `embedding_model_name`: Ollama model for long-term fact and query embeddings. Default: `nomic-embed-text`; pull it on the Ollama server before using semantic memory.
* `memory_schema`: name of the SQLite memory file under `agent/memory/` (see Report Memory). Letters, digits, and underscores only.
* `Monthly_budget`, `Fixed_costs`, `Savings_goal_target`: monthly deposit, fixed costs, and the single numeric monthly savings target used by the snapshot. Amount strings such as `"5k"` or `"Rent: $2000, Car Loan: $500"` (summed) are parsed. `Savings_goal_target` must be a number when set; agent reports need parseable `Monthly_budget` and `Fixed_costs`. `Vehicle_Info` is legacy-mode context.
* `category_goals`: optional `{"Category": amount}` monthly goals. A configured goal replaces the suggested cap for that category and is labeled as configured.
* `chat_data_cache_seconds`: how long the finance chat keeps the loaded transactions in memory before querying BigQuery again (default `900`; `0` disables the cache).
* `merchant_aliases`: optional `{"CLEANED NAME": "Display name"}` map applied by the finance chat after merchant normalization (keys match the cleaned upper-case name exactly or as a word prefix, for example `"WHOLEFDS MKT": "Whole Foods"`).
* `chat_history_turns`: how many recent chat turns are sent to the chat model verbatim; older turns are folded into a rolling summary (default `6`).
* `chat_max_tool_rounds`: tool-call rounds the chat model may use for one question before it must answer (default `3`).
* `chat_turn_seconds`: time limit for answering one chat question, in seconds (default `120`).
* `project_name`, `email_sender`, `email_listings`, `Gmail_app_credentials`: GCP project and email delivery settings (see Setup Instructions).

Optional keys not in `config.example.json`: `fact_model_name` (model for extracting long-term facts after a sent report; defaults to `model_name`), and `qa_report_context_chunks` / `qa_long_fact_limit` (context sizes for the finance app (`Scripts/apps/finance_app.py`); default 5 each, capped at 8 and 10). If `model_server_url` or `model_name` is missing, the finance app (`Scripts/apps/finance_app.py`) falls back to `http://localhost:11434/api/generate` and `gemma4:e4b`.

##### Code defaults

These are the values used when a key is omitted. They are defined only in `DEFAULTS` in `Scripts/finance/settings.py`, and `test_settings.py` checks that this table matches it. `validator_model_name` and `fact_model_name` default to `model_name`. `config.example.json` shows the recommended setup, which differs in places (for example `max_validation_cycles` `5` and `ollama_timeout` `1800`).

| Key | Default |
|---|---|
| `report_mode` | `"legacy"` |
| `guidance` | `"strict"` |
| `dry_run` | `false` |
| `ollama_stream` | `false` |
| `ollama_timeout` | `600` |
| `max_agent_rounds` | `6` |
| `max_tool_calls` | `12` |
| `max_calls_per_tool` | `3` |
| `max_validation_cycles` | `3` |
| `max_report_bytes` | `250000` |
| `max_run_seconds` | `2700` |
| `days_back` | `60` |
| `embedding_model_name` | `"nomic-embed-text"` |
| `memory_schema` | `"memory"` |
| `category_goals` | `{}` |
| `email_listings` | `[]` |
| `qa_report_context_chunks` | `5` |
| `qa_long_fact_limit` | `5` |
| `chat_data_cache_seconds` | `900` |
| `merchant_aliases` | `{}` |
| `chat_history_turns` | `6` |
| `chat_max_tool_rounds` | `3` |
| `chat_turn_seconds` | `120` |
| `anthropic_daily_budget_usd` | `15.0` |
| `anthropic_run_budget_usd` | `5.0` |
| `anthropic_cost_ledger_path` | `"agent/runs/cost_ledger.jsonl"` |

#### Model providers

Every model call goes through one client layer, `Scripts/finance/model_clients.py`: `OllamaClient` (local Ollama, also used for embeddings) and `AnthropicClient` (Claude, through the official `anthropic` Python SDK). `model_roles` is the primary way to configure every role; each role picks its provider (`"ollama"` or `"anthropic"`, for any role) and carries its own options. The tested recommendation (also in `config.example.json`):

```json
"model_roles": {
  "author":    {"provider": "ollama", "model": "gemma4:e4b",
                "options": {"temperature": 0.3, "num_predict": 12000, "repeat_penalty": 1.1, "num_ctx": 32768}},
  "validator": {"provider": "ollama", "model": "gemma4:12b", "think": false,
                "options": {"temperature": 0.1, "num_predict": 4000, "num_ctx": 32768}},
  "eval":      {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "max_tokens": 16000}
}
```

For a fully local setup, use an Ollama eval model instead, for example `"eval": {"provider": "ollama", "model": "gemma4:12b", "think": false}`, or leave `eval` out to reuse the validator. Any role can switch provider by changing its entry.

* **Roles.** `author` writes the report and `validator` reviews it. `eval` writes the run comparisons from `Scripts\reports\compare_runs.py` (see Comparing runs); it is optional and when unset it falls back to the validator's settings. `chat` answers questions in the app's Chat tab (see Finance chat); when unset it uses `model_name` with `ollama_options["chat"]`. `model_roles` is authoritative: a role present there ignores the legacy keys. A role missing from `model_roles` (or the whole key missing) falls back to the legacy keys: `model_name` / `validator_model_name`, `ollama_options[role]`, and `model_server_url`, so older configs keep working unchanged. Roles are resolved once when the config is loaded (`ReportSettings`), so an unknown provider is reported at startup.
* **Ollama role fields.** `model`, optional `options` and `think` (when neither is given, `ollama_options[role]` applies), optional `base_url` (default `model_server_url`) and `timeout` (default `ollama_timeout`). Requests are identical to before: one prompt, `format` for the validator's JSON schema, `options` and `think` only when set.
* **Claude role fields.** `model`, optional `max_tokens` (default `16000`, at most `64000`; thinking tokens count toward it), `effort` (`low`/`medium`/`high`/`xhigh`/`max`; omit it for `claude-haiku-4-5`, which does not support effort), `cache_system` (default `true`), `stream` (default automatic: stream when `max_tokens` is above `16000`; `true`/`false` force it), and `thinking` (only `"adaptive"`, sent as `{"type": "adaptive"}`, for models where thinking is off by default). Claude Opus 5.5 and Sonnet 5.5 think adaptively by default and `effort` controls how deeply; thinking cannot be disabled, so `thinking: {"type": "disabled"}`, `budget_tokens`, and sampling parameters (temperature etc.) are never sent, and the thinking display stays at the API default. `ollama_options`, `options`, and `think` do not apply. The validator uses structured output (`output_config.format` with the same JSON schema). A refusal is recorded with `done_reason: "refusal"` and its `stop_details`; a `max_tokens` stop is treated like Ollama's `length`. The SDK is created with `max_retries=1`, and its timeout, connection, 429, and 5xx errors count as model-call failures for the harness's single retry. A 4xx error such as a bad request (for example an invalid model ID) is not retried: it is raised, the run aborts and is archived with a `_failed` suffix, and nothing is sent.
* **Switching a role to Claude.** Install the SDK into the workspace venv (`..\.venv\Scripts\python.exe -m pip install anthropic`; Ollama-only setups do not need it), set the role to `{"provider": "anthropic", "model": "claude-sonnet-5-5"}`, and provide credentials through the environment: the `ANTHROPIC_API_KEY` environment variable or an `ant auth login` profile. Never put an API key in `config.json`.
* **Models and cost** (per million tokens, input / output): `claude-opus-5-5` $4 / $20, `claude-sonnet-5-5` $2 / $10, `claude-haiku-4-5` $1 / $5, `claude-fable-5-1` $10 / $50, `claude-opus-5` $5 / $25, `claude-sonnet-5` $2 / $10. Cache reads cost 0.1x and cache writes 1.25x the input price. The one price table is `Scripts/finance/pricing.py`; a Claude model missing from it is priced at the Fable rate for budgeting, never as free.
* **Streaming.** The SDK refuses non-streaming requests that could run longer than 10 minutes (about 21k `max_tokens`), so roles above `16000` `max_tokens` stream automatically (`messages.stream`, final message only); the reply, `usage`, `done_reason`, and refusal handling are the same as without streaming and the trace records `streamed` in the response metadata. The role timeout still bounds the whole call: the SDK applies it per network read and the client also checks the total time between stream events, raising a timeout the harness retries once like any other. Read errors and `overloaded_error`/`api_error` events in the middle of a stream count as model-call failures too.
* **Prompt caching.** For Claude, the stable instruction text goes in a cached system block and only the per-call content is the user message: the author's base prompt (context, objective, snapshot, checklist, tools, protocol) is the system part and the tool evidence, revision request, and retries are the user part; for the validator, `validator.system.md` is the system part and the report, snapshot, evidence, and prior review are the user part. Repeated author rounds within a run therefore read the base prompt from the cache. Ollama still receives exactly the same single concatenated prompt as before, and the trace's `prompt` field always records that full text. Trace `model_response` events from Claude include `usage` (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`) so cache hits are visible; `run_started`, `model_call`, `progress.json`, and `metrics.json` record the provider and model per role.
* **Privacy.** When a role uses Anthropic, its prompts leave your machine: the snapshot, report HTML, tool evidence (including transaction-level rows from `get_transaction_detail`), memory excerpts, and validator feedback are sent to the Claude API. Keep every role on Ollama if the data must stay local. Embeddings and memory fact extraction always use Ollama.
* **Recommended setup** (from testing on a 16 GB GPU): author `gemma4:e4b`; validator `gemma4:12b` with `think: false` and `num_predict` `4000`; `num_ctx` `32768` for both; `max_validation_cycles` `5`; `max_run_seconds` `2700`; eval optional (Claude `claude-sonnet-5-5`, or local `gemma4:12b`). `qwen3.5:9b` was unreliable as a validator without thinking and ran out of tokens (thinking consumed `num_predict`) with it. Models of 26B parameters and larger did not fit a 16 GB GPU and slowed to partial CPU inference.

##### Cost backstop

Claude spend has three independent limits:

1. **Run cap** `anthropic_run_budget_usd` (default `5.0`): Claude spend for one report run (the run id in `agent/runs`; the app's reports get an `app_<timestamp>` id). For a replay, a comparison, or a chat session it is the spend of that one invocation.
2. **Daily cap** `anthropic_daily_budget_usd` (default `15.0`): all Claude spend recorded today (local date) by every program: the weekly email (including the scheduled task), the finance app (`apps/finance_app.py`), `reports/compare_runs.py`, `reports/replay_validator.py`, and the finance chat.
3. **Account spend limit** in the Claude Console under **Settings → Limits**. It is the final backstop for anything outside this project; set it as well.

Both caps are on even when `config.json` omits the keys. Only an explicit `0` or `null` disables a cap, and the program prints a warning at startup when one is disabled. Before every Claude request (including the harness's retry and format retries), `AnthropicClient` asks its `BudgetGuard` (`Scripts/finance/budget.py`, attached by `build_model_client`) for the call's worst case: the prompt at the list input price (characters / 3.5 tokens) plus `max_tokens` of output. If today's spend plus that would pass the daily cap, or the run's spend plus that would pass the run cap, the call is refused without contacting the API (`BudgetExceededError`). A report run then stops like an unavailable model: fail decision, last sanitized report or fallback, never sent; the trace records a `budget_exceeded` event with the figures, `progress.log` says `Claude budget guard refused the validator call: Claude daily budget reached: $14.85 of $15.00 spent today; ...`, and the index shows status `budget_exceeded`. The app shows the same message instead of the report. `replay_validator.py` and `compare_runs.py` print it and exit `2`. Ollama calls are never budgeted.

The spend ledger `agent/runs/cost_ledger.jsonl` (`anthropic_cost_ledger_path`, relative to the project root) has one JSON line per Claude call: success, refusal, an empty answer, or a timed-out or interrupted call (estimated as its input at list price, marked `"estimated": true`). Each line holds `timestamp` (UTC), `date` (local), `run_id` (or `replay`, `compare`, `chat`, `other`), `role`, `model`, `outcome`, `usage`, and `cost_usd`. Lines are appended with a single write under a lock file, so the app and the scheduled run can share it. A missing ledger means $0 spent. The guard fails closed: if the ledger cannot be read (or has a corrupt line) or a spend record cannot be written, Claude calls are refused until it is fixed. Each run starts with `Claude budget: $X spent today of $Y daily; run cap $Z` in `progress.log`; Claude `model_response` events carry `cost_usd` (also shown in `progress.log`), `progress.json` and `metrics.json` record `anthropic_cost_usd`, and the index has a `cost` column (blank for runs without Claude calls). For a run whose `metrics.json`/`progress.json` has no `anthropic_cost_usd` (runs from before costs were recorded), the index shows the ledger's total for that run id (a `_failed` folder also matches the id without the suffix); the ledger is read once per index build, from the configured path (the email run, replays, and comparisons pass it; otherwise `cost_ledger.jsonl` inside the runs folder).

##### Claude experiments

`Scripts\reports\replay_validator.py` (logic in `finance/replay.py`) replays one archived validator call against another model, to compare verdicts, consistency, tokens, and cost without rerunning the author. It reads the run's `trace.jsonl`, takes the first validator `model_call` of the chosen cycle (default the last; format retries are skipped), and splits the traced prompt on `VALIDATE THIS REPORT:` into the validator instructions (system, cached for Claude) and the cycle's report, snapshot, evidence, and prior review (user). `--current-prompt` swaps today's `agent/prompts/validator.system.md` in as the system part (plus today's guided addendum when the replayed run was guided). The call uses the configured validator role (`model_roles.validator`) unless the flags override it for this replay only: `--role-model PROVIDER:MODEL` (for example `anthropic:claude-haiku-4-5` or `ollama:gemma4:12b`), `--effort low|medium|high` (Claude), `--think true|false` (Ollama's flag; for Claude `true` requests adaptive thinking and `false` is rejected), and `--max-tokens N` (Claude `max_tokens`, Ollama `num_predict`). From the project root:

```powershell
..\.venv\Scripts\python.exe Scripts\reports\replay_validator.py 20261005T133631_gemma4_e4b --role-model anthropic:claude-haiku-4-5
..\.venv\Scripts\python.exe Scripts\reports\replay_validator.py 20261005T105834_gemma4_e4b --role-model anthropic:claude-sonnet-5-5 --effort low
..\.venv\Scripts\python.exe Scripts\reports\replay_validator.py 20261005T105834_gemma4_e4b --cycle 2 --role-model anthropic:claude-opus-5-5 --effort high --max-tokens 32000 --current-prompt --repeat 3
..\.venv\Scripts\python.exe Scripts\reports\replay_validator.py 20261005T105834_gemma4_e4b --role-model anthropic:claude-opus-5-5 --dry-run-prompt
```

Each call prints the decision (and the decision the run recorded for that cycle), every audit gate status, the error findings (category, message, truncated evidence), tokens in/out with cache reads and writes, seconds, `done_reason`, and for Claude the estimated cost (prices above; cache reads at 0.1x and cache writes at 1.25x the input price). `--repeat K` runs the same call K times and prints a pass/fail tally to check consistency. Every result, including failed calls, is saved as `agent/runs/replays/<timestamp>_<run_id>_<model>.json` (with `_r<n>` for repeats); the run index skips that folder as a run and lists the replays. `--dry-run-prompt` prints the system/user sizes and the worst-case cost (the budget guard's estimate: characters / 3.5 input tokens at the list price plus `max_tokens` of output, times K) without a call. A refusal, malformed or non-schema JSON, or a failed model call prints a clear message and exits `1`; an unknown run or cycle, or invalid flags, exit `2` without a call. With a Claude model the report, snapshot, and evidence in the prompt are sent to the Claude API.

A Claude author needs room for thinking plus a ~20 KB HTML report, so give it about `32000` `max_tokens` (requests stream automatically above `16000`) and a generous timeout:

```json
"author": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high", "max_tokens": 32000, "timeout": 1800}
```

The code defaults used when a key is omitted are listed under Code defaults above; `config.example.json` shows the recommended values. The limits are intentionally conservative for a demo or controlled deployment: they keep the model bounded, prevent runaway tool use, and make each run easy to review in the generated archive.

The agent's transaction-detail tool reads only from the gold `fact_transactions` table and is bounded by month, category, and row count. `get_semantic_memory` ranks a focused agent-written query with embedding cosine similarity and BM25 keyword relevance, then combines rankings with reciprocal-rank fusion and deduplicates facts. Semantic matching can catch paraphrases while BM25 matches exact terms and accounts for term frequency and document length. BM25 remains available if Ollama embeddings are unavailable. Both rankers scan the existing SQLite fact set directly; no separate search service or index is needed. Episodic memory and report history remain separate read-only tools. Previous reports and semantic facts are historical and cannot replace current gold-layer values. Existing SQLite databases are migrated in place; missing vectors are embedded lazily during semantic search.

The `Scripts/reports/weekly_report.py` script is the entry point of the weekly email; the work lives in `finance/`: `report_runner.py` (agent mode: live progress, report generation, archive, budget/validation/dry-run outcomes, delivery, then memory), `legacy_report.py` (legacy mode's BigQuery summary, prompt, and Ollama call), `email_delivery.py` (the multipart message and Gmail send), and `report_memory.py` (saving the sent report and extracting facts). It generates an intelligent weekly financial insights email that:
* Queries your **BigQuery Gold Layer** (`fact_monthly_spending`) to extract spending trends, category analysis, and key metrics
* Processes data through a local **Ollama LLM** (configurable via `model_server_url` & `model_name` in config) to generate contextual insights with playful Japanese food puns from your assistant Mogumogu-chan 🍣🍤
* Uses local SQLite memory and Ollama embeddings to retrieve relevant long-term facts on demand, alongside recent report history
* Saves each completed report and extracts reusable financial facts for future reports
* Outputs an attractive HTML report styled for email readability

### Finance chat

The **Chat** tab of the finance app (`Scripts/apps/finance_app.py`) answers free-form questions about your card spending with a chart, a table, and a short Mogumogu-style explanation. The model only routes and narrates: it picks a tool and fills in a declarative `AnalysisSpec` (metric, up to two dimensions, period, filters, top N, transform, comparison, chart); Python resolves the dates, repairs or rejects the spec against the data catalog, runs it with pandas, and draws the chart. The model never writes code and never sees transaction rows, only the catalog, the compact tool results (at most about 1.5 KB each), and the captions.

Run the app from the project root (it serves on `http://127.0.0.1:7860` only; there is no public share link):

```powershell
& "..\.venv\Scripts\python.exe" Scripts\apps\finance_app.py
```

The **Monthly report** tab is the original month report and Q&A. In the **Chat** tab, type a question or click an example. The example chips are generated from the data profile when the page loads (and on **Refresh data**), so they only use dimensions and dates the data supports, for example:

* "Monthly net spend by category, last 6 months of data"
* "Compare Amex vs Chase by month"
* "Which weekday do I spend the most?"
* "Top categories in Wants vs Needs"
* "Top 5 merchants in Sep 2026" (only when merchants repeat enough to group by; see Data profile)
* Follow-ups such as "now only Amex" or "same thing by week" start from the previous analysis; a new question ("Which weekday do I spend the most?") starts from the defaults (see History below).
* "How was September?" uses the monthly-report snapshot (total, change from the prior month, disposable cash, savings shortfall, top discretionary categories).

If the data cannot be loaded or profiled, safe static examples are shown (`prompts.STATIC_EXAMPLES`).

Each answer shows a collapsible **How I computed this** message (the plain-language caption, any repairs such as "Interpreted category 'Shoping' as 'Shopping'", and the spec), the narrative, the plotly chart, and the aggregated table (at most 25 rows) with friendly column names and units ("Month", "Category", "Net spend ($)", "Transactions"), months as "Aug 2026", and full weekday names (`executor.display_table`; the model's summary still uses the raw table). **Focus month** sets the month used by the `focus_month` period and the snapshot tool (default: the latest month with data). **Refresh data** drops the cached transactions and queries BigQuery again (otherwise the frame is reused for `chat_data_cache_seconds`).

#### Data profile

`profile.build_data_profile(frame, today=...)` describes what the loaded data can support. It is built once per data load next to the catalog (`data.load_chat_data` returns the frame, catalog, and profile and caches the last two with the frame) and holds:

* **Schema**: the gold `fact_transactions` columns the chat uses (issuer, transaction_date, description (normalized into merchant), amount, transaction_type, category, parent_group) with their types and meaning.
* **Date coverage**: first and last date, complete and partial months, the latest complete month, and how many days the data ends before today. A month is complete when the data reaches within 2 days of both its first and last day and the month is over.
* **Transaction types** present (only Sale, Return, and Adjustment are analyzed, as in gold `fact_monthly_spending`).
* **Per dimension** (category, parent_group, issuer, merchant, weekday, day_of_month, month, quarter, week): the number of values, the median and minimum transactions per value, the share of transactions and gross spend in the top 10 values, and a status with a plain-language reason:

| Dimension | Usable when | Limited when | Unusable when |
| --- | --- | --- | --- |
| category, parent_group, issuer, merchant | otherwise | the median value has fewer than 3 transactions, or there are more than 10 values and the top 10 cover less than 40% of transactions | fewer than 2 values |
| weekday | otherwise | the median weekday has fewer than 3 transactions | fewer than 2 weekdays |
| day_of_month | otherwise | the median day has fewer than 2 transactions | |
| month | at least 2 complete months | 1 complete month, or only partial months | a single month |
| quarter / week | at least 2 quarters / 4 weeks | (week) 2 or 3 weeks | a single quarter / week |

The profile drives the chat:

* **Prompt**: the system prompt gets a DATA PROFILE block (coverage, latest complete month, usable dimensions, limited/unusable ones with reasons, transaction types; no amounts or percentages) and a rule not to offer analyses on limited or unusable dimensions unless the user explicitly asks, and then to explain the limitation. When merchant is limited, the catalog shows only the merchant count instead of the top 30 names.
* **Repairs**: `validate_and_repair(..., profile=...)` replaces a limited or unusable group-by dimension with the best usable alternative (merchant to category or group, category and group to each other, day of month to weekday), so "heatmap of activity by store" becomes category by weekday; an unusable dimension without an alternative (for example a single card) is dropped. Grouping by merchant is kept when the user filters to specific merchants (`merchant` or `merchant_contains`). Time dimensions are kept with a note.
* **Relative periods** anchor to the end of the data when it ends before the current month: "this month", "this quarter", and "year to date" end at the last data date, "last month" is the latest complete month, and "last N months"/"last quarter" end with it (for example "Data ends Aug 19, 2026; 'this month' = Aug 2026 (partial)."). Explicit custom dates are kept as given (clamped to the data).
* **Charts**: `auto` draws a line only with at least 3 time points, a heatmap only when both dimensions are usable and the grid is at most 15 x 12, a pie only with at most 6 non-negative slices, and bars otherwise; explicit chart requests are repaired with the same limits.
* **Narrative**: profile repairs and period anchoring are `Limitation` notes; the tool result summary carries them as `limitations` (kept when other detail is trimmed), and the system prompt tells the model to mention each one in plain words.

How a turn works (`Scripts/finance/chat/engine.py`, `run_turn`):

* **Tools** (`tools.py`): `analyze_spending` (parameters: the `AnalysisSpec` JSON schema, flattened), `list_values` (`field`: `category`, `parent_group`, `issuer`, or `merchant`, plus an optional `search`), and `get_month_snapshot` (`month` `YYYY-MM`; runs `report_metrics.calculate_report_snapshot` on that month and the one before it, built from the cached frame, without the report harness). Every spec goes through `validate_and_repair`; an invalid one goes back to the model as an error tool result listing the closest valid values, and the model retries.
* **Limits**: at most `chat_max_tool_rounds` tool rounds per question and `chat_turn_seconds` per question (each model call's timeout is capped at the time left). A model that keeps calling tools, a deadline, an unreachable model, or a Claude budget refusal ends the turn with a plain message (plus the last computed result, if any), never a traceback.
* **Narrative number guard**: every dollar amount and percentage in the answer must match a value in the tool results (`executor.check_narrative_numbers`). On a mismatch the model is asked once to rewrite with the exact figures; if it still does not match, the answer says the figures in the table are authoritative.
* **History**: the last `chat_history_turns` turns are sent as plain text; older turns become a deterministic rolling summary, and the previous spec is included, labeled for follow-ups only, when the message looks like a follow-up (`session.is_follow_up`: a previous analysis exists and the message contains a cue such as "now", "same", "also", "only", "instead", "and by", "break that down", "what about", "that", "it", "those", "compare that", "as a … chart", or "switch to", matched as whole words); otherwise it is left out. The system prompt tells the model to carry over its filters and period only when the user is clearly refining that analysis ("now only Amex", "same but by month", "break that down by store"); a new question starts from the defaults and the answer says which filters apply. Undo, retry, or clearing the chat truncates the session to the turns still shown. The DataFrame is never stored in Gradio state; it lives in the `finance.chat.data` module cache.
* **Ollama thinking**: if a reply hits the token limit with no text while `think` is on, the call is retried once with `think: false`.

The `chat` model role (`model_roles.chat`; when absent it falls back to `model_name` with `ollama_options["chat"]`) can be local or Claude:

* **Local (recommended)**: `{"provider": "ollama", "model": "gemma4:12b", "think": false, "options": {"temperature": 0.15, "num_predict": 1500, "num_ctx": 16384}}` (as in `config.example.json`). It uses Ollama's `/api/chat` with tools.
* **Claude**: for example `{"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "low", "max_tokens": 4000}`. Tools go through the Messages API (`tool_choice` auto; the system prompt and tool definitions are prompt-cached). Every call goes through the budget guard with run id `chat` (the run cap applies to one chat session) and is recorded in the cost ledger; a refused call shows the budget message in the chat.
* **Privacy**: with a Claude chat role, your question, the catalog (category, group, card, and merchant names, the date range, and transaction counts), the conversation, and the aggregated tool results (totals, top groups and merchants, monthly figures) are sent to the Claude API. Transaction rows, descriptions, and dates of individual purchases never leave the machine. Keep the chat role on Ollama if even names and totals must stay local.

Data layer details:

* `data.load_transactions(settings, start=None, client=None)` runs one parameterized query on `gold.fact_transactions` (default: the last 24 months) and caches the frame for `chat_data_cache_seconds`; `data.refresh()` clears it (and the cached catalog and profile). `build_catalog(frame)` lists the date range, issuers, categories with their groups, the top 30 merchants, and the metric definitions. `data.load_chat_data(settings)` returns `ChatData(frame, catalog, profile)`, building the catalog and profile once per load; the app uses it for every chat turn and the example chips.
* Metrics mirror `gold.fact_monthly_spending` exactly (Sale, Return, and Adjustment rows only, so card payments are excluded): `net_spend` (net_real_expense), `gross_spend`, `refunds`, `transaction_count`, `avg_transaction`, and `share_of_total`.
* Merchants are normalized from card descriptors (store numbers, phone numbers, states, `SQ *`/`TST*` prefixes, Amazon variants; payments become `Card Payment` and statement credits `Card Credit`) with optional `merchant_aliases`, and URL/markup characters are stripped.
* Charts use plotly (`..\.venv\Scripts\python.exe -m pip install plotly`); every label is HTML-escaped.

The tests (`Scripts/tests/test_chat_*.py`) use a fixture frame, a fake BigQuery client, and a scripted fake chat client; they never call BigQuery or a model. `test_chat_profile.py` uses two generated fixtures from `chat_test_fixtures.py`: sparse merchants shaped like the mock gold data (531 nearly unique merchants, data ending 2026-08-19, today 2026-10-06) and realistic repeating merchants.

### Report Memory

The `Scripts/finance/memory.py` module manages the report's local SQLite memory database through a `MemoryStore(db_path)` object; `open_memory(memory_schema)` resolves the path, creates or migrates the tables, and returns the store, which the report tools, email workflow, app, and Q&A receive. The older module-level functions (`init_db`, `get_or_create_thread`, ...) still work and delegate to a default store set by `init_db()`. The database is created automatically at `agent/memory/<memory_schema>.sqlite3` (default `agent/memory/memory.sqlite3`) when the weekly report runs. Set the `MEMORY_DB_PATH` environment variable to use a different location; when set, it takes precedence over the `memory_schema`-derived path. The weekly email, the finance app (`Scripts/apps/finance_app.py`)'s reports, and its Q&A all resolve the database the same way (`finance.settings.memory_db_path`). The app's reports use the email recipients (`email_listings`) as their history thread key, so they read the same report history as the weekly email; if memory cannot be opened, the app runs without it.

Memory is organized into three tables:
* `threads` stores stable internal identifiers for report conversations. External keys are hashed before storage.
* `short_term` stores recent prompts and report responses for conversational context. Entries are trimmed to the newest 12 per thread by default.
* `long_facts` stores reusable report facts, such as spending trends or financial goals,
  with a serialized embedding vector and the model name that generated it. Similarity
  search compares vectors only within the same embedding model.

The SQLite database and SQL scratch files are excluded from Git. Delete the local database only if you want to reset the report's memory.

### Setup Instructions

1. **Configure the report settings in `config.json`** (copy `config.example.json`; see Runtime config knobs above). The tested recommendation: author `gemma4:e4b`; validator `gemma4:12b` with `think: false` and `num_predict` `4000`; `num_ctx` `32768`; `max_validation_cycles` `5`; `max_run_seconds` `2700`; the `eval` role (run comparisons) is optional, Claude Sonnet 5.5 or local `gemma4:12b`:
   ```json
   {
     "model_server_url": "http://your-host:11434/api/generate",
     "model_name": "gemma4:e4b",
     "model_roles": {
       "author":    {"provider": "ollama", "model": "gemma4:e4b",
                     "options": {"temperature": 0.3, "num_predict": 12000, "repeat_penalty": 1.1, "num_ctx": 32768}},
       "validator": {"provider": "ollama", "model": "gemma4:12b", "think": false,
                     "options": {"temperature": 0.1, "num_predict": 4000, "num_ctx": 32768}},
       "eval":      {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "max_tokens": 16000}
     },
     "report_mode": "agent",
     "dry_run": true,
     "max_validation_cycles": 5,
     "max_run_seconds": 2700,
     "ollama_timeout": 1800,
     "Monthly_budget": 5000,
     "Fixed_costs": 1500,
     "Savings_goal_target": 1000,
     "project_name": "your-gcp-project-id"
   }
   ```
   `model_server_url` is the shared Ollama server and `model_name` is still used by legacy mode, fact extraction, and the Q&A app. Any role can use `"provider": "anthropic"` or `"ollama"`.

2. **Ensure Ollama is running** locally or on a reachable server:
   * Download from [ollama.com](https://ollama.com) if not installed
   * Pull the author, validator, and embedding models, for example `ollama pull gemma4:e4b`, `ollama pull gemma4:12b`, and `ollama pull nomic-embed-text` (the validator's `"think": false` is set in its `model_roles` entry)
   * Optional: to run a role on Claude instead, see Model providers above

3. **Set up Gmail App Password:** 
   * Generate an app-specific password for your sending email in Google Account settings (2FA required on sender account). Do NOT use your main password!  

4. **Configure recipients** via the `email_listings` array:
   ```json  
   {
     "email_sender": "sender@gmail.com",  // Your Gmail address with app password set
     "Gmail_app_credentials": "your_16_char_password_here", 
     "email_listings": ["subscriber1@gmail.com", "subscriber2@gmail.com"] 
   }
   ```

5. **Run the report with the workspace virtual environment** (it has the Google Cloud packages; the system Python may not). From the project root:
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\reports\weekly_report.py
   ```
   Watch a live run with `Get-Content agent\runs\<run_id>\progress.log -Wait`. Keep `dry_run: true` until reports pass validation consistently.

   The interactive month report and Q&A app uses the same report service (it never sends email or archives runs in `agent/runs`; its Claude calls, if any, are recorded in the cost ledger and count toward the daily cap) and serves on `http://127.0.0.1:7860`:
   ```powershell
   & "..\.venv\Scripts\python.exe" Scripts\apps\finance_app.py
   ```

6. **Run the tests** with the same environment (they mock BigQuery, Ollama, the Anthropic SDK, and SMTP, never make paid API calls, and write only to temporary folders):
   ```powershell
   ..\.venv\Scripts\python.exe -m unittest discover -s Scripts/tests -t Scripts
   ```

7. **Schedule automatic weekly runs:**

   On Windows: Task Scheduler → Create Basic Task (trigger = Weekly) → Action = Start a program. Set **Program** to the venv interpreter (`<path>\Lakehouse\.venv\Scripts\python.exe`), **Arguments** to the quoted full path of `Scripts\reports\weekly_report.py`, and **Start in** to the project root. A task that uses the system Python fails with `ModuleNotFoundError: No module named 'google'`.

   On Linux/macOS: use the virtual environment's interpreter in the crontab entry and run from the project root, for example `0 9 * * 1 cd /path/to/gcp-belperfi-lakehouse-pipeline && ../.venv/bin/python Scripts/reports/weekly_report.py`.

### Report Features
* ✅ Monthly snapshot with spending trends & category analysis  
* ✅ KPI cards showing total net expense, Mom % change, and disposable cash calculation  
* ✅ Top discretionary categories with configured goals or savings-target-based suggested caps
* ✅ Personalized character tips for fun financial habits advice
* ✅ Interactive HTML formatting designed to look beautiful in email clients
