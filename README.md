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
│   │   ├── agent_harness.py       # Bounded author/validator loop, author protocol, HTML gate, retries
│   │   ├── bigquery_tools.py      # Read-only tools for gold-layer spending and transaction evidence
│   │   ├── memory.py              # SQLite-backed short-term and long-term memory store
│   │   ├── memory_search.py       # Hybrid (embedding + BM25) long-fact retrieval
│   │   ├── ollama.py              # Ollama text-generation and embedding clients
│   │   ├── qa.py                  # Finance Q&A over the selected report and long-term facts
│   │   ├── report_metrics.py      # Deterministic snapshot: totals, MoM, disposable cash, goal caps, bar widths
│   │   ├── report_models.py       # Typed snapshot, validation, and finding models
│   │   ├── report_requirements.py # Month-specific report requirements checklist built from the snapshot
│   │   ├── reporting.py           # Builds the snapshot, tools, and harness for one report
│   │   ├── run_archive.py         # Run archive, live progress files, and the run index
│   │   └── settings.py            # Config loading and money parsing
│   ├── agent_harness.py, bigquery_tools.py, memory.py,
│   │   report_metrics.py, report_models.py, run_archive.py
│   │                              # Compatibility shims that re-export Scripts/finance/*
│   ├── email_weekly_report.py     # Orchestrates the weekly run: legacy or agent mode, archive, email
│   ├── Finance Summary and QA.py  # Gradio UI for month reports and finance Q&A
│   ├── model benchmark.py         # Quick Ollama model speed benchmark
│   ├── upload_csv.py              # Uploads issuer-specific CSV files to GCS and archives them
│   ├── run_all_issuers.py         # Runs the uploader for every issuer in config.json
│   ├── generate_mock_cd.py        # Generates mock Chase and Amex CSV data
│   ├── test_agent_harness.py      # Harness protocol, parsing, sanitization, retries, and feedback tests
│   ├── test_finance_services.py   # Shared reporting/tool wiring tests
│   ├── test_memory.py             # SQLite memory and retrieval tests
│   ├── test_email_weekly_report.py # Email workflow, archiving, and memory persistence tests
│   ├── test_report_metrics.py     # Snapshot goal-cap and bar-width math tests
│   ├── test_report_requirements.py # Requirements checklist content and prompt wiring tests
│   └── test_run_progress.py       # Live progress files and run-index tests
├── agent/
│   ├── prompts/
│   │   ├── personality.system.md     # Author persona and grounding instructions
│   │   ├── ui.system.md              # HTML email and presentation constraints
│   │   ├── financial-settings.md     # Financial policy and configuration context
│   │   └── validator.system.md       # Validator-only output and review contract
│   ├── memory/                       # Local SQLite memory files (ignored by Git)
│   └── runs/                         # One folder per run plus index.html (ignored by Git):
│                                     #   run.json, metrics.json, trace.jsonl, progress.json,
│                                     #   progress.log, final_report.html, final_report.txt
├── definitions/
│   ├── bronze/                # Raw landing dependency declarations
│   ├── silver/                # Cleaned SQLX models (e.g., chase_transactions.sqlx)
│   └── gold/                  # Curated analytical views
├── config.json                # Runtime configuration (secrets; ignored by Git)
├── config.example.json        # Example template for creating config.json
├── workflow_settings.yaml     # Dataform project configuration
├── package.json               # Dataform core dependency (`npm run compile` runs `dataform compile`)
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
3. Use the workspace virtual environment one level above the project root (`..\.venv`), which already has the required packages. To install them into a fresh environment:
   ```powershell
   ..\.venv\Scripts\python.exe -m pip install pandas google-cloud-storage
   ```
4. Upload files for one issuer (run from the project root; the issuer key comes from `issuers` in config.json):
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\upload_csv.py amex
   ```
   You can also specify the config file explicitly (a relative path is resolved from the project root):
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\upload_csv.py --config config.json --issuer amex
   ```
5. Upload files for every issuer defined in config.json:
   ```powershell
   ..\.venv\Scripts\python.exe Scripts\run_all_issuers.py
   ```

This workflow is useful for automating local CSV ingestion into GCS while keeping a backup copy of each processed file.

## 📧 Weekly Financial Report Automation

### Agent harness mode

The report can run in two modes. `report_mode: "legacy"` preserves the original single-prompt flow. `report_mode: "agent"` uses bounded tools, an author model, and an independent validator model. The author keeps the consistent Mogumogu layout and section order defined in `agent/prompts/ui.system.md`. The validator checks current financial values, goal math, evidence, history labels, policy, required content, design, and HTML safety. Failed validation feeds structured feedback back to the author for a limited number of revisions. Email is sent only after validation passes.

How the agent run works:
* **Deterministic snapshot.** `report_metrics.py` computes every number before any model runs: monthly totals, MoM change, disposable cash, and the goal and chart fields (savings shortfall, eligible discretionary total, suggested caps, percent of cap, bar widths, MoM direction, and the top three discretionary categories). The author copies these values; the validator checks the report against them and spot-checks the arithmetic.
* **Requirements checklist.** `report_requirements.py` turns the snapshot into a short, month-specific checklist (about 10–20 lines): the exact required section headings in order, every Monthly Snapshot bar amount (refunds labeled), the four Sushi Stack values with month labels, the goal-method disclosures (for example that discretionary cuts alone cannot reach the target, with the remaining gap, or that no numeric savings target is configured), and each Top 3 row's spend, cap label, and percent of cap. `reporting.py` passes it to the harness, which adds it once as a "REPORT REQUIREMENTS CHECKLIST" block after the snapshot in the author base prompt (so revisions inherit it) and in every validator prompt, where the validator must check each item and report failures as error findings. It is content guidance only: no Python code inspects the report HTML against it. `ReportHarness.run` still works without it (`report_requirements=None`).
* **Author protocol.** To request evidence, the author replies with a small JSON object, `{"tool_calls": [...]}`. When the report is ready, it replies with the raw HTML between `<<<REPORT_HTML>>>` and `<<<END_REPORT_HTML>>>`, so the HTML never has to be escaped inside JSON. The plain-text email part is derived from the HTML. The validator still replies with schema-constrained JSON.
* **Recovery instead of aborting.** Malformed or truncated author output gets a format retry. Rejected HTML (unsafe markup, oversize, or a missing history label), over-budget or unknown tool calls, and tool errors go back to the author as feedback. A model call that times out, cannot connect, returns HTTP 5xx, or comes back empty is retried once. If it fails again, the run ends with a fail decision: the deterministic fallback report if the author never produced one, otherwise the last sanitized report. Running out of `max_agent_rounds` ends the run the same way (an `agent_rounds_exhausted` trace event; a revision keeps the last sanitized report and the last review's findings). Nothing is sent on a fail.
* **Focused revisions.** Each revision gets the previous HTML and only the top three errors (arithmetic first, then completeness, evidence, history, policy, html, usability, visual design), with the exact snapshot values they concern. The author is told to fix only those with minimal edits.
* **Evidence-grounded findings and rebuttals.** The validator prompt requires every error finding to quote the wrong report text verbatim and cite the conflicting snapshot field, to search the whole report for paraphrases before calling something missing, and not to repeat a prior finding without new evidence. During a revision the author may dispute a finding the previous report already satisfied by adding, after `<<<END_REPORT_HTML>>>`, a block `<<<REBUTTALS>>>[{"finding": "<short finding message>", "quote": "<verbatim report text>"}]<<<END_REBUTTALS>>>` (optionally wrapped in `<!-- -->`). The block is always stripped from the report HTML, malformed blocks are ignored (`author_rebuttals_ignored` trace event), and valid ones are traced (`author_rebuttals`) and passed to the next validator prompt as "AUTHOR REBUTTALS (verify each quote against the report)". The harness does not check quotes or filter findings; the validator decides. Genuine findings must still be fixed.
* **Stall detection.** When a cycle's error findings match the previous cycle's (normalized category plus message prefix), a `validation_stalled` trace event and progress line are recorded. Each `validation_progress` event carries `repeated_error_count`. This is informational and does not change the decision.

Set `dry_run` to `true` to execute the BigQuery, memory, author, validator, and rendering flow without sending email. Each run is saved under `agent/runs/<run_id>/` with its HTML report, plain text, trace, manifest, and metrics. The run directory is created when the run starts so it can be watched live: `trace.jsonl` is appended as events happen (and rewritten once with the complete trace at the end), `progress.json` holds the current status, phase, round, validation cycle, tool-call count, latest error count, repeated-error count, stalled-cycle count, and author-rebuttal count, and `progress.log` has one timestamped line per model call (with elapsed seconds), tool call, and validation result. The same lines print to the console. Follow a run with `Get-Content agent\runs\<run_id>\progress.log -Wait`. A run that raises is renamed with a `_failed` suffix. Failures after the agent harness starts are also archived with the available trace and traceback before control returns to the email workflow. The JSONL trace records model prompts/raw responses (with elapsed seconds, Ollama's `done_reason`, and prompt/output token counts), tool calls and errors, model-call failures and retries, recovery actions, and each validation cycle's outcome. A `prompt_eval_count` close to `num_ctx` means the prompt was likely truncated; a `done_reason` of `length` means the output hit `num_predict`. For revisions, the validator receives the prior review as comparison context and must independently reassess the new report. Metrics summarize model calls, rounds, tools, retries, report size, and initial-to-final validator-reported error counts. These progress counts help spot resolved or recurring findings; they are based on model findings, not a deterministic correctness score, and do not train or permanently change the model. Traces can contain financial details and transaction evidence, so keep the run archive private and do not commit or share it indiscriminately. Open `agent/runs/index.html` in a browser to view reports and compare objective metrics across models and runs. The index is regenerated automatically after each run.

#### Runtime config knobs

The live project configuration is read from `config.json` (not `config.example.json`). The agent mode adds a few important runtime controls that keep the report process bounded, inspectable, and safe:

* `report_mode`: set to `"agent"` to use the structured author/validator flow. `"legacy"` (the default when the key is omitted) keeps the original one-shot report generation flow.
* `dry_run`: when `true`, the project runs the full BigQuery + prompt + validation + archive flow without sending email. This is recommended while tuning the model or validating report quality. Default `false`. Only agent mode honors it; legacy mode currently sends regardless.
* `model_server_url`: Ollama server, either the base URL (`http://your-host:11434`) or the full endpoint (`http://your-host:11434/api/generate`); the generate and embed endpoints are derived from it.
* `model_name`: the author model.
* `validator_model_name`: the model used to validate the authored report. A different model family from the author is recommended (for example author `gemma4:12b`, validator `qwen3.5:9b`): a same-model pair tends to share blind spots and approve its own weak output. Defaults to `model_name` when omitted.
* `ollama_options`: optional per-role Ollama generation settings, sent as the request `options`:
  ```json
  "ollama_options": {
    "author": {"temperature": 0.3, "num_predict": 12000, "repeat_penalty": 1.1, "num_ctx": 32768},
    "validator": {"temperature": 0.1, "num_predict": 4000, "num_ctx": 32768}
  }
  ```
  A role may also set `"think": true|false`, sent as Ollama's top-level `think` flag for reasoning models. Hidden thinking counts toward `num_predict`: a reasoning validator such as `qwen3.5:9b` can spend its whole budget thinking and return no JSON, so the example sets `"think": false` for the validator (or raise its `num_predict` substantially if you enable thinking). Unset keys are not sent, so the server's defaults apply. `num_predict` caps output so a model stuck in a loop stops quickly instead of running until `ollama_timeout`. About `32768` for `num_ctx` covers the largest validator prompts; a much larger context on a 16 GB server wastes memory and can push the model partly onto the CPU.
* `max_agent_rounds`: how many author rounds (tool requests and format retries) are allowed per phase (initial report, then each revision). Running out ends the run with a fail decision (the fallback report in the initial phase, the last sanitized report in a revision), which is archived normally and never sent.
* `max_tool_calls`: total number of tool calls permitted across the run.
* `max_calls_per_tool`: per-tool call ceiling to prevent one tool from being overused.
* `max_validation_cycles`: number of revision loops after the first review before the report is rejected (total reviews = this value + 1).
* `max_report_bytes`: maximum generated HTML size allowed before the report is rejected as too large.
* `days_back`: legacy mode only; how many days of BigQuery history the summary query looks back over. Agent mode reads the latest month and its predecessor.
* `ollama_stream`: legacy mode only; keep `false` so the model returns a complete response.
* `ollama_timeout`: the request timeout for Ollama calls in seconds (default `600` when omitted). Increase this for larger or slower models that need more time to reason. A typical starting point is `1800` (30 minutes), with larger models sometimes needing more.
* `embedding_model_name`: Ollama model for long-term fact and query embeddings. Default: `nomic-embed-text`; pull it on the Ollama server before using semantic memory.
* `memory_schema`: name of the SQLite memory file under `agent/memory/` (see Report Memory).
* `Monthly_budget`, `Fixed_costs`, `Savings_goal_target`: monthly deposit, fixed costs, and the single numeric monthly savings target used by the snapshot. `Vehicle_Info` is legacy-mode context.
* `category_goals`: optional `{"Category": amount}` monthly goals. A configured goal replaces the suggested cap for that category and is labeled as configured.

Optional keys not in `config.example.json`: `fact_model_name` (model for extracting long-term facts after a sent report; defaults to `model_name`), and `qa_report_context_chunks` / `qa_long_fact_limit` (context sizes for the Finance Summary and QA app; default 5 each).

The `max_*` and `days_back` values in `config.example.json` are also the code defaults used when a key is omitted. These values are intentionally conservative for a demo or controlled deployment: they keep the model bounded, prevent runaway tool use, and make each run easy to review in the generated archive.

The agent's transaction-detail tool reads only from the gold `fact_transactions` table and is bounded by month, category, and row count. `get_semantic_memory` ranks a focused agent-written query with embedding cosine similarity and BM25 keyword relevance, then combines rankings with reciprocal-rank fusion and deduplicates facts. Semantic matching can catch paraphrases while BM25 matches exact terms and accounts for term frequency and document length. BM25 remains available if Ollama embeddings are unavailable. Both rankers scan the existing SQLite fact set directly; no separate search service or index is needed. Episodic memory and report history remain separate read-only tools. Previous reports and semantic facts are historical and cannot replace current gold-layer values. Existing SQLite databases are migrated in place; missing vectors are embedded lazily during semantic search.

The `email_weekly_report.py` script generates an intelligent weekly financial insights email that:
* Queries your **BigQuery Gold Layer** (`fact_monthly_spending`) to extract spending trends, category analysis, and key metrics
* Processes data through a local **Ollama LLM** (configurable via `model_server_url` & `model_name` in config) to generate contextual insights with playful Japanese food puns from your assistant Mogumogu-chan 🍣🍤
* Uses local SQLite memory and Ollama embeddings to retrieve relevant long-term facts on demand, alongside recent report history
* Saves each completed report and extracts reusable financial facts for future reports
* Outputs an attractive HTML report styled for email readability

### Report Memory

The `Scripts/finance/memory.py` module (re-exported by `Scripts/memory.py`) manages the report's local SQLite memory database. The database is created automatically at `agent/memory/<memory_schema>.sqlite3` (default `agent/memory/memory.sqlite3`) when the weekly report runs. Set the `MEMORY_DB_PATH` environment variable to use a different location; when set, it takes precedence over the `memory_schema`-derived path. The weekly email, the Finance Summary and QA app's reports, and its Q&A all resolve the database the same way (`finance.settings.memory_db_path`). The app's reports use the email recipients (`email_listings`) as their history thread key, so they read the same report history as the weekly email; if memory cannot be opened, the app runs without it.

Memory is organized into three tables:
* `threads` stores stable internal identifiers for report conversations. External keys are hashed before storage.
* `short_term` stores recent prompts and report responses for conversational context. Entries are trimmed to the newest 12 per thread by default.
* `long_facts` stores reusable report facts, such as spending trends or financial goals,
  with a serialized embedding vector and the model name that generated it. Similarity
  search compares vectors only within the same embedding model.

The SQLite database and SQL scratch files are excluded from Git. Delete the local database only if you want to reset the report's memory.

### Setup Instructions

1. **Configure the report settings in `config.json`** (copy `config.example.json`; see Runtime config knobs above):
   ```json
   {
     "model_server_url": "http://your-host:11434/api/generate",
     "model_name": "gemma4:12b",
     "validator_model_name": "qwen3.5:9b",
     "report_mode": "agent",
     "dry_run": true,
     "Monthly_budget": 5000,
     "Fixed_costs": 1500,
     "Savings_goal_target": 1000,
     "project_name": "your-gcp-project-id"
   }
   ```

2. **Ensure Ollama is running** locally or on a reachable server:
   * Download from [ollama.com](https://ollama.com) if not installed
   * Pull the author, validator, and embedding models, for example `ollama pull gemma4:12b`, `ollama pull qwen3.5:9b`, and `ollama pull nomic-embed-text`

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
   ..\.venv\Scripts\python.exe Scripts\email_weekly_report.py
   ```
   Watch a live run with `Get-Content agent\runs\<run_id>\progress.log -Wait`. Keep `dry_run: true` until reports pass validation consistently.

   The interactive month report and Q&A app uses the same report service (it never sends email or writes to `agent/runs`) and serves on `http://127.0.0.1:7860`:
   ```powershell
   & "..\.venv\Scripts\python.exe" "Scripts\Finance Summary and QA.py"
   ```

6. **Run the tests** with the same environment (they mock BigQuery, Ollama, and SMTP and write only to temporary folders):
   ```powershell
   ..\.venv\Scripts\python.exe -m unittest discover -s Scripts -p "test_*.py"
   ```

7. **Schedule automatic weekly runs:**

   On Windows: Task Scheduler → Create Basic Task (trigger = Weekly) → Action = Start a program. Set **Program** to the venv interpreter (`<path>\Lakehouse\.venv\Scripts\python.exe`), **Arguments** to the quoted full path of `Scripts\email_weekly_report.py`, and **Start in** to the project root. A task that uses the system Python fails with `ModuleNotFoundError: No module named 'google'`.

   On Linux/macOS: use the virtual environment's interpreter in the crontab entry and run from the project root, for example `0 9 * * 1 cd /path/to/gcp-belperfi-lakehouse-pipeline && ../.venv/bin/python Scripts/email_weekly_report.py`.

### Report Features
* ✅ Monthly snapshot with spending trends & category analysis  
* ✅ KPI cards showing total net expense, Mom % change, and disposable cash calculation  
* ✅ Top discretionary categories with configured goals or savings-target-based suggested caps
* ✅ Personalized character tips for fun financial habits advice
* ✅ Interactive HTML formatting designed to look beautiful in email clients
