# Project Instructions for Coding Agents

This is the project root for the Belperfi personal-finance lakehouse and report pipeline. Read this file and `README.md` before making project-wide changes. For report behavior, read the canonical prompts under `agent/prompts/`: `personality.system.md`, `ui.system.md`, `financial-settings.md`, and `validator.system.md`.

## Architecture and sources of truth

- The report pipeline is primarily in `Scripts/finance/`; `Scripts/email_weekly_report.py` orchestrates the email workflow and shares report generation through `finance.reporting`.
- BigQuery gold-layer data and the deterministic `ReportSnapshot` (`Scripts/finance/report_metrics.py`) are authoritative for current amounts, category groups, monthly totals, MoM values, disposable cash, savings shortfall, top-three discretionary categories, suggested/configured caps, percent of cap, and snapshot bar widths. Change goal or chart math there (with tests), and keep `ui.system.md` and `validator.system.md` consistent with it; the author copies these fields and the validator verifies and spot-checks them.
- Memory, prior reports, and run archives are historical context only. Never reuse their figures as current facts.
- The sole configured monthly savings target is numeric `Savings_goal_target` in both `config.json` and `config.example.json`. Do not reintroduce a competing `Savings_goal` setting. Do not reveal or copy runtime secrets from `config.json` into code, prompts, logs, or examples.
- `agent/prompts/*.md` are runtime instructions consumed by the report author and validator. Update them when report behavior or policy changes; do not assume external coding agents automatically load them.

## Report requirements

- Preserve the required report order and content: friendly Mogumogu greeting; Monthly Snapshot with all supplied main category groups and horizontal category bars; Spending Trends & Analysis; Mogumogu's Financial Health Check; four-metric Sushi Stack; Top 3 Discretionary Categories table; and actionable weekly tips with an encouraging sign-off.
- Keep the established cheerful, Japanese-food-inspired Mogumogu personality, including a few appropriate food emoji (not excessive decoration).
- Design for email clients using Material 3-inspired hierarchy and spacing: visible pale blush-pink canvas, light report surface, bright rose/hot-pink/coral accents, dark readable text, restrained surfaces, and consistent spacing. Do not let purple/plum replace pink.
- Snapshot bars use one rose/pink family, one scale across all groups (bar width proportional to spend relative to the largest positive category), and right-aligned amounts sharing a common edge. Label refunds/negative amounts without letting them distort the positive bar scale.
- Keep chart values, table numbers, labels, and trend semantics consistent with the authoritative snapshot. A lower expense is generally favorable and a higher one generally unfavorable; label the direction as well as using color.
- Top-three categories must come from supplied Wants/Discretionary groups, not Needs or fixed/uncontrollable groups. Use configured category goals if present; otherwise follow the savings-target and spending-trend recommendation method in `ui.system.md`. Label advisor-suggested caps distinctly from configured goals; never invent targets.
- Avoid transaction-level or merchant-specific claims (for example, a particular habit or merchant) unless supported by `get_transaction_detail` evidence. Tool calls are evidence-driven, not a quota: do not make redundant calls. If a tool fails, its error should be returned to the author with prior context so it can correct/retry or skip it.
- Treat the validator as a skeptical reviewer. It must independently reason through the actual HTML/CSS, math, category eligibility, and goal calculations; missing or unverifiable hard requirements should fail. Use `visual_design` findings for Material 3/design defects and `usability` for readability/accessibility defects. Keep this behavior in the canonical validator prompt, not deterministic Python checks unless specifically requested.

## Change and test workflow

- Inspect the relevant implementation and prompt files before editing; keep changes focused and avoid modifying archived runs unless explicitly requested.
- Do not send email or alter production finance data while validating. Respect `dry_run` in runtime configuration.
- Run the Python unit tests from this project directory with the workspace virtual environment: `python -m unittest discover -s Scripts -p "test_*.py"`.
- Never expose credentials, account identifiers, or transaction-level personal details in responses or committed examples.
