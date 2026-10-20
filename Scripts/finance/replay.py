"""Replay one archived validator call against another model (validator experiments).

load_validator_calls() reads a run's trace.jsonl and returns the first validator ``model_call``
of each validation cycle (format retries are skipped). The traced prompt is split back into
the validator instructions (system) and the cycle's review content (user) on
``VALIDATOR_REPORT_MARKER``; ``current_prompt`` swaps today's agent/prompts/validator.system.md
in as the system part (plus today's guided addendum when the traced run was guided). replay_validator() sends it to the configured validator role, or to a
provider:model given on the command line, with VALIDATOR_RESPONSE_SCHEMA, prints the verdict,
and saves each result as agent/runs/replays/<stamp>_<run_id>_<model>.json.
``Scripts/reports/replay_validator.py`` is the CLI. Nothing here changes the archived run.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

from .harness.prompts import GUIDED_VALIDATOR_HEADER, VALIDATOR_REPORT_MARKER, PromptParts, validator_system
from .harness.protocol import VALIDATOR_RESPONSE_SCHEMA, parse_json_response
from .model_clients import ModelClient, build_model_client, resolve_model_role
from .run_archive import NON_RUN_DIRS, REPLAYS_DIR, update_index
from .budget import BudgetExceededError, configured_ledger_path
from .pricing import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, cost_usd, worst_case_call_usd  # noqa: F401
from .settings import PROJECT_ROOT, ReportSettings, load_config

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "agent" / "runs"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "agent" / "prompts"
VALIDATOR_SYSTEM_FILE = "validator.system.md"
PROVIDERS = ("ollama", "anthropic")
EVIDENCE_PREVIEW_CHARS = 200
RAW_EXCERPT_CHARS = 600
FORMAT_RETRY_TEXT = "\n\nFORMAT RETRY:"


class ReplayError(RuntimeError):
    """The replay could not be set up (unknown run or cycle, bad flags, unusable trace)."""


class ValidatorCall(NamedTuple):
    """One traced validator call: its cycle, the full prompt sent, and the original model."""

    cycle: int
    prompt: str
    provider: str | None
    model: str | None


# ----- reading the archived run -----


def _trace_events(trace_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def resolve_run_dir(run_id: str, runs_root: Path) -> Path:
    name = Path(run_id.strip().rstrip("\\/")).name
    folder = Path(runs_root) / name
    if not name or name in NON_RUN_DIRS or not folder.is_dir():
        raise ReplayError(f"Unknown run ID {run_id!r} in {runs_root}")
    if not (folder / "trace.jsonl").is_file():
        raise ReplayError(f"Run {name} has no trace.jsonl")
    return folder


def load_validator_calls(run_dir: Path) -> tuple[list[ValidatorCall], dict[int, str]]:
    """First validator model_call per cycle (in order) and each cycle's recorded decision.

    Older traces without validation_cycle/attempt fields are numbered in order, skipping
    prompts that carry a format-retry request.
    """
    calls: list[ValidatorCall] = []
    decisions: dict[int, str] = {}
    seen: set[int] = set()
    for event in _trace_events(Path(run_dir) / "trace.jsonl"):
        kind = event.get("event")
        if kind == "validation" and isinstance(event.get("cycle"), int) and event.get("decision"):
            decisions[event["cycle"]] = str(event["decision"])
        if kind != "model_call" or event.get("role") != "validator" or not isinstance(event.get("prompt"), str):
            continue
        attempt = event.get("attempt")
        if attempt not in (None, 1):
            continue
        prompt = event["prompt"]
        cycle = event.get("validation_cycle")
        if not isinstance(cycle, int):
            if FORMAT_RETRY_TEXT in prompt:
                continue
            cycle = len(calls) + 1
        if cycle in seen:
            continue
        seen.add(cycle)
        calls.append(ValidatorCall(cycle, prompt, event.get("provider"), event.get("model")))
    return calls, decisions


def select_call(calls: list[ValidatorCall], cycle: str | int = "last") -> ValidatorCall:
    if not calls:
        raise ReplayError("The run's trace has no validator model_call to replay")
    if cycle in (None, "last"):
        return calls[-1]
    try:
        wanted = int(cycle)
    except (TypeError, ValueError):
        raise ReplayError(f"--cycle must be a number or 'last', got {cycle!r}") from None
    for call in calls:
        if call.cycle == wanted:
            return call
    available = ", ".join(str(call.cycle) for call in calls)
    raise ReplayError(f"Cycle {wanted} not found; the run has validator cycles {available}")


def split_validator_prompt(prompt: str) -> PromptParts:
    """Traced validator prompt -> (validator instructions, this cycle's review content)."""
    index = prompt.find(VALIDATOR_REPORT_MARKER)
    if index < 0:
        raise ReplayError("The traced validator prompt has no 'VALIDATE THIS REPORT:' section; cannot rebuild it")
    return PromptParts(prompt[:index], prompt[index:])


def current_validator_system(prompt_dir: Path = DEFAULT_PROMPT_DIR) -> str:
    """Today's validator instructions, as reporting.generate_finance_report loads them."""
    return (Path(prompt_dir) / VALIDATOR_SYSTEM_FILE).read_text(encoding="utf-8")


def build_replay_prompt(
    run_dir: Path, cycle: str | int = "last", *, current_prompt: bool = False, prompt_dir: Path = DEFAULT_PROMPT_DIR
) -> tuple[ValidatorCall, PromptParts, str | None]:
    """The chosen validator call, its rebuilt prompt parts, and the decision the run recorded for it."""
    calls, decisions = load_validator_calls(run_dir)
    call = select_call(calls, cycle)
    parts = split_validator_prompt(call.prompt)
    if current_prompt:
        guidance = "guided" if f"\n\n{GUIDED_VALIDATOR_HEADER}" in parts.system else "strict"
        parts = PromptParts(validator_system(current_validator_system(prompt_dir), guidance), parts.user)
    return call, parts, decisions.get(call.cycle)


# ----- the replay role -----


def _parse_role_model(value: str) -> tuple[str, str]:
    provider, sep, model = value.partition(":")
    provider = provider.strip().lower()
    if not sep or provider not in PROVIDERS or not model.strip():
        raise ReplayError(f"--role-model must be PROVIDER:MODEL with provider ollama or anthropic, got {value!r}")
    return provider, model.strip()


def replay_role(
    config: dict[str, Any] | ReportSettings,
    role_model: str | None = None,
    *,
    effort: str | None = None,
    think: bool | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(config with model_roles.validator replaced, resolved validator spec) for this replay only.

    Starts from the configured validator role; role_model switches provider/model (keeping the
    role's provider-specific settings when the provider is unchanged), then the flags apply:
    effort (Claude), think (Ollama's flag; for Claude true requests adaptive thinking and false
    is rejected because Claude thinking cannot be disabled), and max_tokens (Claude max_tokens,
    Ollama num_predict).
    """
    settings = ReportSettings.coerce(config)
    base = settings.role("validator")
    entry = {key: value for key, value in base.items() if key != "role" and value is not None}
    if role_model:
        provider, model = _parse_role_model(role_model)
        if provider != base.get("provider"):
            entry = {"timeout": base.get("timeout")}
            if provider == "ollama" and base.get("base_url"):
                entry["base_url"] = base["base_url"]
        entry.update(provider=provider, model=model)
    provider = entry.get("provider")
    if effort is not None:
        if provider != "anthropic":
            raise ReplayError("--effort applies only to anthropic models")
        entry["effort"] = effort
    if think is not None:
        if provider == "anthropic":
            if not think:
                raise ReplayError("Claude thinking cannot be disabled; use --effort low for shallower thinking")
            entry["thinking"] = "adaptive"
        else:
            entry["think"] = think
    if max_tokens is not None:
        if provider == "anthropic":
            entry["max_tokens"] = max_tokens
        else:
            entry["options"] = {**(entry.get("options") or {}), "num_predict": max_tokens}
    raw = copy.deepcopy(dict(settings.raw))
    roles = raw.get("model_roles") if isinstance(raw.get("model_roles"), dict) else {}
    raw["model_roles"] = {**roles, "validator": entry}
    try:
        spec = resolve_model_role(raw, "validator")
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    return raw, spec


# ----- cost and results -----


def estimate_cost(provider: str | None, model: str | None, usage: dict[str, Any] | None) -> float | None:
    """USD for one Anthropic call (finance.pricing: cache reads 0.1x, cache writes 1.25x input)."""
    if provider != "anthropic" or not isinstance(usage, dict):
        return None
    return round(cost_usd(model, usage), 4)


def worst_case_cost(provider: str | None, model: str | None, prompt_chars: int, max_tokens: int | None) -> float | None:
    """The budget guard's worst case for one call: chars/3.5 input tokens at list price plus max_tokens output."""
    if provider != "anthropic" or not max_tokens:
        return None
    return round(worst_case_call_usd(model, prompt_chars, max_tokens), 4)


def parse_validator_payload(text: str) -> dict[str, Any]:
    payload = parse_json_response(text, {"decision", "audit", "findings"})
    if payload.get("decision") not in ("pass", "fail"):
        raise ValueError(f"decision must be 'pass' or 'fail', got {payload.get('decision')!r}")
    if not isinstance(payload.get("findings"), list) or not isinstance(payload.get("audit"), dict):
        raise ValueError("audit must be an object and findings a list")
    return payload


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _output_path(folder: Path, created: datetime, run_id: str, model: str, repeat_index: int, repeat: int) -> Path:
    stamp = created.astimezone().strftime("%Y%m%dT%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    base = f"{stamp}_{run_id}_{safe_model}" + (f"_r{repeat_index}" if repeat > 1 else "")
    path, suffix = folder / f"{base}.json", 1
    while path.exists():
        suffix += 1
        path = folder / f"{base}_{suffix}.json"
    return path


def replay_once(client: ModelClient, parts: PromptParts, timeout: float | None) -> dict[str, Any]:
    """One validator call: status ok/refusal/malformed_json/error plus verdict, usage, and timing."""
    started = time.perf_counter()
    try:
        reply = client.generate(parts.user, system=parts.system, json_schema=VALIDATOR_RESPONSE_SCHEMA, timeout=timeout)
    except BudgetExceededError:
        raise  # a spend cap stops the whole replay (replay_validator turns it into a ReplayError)
    except Exception as exc:  # any provider or network failure is reported, never raised
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 3)}
    seconds = round(time.perf_counter() - started, 3)
    usage = getattr(reply, "usage", None)
    if not isinstance(usage, dict):
        usage = {"input_tokens": getattr(reply, "prompt_eval_count", None), "output_tokens": getattr(reply, "eval_count", None)}
    result: dict[str, Any] = {
        "seconds": seconds,
        "done_reason": getattr(reply, "done_reason", None),
        "usage": usage,
        "metadata": getattr(reply, "metadata", None) or {},
    }
    if result["done_reason"] == "refusal":
        details = result["metadata"].get("stop_details")
        result.update(status="refusal", error="The model declined to answer" + (f" ({details})" if details else ""))
        return result
    try:
        payload = parse_validator_payload(str(reply))
    except (ValueError, json.JSONDecodeError) as exc:
        cut = " after hitting the token limit" if result["done_reason"] == "length" else ""
        result.update(
            status="malformed_json",
            error=f"Validator output was not valid schema JSON{cut}: {exc}",
            raw_excerpt=str(reply)[:RAW_EXCERPT_CHARS],
        )
        return result
    result.update(status="ok", decision=payload["decision"], summary=payload.get("summary"), payload=payload)
    return result


def describe_result(result: dict[str, Any]) -> list[str]:
    """Readable lines for one replay result."""
    lines: list[str] = []
    if result["status"] == "ok":
        payload = result["payload"]
        lines.append(f"Decision: {payload['decision'].upper()}" + (
            f" (run recorded: {result['original_decision']})" if result.get("original_decision") else ""
        ))
        audit = payload.get("audit") or {}
        gates = ", ".join(f"{gate}={(value or {}).get('status', '?')}" for gate, value in audit.items() if isinstance(value, dict))
        lines.append(f"Audit: {gates or '(none)'}")
        errors = [item for item in payload.get("findings") or [] if isinstance(item, dict) and item.get("severity") == "error"]
        lines.append(f"Error findings: {len(errors)}")
        for item in errors:
            lines.append(f"  - [{item.get('category')}] {_clip(item.get('message'), 240)}")
            lines.append(f"    evidence: {_clip(item.get('evidence'), EVIDENCE_PREVIEW_CHARS)}")
    else:
        lines.append(f"FAILED ({result['status']}): {result.get('error')}")
        if result.get("raw_excerpt"):
            lines.append(f"  raw output starts: {_clip(result['raw_excerpt'], EVIDENCE_PREVIEW_CHARS)}")
    usage = result.get("usage") or {}
    if usage:
        lines.append(
            f"Tokens in/out: {usage.get('input_tokens')}/{usage.get('output_tokens')}; "
            f"cache read {usage.get('cache_read_input_tokens', 0) or 0}, cache write {usage.get('cache_creation_input_tokens', 0) or 0}"
        )
    lines.append(f"Seconds: {result.get('seconds')}; done_reason: {result.get('done_reason')}")
    if result.get("estimated_cost_usd") is not None:
        lines.append(f"Estimated cost: ${result['estimated_cost_usd']:.4f}")
    return lines


def replay_validator(
    run_id: str,
    *,
    config: dict[str, Any] | ReportSettings,
    cycle: str | int = "last",
    role_model: str | None = None,
    effort: str | None = None,
    think: bool | None = None,
    max_tokens: int | None = None,
    repeat: int = 1,
    current_prompt: bool = False,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    client_factory: Callable[[dict[str, Any], dict[str, Any]], ModelClient] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    echo: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Replay the chosen validator call repeat times; save and print each result, then a tally.

    Raises ReplayError for setup problems (nothing is called). Call failures are returned as
    results with status refusal/malformed_json/error (and saved), never raised.
    """
    if repeat < 1:
        raise ReplayError("--repeat must be at least 1")
    run_dir = resolve_run_dir(run_id, runs_root)
    call, parts, original_decision = build_replay_prompt(run_dir, cycle, current_prompt=current_prompt, prompt_dir=prompt_dir)
    replay_config, spec = replay_role(config, role_model, effort=effort, think=think, max_tokens=max_tokens)
    if not spec.get("model"):
        raise ReplayError("No validator model configured; pass --role-model PROVIDER:MODEL")
    factory = client_factory or (lambda cfg, _spec: build_model_client(cfg, "validator", run_id="replay"))
    try:
        client = factory(replay_config, spec)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    provider, model = spec["provider"], spec["model"]
    public_spec = ReportSettings.from_config(replay_config).public_role("validator")
    echo(
        f"Replaying {run_dir.name} validator cycle {call.cycle} (originally {call.provider or '?'}:{call.model or '?'}) "
        f"with {provider}:{model}; system {len(parts.system):,} chars"
        f"{' (current validator.system.md)' if current_prompt else ''}, user {len(parts.user):,} chars."
    )
    if provider == "anthropic":
        echo("Note: the report, snapshot, and evidence in this prompt are sent to the Claude API.")
    folder = Path(runs_root) / REPLAYS_DIR
    results: list[dict[str, Any]] = []
    for index in range(1, repeat + 1):
        if repeat > 1:
            echo(f"\n--- replay {index}/{repeat} ---")
        try:
            outcome = replay_once(client, parts, spec.get("timeout"))
        except BudgetExceededError as exc:
            done = f" after {len(results)} saved replay(s)" if results else ""
            raise ReplayError(f"{exc}. Replay stopped{done}; no further Claude calls were made.") from exc
        created = now()
        result = {
            "created_at": created.isoformat(),
            "run_id": run_dir.name,
            "cycle": call.cycle,
            "original_provider": call.provider,
            "original_model": call.model,
            "original_decision": original_decision,
            "provider": provider,
            "model": model,
            "role": public_spec,
            "current_prompt": current_prompt,
            "system_chars": len(parts.system),
            "user_chars": len(parts.user),
            "repeat_index": index,
            "repeat": repeat,
            **outcome,
        }
        result["estimated_cost_usd"] = estimate_cost(provider, model, result.get("usage"))
        for line in describe_result(result):
            echo(line)
        folder.mkdir(parents=True, exist_ok=True)
        path = _output_path(folder, created, run_dir.name, model, index, repeat)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        result["saved_to"] = str(path)
        echo(f"Saved: {path}")
        results.append(result)
    echo("\n" + tally_line(results))
    try:
        update_index(Path(runs_root), ledger_path=configured_ledger_path(config))
    except OSError as exc:  # the index is a convenience; the results are already saved
        echo(f"Warning: could not refresh the run index ({exc})")
    return results


def tally_line(results: list[dict[str, Any]]) -> str:
    passed = sum(1 for item in results if item.get("decision") == "pass")
    failed = sum(1 for item in results if item.get("decision") == "fail")
    errors = len(results) - passed - failed
    costs = [item["estimated_cost_usd"] for item in results if item.get("estimated_cost_usd") is not None]
    line = f"Tally over {len(results)} replay(s): pass {passed}, fail {failed}, errors {errors}."
    if costs:
        line += f" Estimated total cost ${sum(costs):.4f}."
    return line


# ----- CLI -----


def _bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _resolve_config_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_validator.py",
        description="Replay one archived validator call (default: the last cycle) against the configured validator "
        "or another PROVIDER:MODEL, to compare verdicts, consistency, tokens, and cost.",
    )
    parser.add_argument("run_id", metavar="RUN_ID", help="run folder name under agent/runs")
    parser.add_argument("--cycle", default="last", help="validation cycle number, or 'last' (default)")
    parser.add_argument("--role-model", metavar="PROVIDER:MODEL", help="e.g. anthropic:claude-haiku-4-5 or ollama:gemma4:12b (default: model_roles.validator)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="Claude effort for this replay")
    parser.add_argument("--think", type=_bool_flag, metavar="true|false", help="Ollama think flag; for Claude, true requests adaptive thinking")
    parser.add_argument("--max-tokens", type=int, metavar="N", help="Claude max_tokens (up to 64000) or Ollama num_predict")
    parser.add_argument("--repeat", type=int, default=1, metavar="K", help="run the same call K times and print a pass/fail tally")
    parser.add_argument("--dry-run-prompt", action="store_true", help="print prompt sizes and the worst-case cost without calling a model")
    parser.add_argument("--current-prompt", action="store_true", help="use today's agent/prompts/validator.system.md as the system prompt")
    parser.add_argument("--config", help="config file (default: config.json in the project root)")
    parser.add_argument("--runs-dir", help=argparse.SUPPRESS)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    config: dict[str, Any] | ReportSettings | None = None,
    runs_root: Path | None = None,
    prompt_dir: Path | None = None,
    client_factory: Callable[[dict[str, Any], dict[str, Any]], ModelClient] | None = None,
) -> int:
    """CLI entry point: 0 when every replay returned a verdict, 1 when any call failed, 2 for setup errors."""
    args = build_parser().parse_args(argv)
    root = Path(args.runs_dir) if args.runs_dir else (runs_root or DEFAULT_RUNS_ROOT)
    prompts = prompt_dir or DEFAULT_PROMPT_DIR
    try:
        if config is None:
            config = load_config(_resolve_config_path(args.config))
        flags = dict(effort=args.effort, think=args.think, max_tokens=args.max_tokens)
        if args.dry_run_prompt:
            run_dir = resolve_run_dir(args.run_id, root)
            call, parts, original = build_replay_prompt(run_dir, args.cycle, current_prompt=args.current_prompt, prompt_dir=prompts)
            _, spec = replay_role(config, args.role_model, **flags)
            total = len(parts.system) + len(parts.user)
            print(
                f"Run {run_dir.name}, validator cycle {call.cycle} (originally {call.provider or '?'}:{call.model or '?'}, "
                f"recorded decision {original or 'unknown'})."
            )
            print(
                f"System: {len(parts.system):,} chars{' (current validator.system.md)' if args.current_prompt else ''}; "
                f"user: {len(parts.user):,} chars; about {total // 4:,} input tokens."
            )
            line = f"Replay model: {spec['provider']}:{spec['model']}."
            cost = worst_case_cost(spec["provider"], spec["model"], total, spec.get("max_tokens"))
            if cost is not None:
                line += (
                    f" Worst-case cost ${cost:.4f} per call (prompt at list price plus max_tokens {spec['max_tokens']} output)"
                    + (f", ${cost * args.repeat:.4f} for {args.repeat} calls." if args.repeat > 1 else ".")
                )
            print(line)
            print("No model call was made.")
            return 0
        results = replay_validator(
            args.run_id,
            config=config,
            cycle=args.cycle,
            role_model=args.role_model,
            repeat=args.repeat,
            current_prompt=args.current_prompt,
            runs_root=root,
            prompt_dir=prompts,
            client_factory=client_factory,
            **flags,
        )
    except ReplayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    failed = [item for item in results if item.get("status") != "ok"]
    if failed:
        print(f"Error: {len(failed)} of {len(results)} replay call(s) failed ({', '.join(sorted({item['status'] for item in failed}))}).", file=sys.stderr)
        return 1
    return 0
