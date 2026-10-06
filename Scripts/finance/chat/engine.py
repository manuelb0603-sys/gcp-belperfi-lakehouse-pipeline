"""The finance chat turn loop: model <-> tools, the narrative number guard, and UI events.

``run_turn`` sends the question (plus the session context) to the chat model with the three tools,
runs requested tools deterministically for at most ``chat_max_tool_rounds`` rounds within
``chat_turn_seconds``, checks every number in the answer against the tool summaries, and yields
UI events in order:

1. ``explanation`` (status "pending" while working, then "done"): how the answer was computed,
   with the plain-language caption, repair notes, and the spec;
2. ``text``: the narrative;
3. ``figure``: the plotly figure of the last analysis (when it has a chart);
4. ``table``: its aggregated table (at most ``MAX_TABLE_ROWS`` rows).

Budget refusals, an unreachable model, and the turn deadline become friendly text, never a traceback.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterator, Mapping

import pandas as pd

from ..budget import BudgetExceededError
from ..model_clients import ChatReply, EmptyModelResponse, is_anthropic_transient, tool_message
from ..settings import ReportSettings
from . import prompts
from .executor import check_narrative_numbers
from .session import ChatSession, Turn
from .tools import ToolContext, ToolOutcome, dispatch, tool_definitions

logger = logging.getLogger(__name__)

MAX_TABLE_ROWS = 25


class TurnDeadlineExceeded(TimeoutError):
    """The per-turn time limit (chat_turn_seconds) ran out."""


@dataclass
class ChatEvent:
    kind: str  # "explanation", "text", "figure", or "table"
    content: Any
    status: str | None = None  # explanation only: "pending" or "done"
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- helpers -----


def _spec_dict(outcome: ToolOutcome) -> dict[str, Any]:
    return outcome.spec.model_dump(mode="json", exclude_defaults=True) if outcome.spec is not None else {}


def explanation_text(outcomes: list[ToolOutcome]) -> str:
    """Markdown for the "How I computed this" message."""
    if not outcomes:
        return prompts.NO_TOOLS_TEXT
    parts = []
    for outcome in outcomes:
        if outcome.name == "analyze_spending" and outcome.ok and outcome.result is not None:
            lines = [outcome.result.caption]
            lines += [f"- {note}" for note in outcome.notes[:6]]
            lines.append(f"Spec: `{json.dumps(_spec_dict(outcome), ensure_ascii=False, sort_keys=True)}`")
            parts.append("\n".join(lines))
        elif outcome.name == "get_month_snapshot" and outcome.ok:
            parts.append(
                f"Monthly snapshot for {outcome.payload.get('month')} compared with {outcome.payload.get('previous_month') or 'no prior month'} "
                "(same calculation as the monthly report; card payments excluded)."
            )
        elif outcome.name == "list_values" and outcome.ok:
            search = outcome.payload.get("search")
            parts.append(f"Looked up {outcome.payload.get('field')} names" + (f" containing '{search}'." if search else "."))
        else:
            parts.append(f"A tool request was rejected and corrected: {str(outcome.payload.get('error', ''))[:240]}")
    return "\n\n".join(parts)


def _pending_text(outcomes: list[ToolOutcome]) -> str:
    captions = [o.result.caption.split(".")[0] for o in outcomes if o.ok and o.result is not None]
    return prompts.PENDING_RUNNING_PREFIX + captions[-1] if captions else prompts.PENDING_TEXT


def _guard_data(outcomes: list[ToolOutcome]) -> dict[str, Any]:
    return {"results": [outcome.payload for outcome in outcomes if outcome.ok]}


def _fallback_narrative(outcomes: list[ToolOutcome]) -> str:
    captions = [o.result.caption for o in outcomes if o.ok and o.result is not None]
    return f"{prompts.FALLBACK_INTRO} {captions[-1]}" if captions else ""


def _is_model_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, EmptyModelResponse)) or is_anthropic_transient(exc):
        return True
    try:
        import requests
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status is None or status >= 500 or status == 404
    return False


def _short_reason(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:160]


def _call_model(
    client: Any,
    messages: list[dict[str, Any]],
    system: str,
    tools: list[dict[str, Any]],
    deadline: float,
    clock: Callable[[], float],
) -> ChatReply:
    """One model call within the deadline; retries once with think off when thinking used every token."""

    def call(**extra: Any) -> ChatReply:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TurnDeadlineExceeded("chat turn deadline reached")
        default = getattr(client, "default_timeout", None)
        timeout = min(remaining, default) if isinstance(default, (int, float)) else remaining
        return client.chat(messages, system=system, tools=tools, timeout=timeout, **extra)

    reply = call()
    if (
        reply.done_reason == "length" and not reply.text.strip() and not reply.tool_calls
        and getattr(client, "think", None)
    ):
        logger.info("Chat model spent its token budget thinking; retrying once with think off")
        reply = call(think=False)
        reply.metadata["think_retry"] = True
    return reply


# ----- the turn loop -----


def run_turn(
    session: ChatSession,
    user_text: str,
    *,
    client: Any,
    df: pd.DataFrame,
    catalog: Mapping[str, Any],
    settings: ReportSettings | Mapping[str, Any],
    today: date | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[ChatEvent]:
    """Answer one user message; yields ChatEvents and records the turn in ``session``."""
    settings = ReportSettings.coerce(settings)
    today = today or date.today()
    question = (user_text or "").strip()
    turn = Turn(question)
    if not question:
        turn.assistant_text = prompts.EMPTY_QUESTION
        session.add_turn(turn)
        yield ChatEvent("text", prompts.EMPTY_QUESTION)
        return

    deadline = clock() + settings.chat_turn_seconds
    max_rounds = settings.chat_max_tool_rounds
    ctx = ToolContext(df, catalog, settings, today, session.focus_month)
    system = prompts.build_system_prompt(catalog, today=today, focus_month=session.focus_month)
    tools = tool_definitions()
    messages = session.history_messages() + [{
        "role": "user",
        "content": prompts.turn_prompt(question, summary=session.rolling_summary(), last_spec=session.last_spec),
    }]
    outcomes: list[ToolOutcome] = []
    text = ""
    yield ChatEvent("explanation", prompts.PENDING_TEXT, status="pending")

    try:
        rounds = 0
        while True:
            reply = _call_model(client, messages, system, tools, deadline, clock)
            if reply.tool_calls:
                if rounds >= max_rounds:
                    text = _fallback_narrative(outcomes) or prompts.TOOL_LIMIT_MESSAGE
                    break
                rounds += 1
                messages.append(reply.assistant_message())
                for call in reply.tool_calls:
                    outcome = dispatch(call.get("name", ""), call.get("arguments"), ctx)
                    outcomes.append(outcome)
                    messages.append(tool_message(call.get("id"), call.get("name", ""), outcome.content(), is_error=not outcome.ok))
                if rounds >= max_rounds:
                    messages.append({"role": "user", "content": prompts.FINAL_ROUND_NOTE})
                yield ChatEvent("explanation", _pending_text(outcomes), status="pending")
                continue
            text = reply.text.strip()
            if not text:
                fallback = _fallback_narrative(outcomes)
                if reply.done_reason == "refusal":
                    text = prompts.REFUSAL_MESSAGE
                elif reply.done_reason == "length":
                    text = fallback or prompts.LENGTH_MESSAGE
                else:
                    text = fallback or prompts.EMPTY_MESSAGE
                break
            text = _guard_narrative(text, reply, outcomes, client, messages, system, tools, deadline, clock)
            break
    except BudgetExceededError as exc:
        text = prompts.BUDGET_MESSAGE.format(reason=str(exc).rstrip(".") + ".")
    except TurnDeadlineExceeded:
        text = _with_fallback(prompts.DEADLINE_MESSAGE.format(seconds=settings.chat_turn_seconds), outcomes)
    except Exception as exc:  # never show a traceback in the chat
        if clock() >= deadline:
            text = _with_fallback(prompts.DEADLINE_MESSAGE.format(seconds=settings.chat_turn_seconds), outcomes)
        elif _is_model_unavailable(exc):
            text = _with_fallback(prompts.UNAVAILABLE_MESSAGE.format(reason=_short_reason(exc)), outcomes)
        else:
            logger.exception("Finance chat turn failed")
            text = _with_fallback(prompts.ERROR_MESSAGE.format(reason=f"{type(exc).__name__}: {_short_reason(exc)}"), outcomes)

    analyses = [o for o in outcomes if o.ok and o.result is not None]
    turn.specs = [_spec_dict(o) for o in analyses]
    turn.captions = [o.result.caption for o in analyses]
    turn.summaries = [o.payload for o in outcomes if o.ok]
    turn.assistant_text = text
    session.add_turn(turn)

    yield ChatEvent("explanation", explanation_text(outcomes), status="done",
                    metadata={"specs": turn.specs, "tool_calls": [o.name for o in outcomes]})
    yield ChatEvent("text", text)
    if analyses:
        result = analyses[-1].result
        figure = None
        try:
            from .charts import build_figure

            figure = build_figure(result)
        except Exception:  # a chart problem must not lose the answer
            logger.exception("Could not build the chat chart")
        if figure is not None:
            yield ChatEvent("figure", figure, metadata={"chart": result.chart})
        if not result.table.empty:
            yield ChatEvent("table", result.table.head(MAX_TABLE_ROWS).copy(), metadata={"rows": int(len(result.table))})


def _with_fallback(message: str, outcomes: list[ToolOutcome]) -> str:
    fallback = _fallback_narrative(outcomes)
    return f"{message}\n\n{fallback}" if fallback else message


def _guard_narrative(
    text: str,
    reply: ChatReply,
    outcomes: list[ToolOutcome],
    client: Any,
    messages: list[dict[str, Any]],
    system: str,
    tools: list[dict[str, Any]],
    deadline: float,
    clock: Callable[[], float],
) -> str:
    """Check the answer's numbers against the tool summaries; regenerate once, then flag."""
    problems = check_narrative_numbers(text, _guard_data(outcomes))
    if not problems:
        return text
    logger.info("Chat narrative numbers not in the tool results: %s", problems)
    messages.append(reply.assistant_message())
    messages.append({"role": "user", "content": prompts.narrative_fix(problems)})
    try:
        retry = _call_model(client, messages, system, tools, deadline, clock)
    except BudgetExceededError:
        raise
    except Exception as exc:  # keep the first answer, flagged
        logger.info("Narrative regeneration failed: %s", exc)
        retry = None
    if retry is not None and retry.text.strip() and not retry.tool_calls:
        text = retry.text.strip()
        problems = check_narrative_numbers(text, _guard_data(outcomes))
    if problems:
        has_table = any(o.ok and o.result is not None for o in outcomes)
        text = f"{text}\n\n{prompts.AUTHORITATIVE_NOTE if has_table else prompts.UNVERIFIED_NOTE}"
    return text
