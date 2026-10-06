"""Per-conversation state for the finance chat (small; the transaction frame is never stored here).

A ``ChatSession`` keeps each turn's user text, the specs it ran, the compact summaries the model
saw, and the assistant's answer. The last ``history_turns`` turns are replayed verbatim; older
turns are folded into a deterministic rolling summary. ``last_spec`` lets follow-ups ("now by
month") start from the previous analysis. ``sync_history`` drops turns the UI no longer shows
(undo, retry, or clear).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

ROLLING_SUMMARY_MAX_CHARS = 1200
NO_ANSWER = "(no answer)"


@dataclass
class Turn:
    user_text: str
    specs: list[dict[str, Any]] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""


def _clip(text: str, size: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= size else text[: size - 1] + "…"


def _role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


class ChatSession:
    def __init__(self, session_id: str | None = None, *, focus_month: date | None = None, history_turns: int = 6) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.focus_month = focus_month
        self.history_turns = max(1, int(history_turns))
        self.turns: list[Turn] = []
        self.last_spec: dict[str, Any] | None = None
        self.client: Any = None  # the chat model client, built once per session (one BudgetGuard)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if turn.specs:
            self.last_spec = dict(turn.specs[-1])

    def _recompute_last_spec(self) -> None:
        self.last_spec = next((dict(turn.specs[-1]) for turn in reversed(self.turns) if turn.specs), None)

    def sync_history(self, ui_history: Iterable[Any] | None) -> int:
        """Truncate to the user turns the UI still shows (its history excludes the new message).

        Returns how many turns were dropped.
        """
        shown = sum(1 for message in (ui_history or []) if _role(message) == "user")
        dropped = max(0, len(self.turns) - shown)
        if dropped:
            del self.turns[shown:]
            self._recompute_last_spec()
        return dropped

    def recent_turns(self) -> list[Turn]:
        return self.turns[-self.history_turns:]

    def older_turns(self) -> list[Turn]:
        return self.turns[: max(0, len(self.turns) - self.history_turns)]

    def rolling_summary(self, max_chars: int = ROLLING_SUMMARY_MAX_CHARS) -> str:
        """Deterministic one-line-per-turn summary of the turns older than the recent window."""
        lines = []
        for turn in self.older_turns():
            line = f"- Q: {_clip(turn.user_text, 100)}"
            if turn.captions:
                line += f" | computed: {_clip(turn.captions[-1], 120)}"
            line += f" | answer: {_clip(turn.assistant_text or NO_ANSWER, 120)}"
            lines.append(line)
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            if total + len(line) + 1 > max_chars:
                break
            kept.insert(0, line)
            total += len(line) + 1
        if len(kept) < len(lines):
            kept.insert(0, f"- ({len(lines) - len(kept)} earlier turns omitted)")
        return "\n".join(kept)

    def history_messages(self) -> list[dict[str, Any]]:
        """Provider-neutral messages for the recent turns (plain text; no tool rounds or thinking)."""
        messages: list[dict[str, Any]] = []
        for turn in self.recent_turns():
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.assistant_text or NO_ANSWER})
        return messages
