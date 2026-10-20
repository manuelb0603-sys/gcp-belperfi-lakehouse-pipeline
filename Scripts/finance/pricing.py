"""The one Claude price table, used for spend records, budget checks, and cost estimates.

Prices are USD per million tokens (input, output). Prompt-cache reads cost 0.1x the input
price and cache writes (5-minute) 1.25x. A Claude model missing from the table is priced
conservatively at the most expensive listed rate (Fable), never as free.
"""

from __future__ import annotations

from typing import Any, Mapping

# USD per million tokens (input, output). Longest matching prefix wins, so dated model ids
# such as "claude-haiku-4-5-20251001" use their family's price.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
}
# Unknown Claude models are billed at this rate for budgeting (the most expensive listed model).
UNKNOWN_MODEL_PRICE = ANTHROPIC_PRICES["claude-fable-5-1"]
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25
# Characters per token used when the input token count is not known yet (conservative: real
# English prompts average closer to 4 characters per token).
CHARS_PER_TOKEN = 3.5
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def known_price(model: str | None) -> tuple[float, float] | None:
    """The listed (input, output) price for a model id, or None when it is not in the table."""
    name = str(model or "")
    matches = [key for key in ANTHROPIC_PRICES if name == key or name.startswith(key + "-")]
    return ANTHROPIC_PRICES[max(matches, key=len)] if matches else None


def price_for(model: str | None) -> tuple[float, float]:
    """(input, output) USD per million tokens; unknown models get UNKNOWN_MODEL_PRICE."""
    return known_price(model) or UNKNOWN_MODEL_PRICE


def usage_tokens(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """The four Anthropic usage counts as ints (missing or None -> 0)."""
    usage = usage or {}
    return {key: int(usage.get(key) or 0) for key in USAGE_KEYS}


def cost_usd(model: str | None, usage: Mapping[str, Any] | None) -> float:
    """USD for one call: uncached input, cache reads (0.1x), cache writes (1.25x), and output."""
    tokens = usage_tokens(usage)
    input_price, output_price = price_for(model)
    total = (
        tokens["input_tokens"] * input_price
        + tokens["cache_read_input_tokens"] * input_price * CACHE_READ_MULTIPLIER
        + tokens["cache_creation_input_tokens"] * input_price * CACHE_WRITE_MULTIPLIER
        + tokens["output_tokens"] * output_price
    ) / 1_000_000
    return round(total, 6)


def estimate_input_tokens(chars: int) -> int:
    """Input tokens estimated from prompt characters (chars / 3.5, rounded up)."""
    chars = max(0, int(chars))
    return -(-chars * 2 // 7)  # ceil(chars / 3.5) in integer math


def worst_case_call_usd(model: str | None, input_chars_or_tokens: int, max_tokens: int, *, tokens: bool = False) -> float:
    """Upper estimate for one call before it is made: the prompt at the list input price plus max_tokens output.

    input_chars_or_tokens is prompt characters (estimated as chars / 3.5 tokens) unless tokens=True.
    """
    input_tokens = int(input_chars_or_tokens) if tokens else estimate_input_tokens(input_chars_or_tokens)
    input_price, output_price = price_for(model)
    return round((max(0, input_tokens) * input_price + max(0, int(max_tokens or 0)) * output_price) / 1_000_000, 6)
