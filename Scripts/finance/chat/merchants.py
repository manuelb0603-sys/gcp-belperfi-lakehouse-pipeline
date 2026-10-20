"""Merchant-name normalization for the finance chat.

Card descriptors are noisy ("SQ *BLUE BOTTLE #123", padded Amex lines with phone and
state). ``normalize_merchant`` turns them into a stable display name so grouping by
merchant is meaningful. It also strips characters that could be read as links, markup,
or HTML, because merchant names end up in chart labels and model-visible summaries.
"""

from __future__ import annotations

import re
from typing import Mapping

CARD_PAYMENT = "CARD PAYMENT"
CARD_CREDIT = "CARD CREDIT"
UNKNOWN = "Unknown"

US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR".split()
)

_PAYMENT = re.compile(
    r"THANK\s*YOU|\bAUTOPAY\b|\bAUTOMATIC PAYMENT\b|^(?:ONLINE |MOBILE |ELECTRONIC |INTERNET )?PAYMENT\b"
)
_CREDIT = re.compile(r"\bSTATEMENT CREDIT\b|\bCREDIT\s*$")
_URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
_PREFIX = re.compile(r"^(?:(?:SQ|TST|PY|PP|PAYPAL|SP)\s*\*\s*|SP\s+)")
_AMAZON = re.compile(r"^(?:AMZN\b|AMAZON(?:\.COM|\s+MKTP|\s+MKTPL|\s+MKTPLACE|\s+MARKETPLACE)\b)")
_PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b|\b\d{10}\b")
_STORE_NUMBER = re.compile(r"(?:\b(?:STORE|STR|NO|NUM)\s*)?#\s*\d+|\b(?:STORE|STR)\s+\d+\b")
_DIGIT_RUN = re.compile(r"\b[A-Z]?-?\d{3,}\b")
_INJECTION_CHARS = re.compile(r"[\[\]()!<>*_`{}|\\]")
_TRAILING_NOISE = re.compile(r"(?:\s+(?:STORE|STR|NO|NUM|#))+\s*$")
_EDGE_PUNCT = " -.,/:;'\"&+"


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_state(text: str) -> str:
    """Drop one trailing two-letter US state code (keeps single-word names intact)."""
    parts = text.split(" ")
    if len(parts) > 1 and parts[-1] in US_STATES:
        return " ".join(parts[:-1])
    return text


def _title(text: str) -> str:
    """Title-case word by word without str.title's apostrophe bug (MCDONALD'S -> Mcdonald's)."""
    return " ".join(word[:1].upper() + word[1:].lower() for word in text.split(" "))


def _alias_key(text: str) -> str:
    return _collapse(str(text).upper())


def _apply_alias(name: str, aliases: Mapping[str, str] | None) -> str | None:
    if not aliases:
        return None
    keyed = {_alias_key(key): value for key, value in aliases.items()}
    if name in keyed:
        return keyed[name]
    for key in sorted(keyed, key=len, reverse=True):
        if key and name.startswith(key + " "):
            return keyed[key]
    return None


def clean_label(text: object, max_length: int = 60) -> str:
    """Injection hygiene for any free-text label: no URLs or markup characters, bounded length."""
    value = _URL.sub(" ", str(text if text is not None else ""))
    value = _collapse(_INJECTION_CHARS.sub(" ", value))
    return value[:max_length].strip()


def normalize_merchant(description: object, aliases: Mapping[str, str] | None = None) -> str:
    """Stable, display-ready merchant name for a raw card descriptor.

    Payments and autopay become "Card Payment"; statement credits "Card Credit". Alias
    keys match the cleaned upper-case name exactly or as a word prefix; alias values are
    display names used as written (after the same injection hygiene).
    """
    text = _collapse(str(description if description is not None else "").upper())
    if not text:
        return UNKNOWN
    if _PAYMENT.search(text):
        return _title(CARD_PAYMENT)
    if _CREDIT.search(text):
        return _title(CARD_CREDIT)

    text = _URL.sub(" ", text)
    text = _collapse(_PREFIX.sub("", text))
    if _AMAZON.search(text):
        text = "AMAZON"
    text = _PHONE.sub(" ", text)
    text = _STORE_NUMBER.sub(" ", text)
    text = _DIGIT_RUN.sub(" ", text)
    text = _collapse(_INJECTION_CHARS.sub(" ", text))
    text = _collapse(_TRAILING_NOISE.sub("", text)).strip(_EDGE_PUNCT)
    text = _collapse(_strip_state(text)).strip(_EDGE_PUNCT)
    text = _collapse(_TRAILING_NOISE.sub("", text)).strip(_EDGE_PUNCT)
    if not text:
        return UNKNOWN

    alias = _apply_alias(text, aliases)
    if alias is not None:
        return clean_label(alias) or UNKNOWN
    return _title(text)
