"""Mechanical HTML gate for author reports and the plain-text derivation."""

from __future__ import annotations

import re
from html.parser import HTMLParser


HISTORY_LABEL = "Previous report for reference:"


class _UnsafeMarkupParser(HTMLParser):
    # This parser is a lightweight blocklist: if the model tries to inject HTML that can
    # execute or affect the email client, the report is rejected (and sent back to the
    # author as revision feedback) instead of being sent.
    unsafe_tags = {"script", "iframe", "object", "embed", "form", "input", "link", "meta"}

    def __init__(self) -> None:
        """Initialize the parser state for unsafe-tag detection."""
        super().__init__()
        self.unsafe = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Mark the HTML as unsafe when a blocked tag appears."""
        if tag in self.unsafe_tags:
            self.unsafe = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing blocked tags the same as normal opening tags."""
        self.handle_starttag(tag, attrs)


class _ReportTextExtractor(HTMLParser):
    """Extract a readable plain-text alternative from a valid HTML report."""

    block_tags = {"address", "article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "style":
            self.in_style = True
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style":
            self.in_style = False
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.in_style:
            self.parts.append(data)


def _plain_text_from_html(report_html: str) -> str:
    parser = _ReportTextExtractor()
    parser.feed(report_html)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


class ReportSanitizationError(ValueError):
    """A report failed the mechanical HTML gate; category maps to a validator finding category."""

    def __init__(self, message: str, category: str = "html") -> None:
        super().__init__(message)
        self.category = category


def sanitize_html(html: str, max_bytes: int = 250_000, require_history_label: bool = False) -> str:
    """Strip unsafe markup while preserving the report layout the model intended."""
    # This is the final mechanical gate before validation: if the model produces HTML that
    # could execute, is too large, or lacks a required history label, ReportSanitizationError
    # is raised and the harness returns it to the author as feedback; nothing unsanitized is sent.
    if not html or len(html.encode("utf-8")) > max_bytes:
        raise ReportSanitizationError("Report HTML is empty or exceeds the configured size limit")
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*>.*?<\s*/\s*\1\s*>", "", html, flags=re.I | re.S)
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*?/?>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+on[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:href|src)\s*=\s*(['\"])\s*javascript:[^'\"]*\1", "", cleaned, flags=re.I)
    parser = _UnsafeMarkupParser()
    parser.feed(cleaned)
    if parser.unsafe:
        raise ReportSanitizationError("Report contains unsafe HTML elements")
    if require_history_label and HISTORY_LABEL not in cleaned:
        raise ReportSanitizationError("Report must explicitly label prior report excerpts", category="history")
    return cleaned
