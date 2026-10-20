"""One HTML-to-text helper for report plain text, history excerpts, and Q&A retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable
from html.parser import HTMLParser


# Tags that start/end a line in the report's plain-text alternative.
BLOCK_TAGS = frozenset({"address", "article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "tr"})


class _TextCollector(HTMLParser):
    def __init__(self, skip_tags: frozenset[str], block_tags: frozenset[str], strip_parts: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_tags = skip_tags
        self.block_tags = block_tags
        self.strip_parts = strip_parts
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in self.skip_tags:
            self.skip_depth += 1
        if name in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in self.skip_tags and self.skip_depth:
            self.skip_depth -= 1
        if name in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.strip_parts:
            if data.strip():
                self.parts.append(data.strip())
        else:
            self.parts.append(data)


def html_to_text(
    markup: str,
    *,
    skip_tags: Iterable[str] = (),
    block_newlines: bool | Iterable[str] = False,
    separator: str | None = None,
) -> str:
    """Extract visible text from HTML.

    skip_tags: elements whose text is dropped (e.g. style, script). block_newlines: True for
    BLOCK_TAGS, or an explicit tag set; each start/end tag of those elements adds a newline.
    separator None keeps text nodes verbatim, then collapses runs of blank lines to one blank
    line and strips the result; a string strips each text node, drops empty ones, and joins the
    rest with it.
    """
    if block_newlines is True:
        blocks = BLOCK_TAGS
    elif block_newlines is False:
        blocks = frozenset()
    else:
        blocks = frozenset(tag.lower() for tag in block_newlines)
    parser = _TextCollector(frozenset(tag.lower() for tag in skip_tags), blocks, strip_parts=separator is not None)
    # No close(): like the extractors this replaced, an unterminated trailing fragment is not flushed.
    parser.feed(markup or "")
    if separator is not None:
        return separator.join(parser.parts)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()
