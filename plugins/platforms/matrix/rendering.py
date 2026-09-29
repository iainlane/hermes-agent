"""Matrix formatted HTML and Markdown rendering."""

from __future__ import annotations

from html import escape as _html_escape
from html.parser import HTMLParser
import re


class _MatrixHtmlSanitizer(HTMLParser):
    """Allowlist sanitizer for Matrix-compatible formatted HTML."""

    _ALLOWED_TAGS = {
        "a", "b", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "li", "ol",
        "p", "pre", "s", "strike", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul"}
    _VOID_TAGS = {"br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._unknown_open: dict[str, list[int]] = {}

    def set_cdata_mode(self, elem: str, *, escapable: bool = False) -> None:
        # HTMLParser reads the content after <title>, <script>, <plaintext> and similar start tags as raw
        # text. A lone placeholder with one of those names would turn the rest of the message into escaped
        # markup, so the sanitiser parses the content of every element as HTML.
        return

    @staticmethod
    def _safe_url(value: str) -> str:
        stripped = re.sub(r"[\x00-\x1f\x7f]+", "", value or "").strip()
        match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", stripped)
        scheme = match.group(1).lower() if match else ""
        if scheme and scheme not in {"http", "https", "matrix", "mailto"}:
            return ""
        return stripped

    def _safe_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        safe: list[str] = []
        for key, value in attrs:
            attr = str(key or "").lower()
            raw_value = "" if value is None else str(value)
            if tag == "a" and attr == "href":
                href = self._safe_url(raw_value)
                if href:
                    safe.append(f' href="{_html_escape(href, quote=True)}"')
            elif tag == "code" and attr == "class" and re.fullmatch(r"language-[A-Za-z0-9_+.-]{1,64}", raw_value):
                safe.append(f' class="{_html_escape(raw_value, quote=True)}"')
        return "".join(safe)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._ALLOWED_TAGS:
            self._parts.append(f"<{tag}>" if tag in self._VOID_TAGS else f"<{tag}{self._safe_attrs(tag, attrs)}>")
            return
        # The plain body shows an argument placeholder such as <name> as text, so an unknown start tag
        # stays visible unless a matching end tag follows.
        self._unknown_open.setdefault(tag, []).append(len(self._parts))
        self._parts.append(_html_escape(self.get_starttag_text() or ""))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag not in self._ALLOWED_TAGS:
            if opened := self._unknown_open.get(tag):
                self._parts[opened.pop()] = ""
            return
        if tag in self._VOID_TAGS:
            return
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._parts.append(_html_escape(data))

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._parts)

_TEX_TOKEN_RE = re.compile(r"HERMESTEX(?:DISPLAY|INLINE)(\d+)HERMESTEXEND")

_TEX_DISPLAY_TOKEN = "HERMESTEXDISPLAY%dHERMESTEXEND"

_TEX_INLINE_TOKEN = "HERMESTEXINLINE%dHERMESTEXEND"

def _latex_to_tokens(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace ``$$...$$``/``$...$`` with sentinel tokens.

    Returns the tokenized text plus an ordered ``(tag, tex)`` store, where tag
    is ``div`` for display math and ``span`` for inline math. Dollars that do
    not form a pair (prices, literals) are left untouched.
    """
    if not text or "$" not in text:
        return text, []
    store: list[tuple[str, str]] = []

    def _sub_display(match: re.Match[str]) -> str:
        store.append(("div", match.group(1).strip()))
        return _TEX_DISPLAY_TOKEN % (len(store) - 1)

    def _sub_inline(match: re.Match[str]) -> str:
        store.append(("span", match.group(1).strip()))
        return _TEX_INLINE_TOKEN % (len(store) - 1)

    text = re.sub(r"\$\$([^\n$]+?)\$\$", _sub_display, text)
    text = re.sub(r"(?<![\\$\w])\$([^\n$]+?)\$(?!\w)", _sub_inline, text)
    return text, store

def _tokens_to_mx_maths(html: str, store: list[tuple[str, str]]) -> str:
    """Expand sentinel tokens into ``data-mx-maths`` markup (TeX HTML-escaped)."""

    def _expand(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= len(store):
            # Not one of our tokens (user-typed text that collides with the
            # sentinel format) — leave it verbatim.
            return match.group(0)
        tag, tex = store[idx]
        escaped = _html_escape(tex, quote=True)
        return f'<{tag} data-mx-maths="{escaped}">{escaped}</{tag}>'

    return _TEX_TOKEN_RE.sub(_expand, html)

def _sanitize_matrix_html(html: str) -> str:
    sanitizer = _MatrixHtmlSanitizer()
    try:
        sanitizer.feed(html or "")
        sanitizer.close()
        return sanitizer.get_html()
    except Exception:
        return _html_escape(html or "")

def _pre_sanitize_matrix_markdown(text: str) -> str:
    """Remove unsafe raw HTML before Markdown conversion can escape it."""
    result = re.sub(r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", "", text or "")
    result = re.sub(r"""(?is)\s+on[a-z0-9_-]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", "", result)
    return re.sub(
        r"""(?is)\s+(href|src)\s*=\s*("[^"]*(?:javascript|data|vbscript):[^"]*"|'[^']*(?:javascript|data|vbscript):[^']*'|[^\s>]*(?:javascript|data|vbscript):[^\s>]*)""",
        "", result)
