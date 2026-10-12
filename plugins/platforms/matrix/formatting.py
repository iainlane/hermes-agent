"""Fallback Markdown rendering for Matrix payloads."""

import re
from html import escape as _html_escape


class MatrixFormattingMixin:
    @staticmethod
    def _markdown_to_html_fallback(text: str) -> str:
        """Comprehensive regex Markdown-to-HTML for Matrix."""
        from plugins.platforms.matrix.adapter import MatrixAdapter

        placeholders: list = []

        def _is_bq_line(ln: str) -> bool:
            return ln.startswith(("&gt; ", "> ")) or ln in ("&gt;", ">")

        def _protect_html(html_fragment: str) -> str:
            idx = len(placeholders)
            placeholders.append(html_fragment)
            return f"\x00PROTECTED{idx}\x00"

        result = re.sub(
            r"```(\w*)\n(.*?)```",
            lambda m: _protect_html(
                f'<pre><code class="language-{_html_escape(m.group(1))}">{_html_escape(m.group(2))}</code></pre>'
                if m.group(1) else f"<pre><code>{_html_escape(m.group(2))}</code></pre>"),
            text, flags=re.DOTALL)
        result = re.sub(r"`([^`\n]+)`", lambda m: _protect_html(f"<code>{_html_escape(m.group(1))}</code>"), result)
        # Protect markdown links before escaping.
        result = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: _protect_html(
                f'<a href="{MatrixAdapter._sanitize_link_url(m.group(2))}">{_html_escape(m.group(1))}</a>'),
            result)
        result = "".join(p if p.startswith("\x00PROTECTED") else _html_escape(p)
                         for p in re.split(r"(\x00PROTECTED\d+\x00)", result))
        # Block-level transforms (line-oriented): hr, headers, blockquote, lists.
        lines = result.split("\n")
        out_lines: list = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if re.match(r"^[\s]*([-*_])\s*\1\s*\1[\s\-*_]*$", line):
                out_lines.append("<hr>")
                i += 1
                continue
            hdr = re.match(r"^(#{1,6})\s+(.+)$", line)
            if hdr:
                level = len(hdr.group(1))
                out_lines.append(f"<h{level}>{hdr.group(2).strip()}</h{level}>")
                i += 1
                continue
            if _is_bq_line(line):
                bq_lines = []
                while i < len(lines) and _is_bq_line(lines[i]):
                    ln = lines[i]
                    bq_lines.append(ln[5:] if ln.startswith("&gt; ") else ln[2:] if ln.startswith("> ") else "")
                    i += 1
                out_lines.append(f"<blockquote>{'<br>'.join(bq_lines)}</blockquote>")
                continue
            for item_re, tag in ((r"^[\s]*[-*+]\s+(.+)$", "ul"), (r"^[\s]*\d+[.)]\s+(.+)$", "ol")):
                if re.match(item_re, line):
                    items = []
                    while i < len(lines):
                        item_match = re.match(item_re, lines[i])
                        if item_match is None:
                            break
                        items.append(item_match.group(1))
                        i += 1
                    out_lines.append(f"<{tag}>{''.join(f'<li>{item}</li>' for item in items)}</{tag}>")
                    break
            else:
                out_lines.append(line)
                i += 1
        result = "\n".join(out_lines)
        for pattern, repl in (
            (r"\*\*(.+?)\*\*", r"<strong>\1</strong>"), (r"__(.+?)__", r"<strong>\1</strong>"),
            (r"\*(.+?)\*", r"<em>\1</em>"), (r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>"),
            (r"~~(.+?)~~", r"<del>\1</del>")):
            result = re.sub(pattern, repl, result, flags=re.DOTALL)
        result = re.sub(r"\n", "<br>\n", result)
        result = re.sub(r"<br>\n(</?(?:pre|blockquote|h[1-6]|ul|ol|li|hr))", r"\n\1", result)
        result = re.sub(r"(</(?:pre|blockquote|h[1-6]|ul|ol|li)>)<br>", r"\1", result)
        for idx, original in enumerate(placeholders):
            result = result.replace(f"\x00PROTECTED{idx}\x00", original)
        return result

