"""Command reply formatting for messaging clients."""

import re
from typing import Any


_MATRIX_CODE_COMMAND_RE = re.compile(
    r"`/([A-Za-z0-9][A-Za-z0-9_-]*)(?=[\s`])"
)


def _platformize_command_mentions(text: str, platform: Any) -> str:
    """Render command mentions using syntax the target client can send.

    Dispatch remains slash-based internally. Telegram receives its existing
    name sanitization; Matrix receives bang-prefixed command tokens because
    Element clients may reserve slash commands locally.
    """
    from gateway.run import _telegramize_command_mentions

    rendered = _telegramize_command_mentions(text, platform)
    platform_value = getattr(platform, "value", platform)
    if platform_value != "matrix":
        return rendered
    return _MATRIX_CODE_COMMAND_RE.sub(
        lambda match: f"`!{match.group(1)}",
        rendered,
    )
