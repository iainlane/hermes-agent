"""Standalone Matrix delivery through the Client-Server API."""

import asyncio
from contextlib import suppress
import os
import re
import time

from agent.secret_scope import get_secret
from gateway.platforms._shared import send_error
from gateway.platforms.base import BasePlatformAdapter
from typing import Any


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """standalone_sender_fn: out-of-process delivery via the Client-Server API (cron without gateway)."""
    extra = getattr(pconfig, "extra", {}) or {}
    try:
        import aiohttp
    except ImportError:
        return send_error("aiohttp not installed. Run: pip install aiohttp")
    try:
        # In-turn reads inside an installed secret scope: honor get_secret, no env fallback — for the
        # homeserver too, so the scoped token is never sent to the default profile's server.
        homeserver = (extra.get("homeserver") or get_secret("MATRIX_HOMESERVER", "") or "").rstrip("/")
        token = getattr(pconfig, "token", None) or get_secret("MATRIX_ACCESS_TOKEN", "") or ""
        if not homeserver or not token:
            return send_error("Matrix not configured (MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN required)")
        from urllib.parse import quote
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        # asyncio.wait_for, not aiohttp.ClientTimeout: cron invokes this via
        # run_coroutine_threadsafe ("Timeout context manager should be used inside a task").
        async with aiohttp.ClientSession() as session:
            async def _do_send(payload):
                txn_id = f"hermes_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
                url = f"{homeserver}/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{txn_id}"
                async with session.put(url, headers=headers, json=payload) as resp:
                    if resp.status not in {200, 201}:
                        return send_error(f"Matrix API error ({resp.status}): {await resp.text()}")
                    data = await resp.json()
                    return {"success": True, "platform": "matrix", "chat_id": chat_id,
                            "message_id": data.get("event_id")}
            for payload in _standalone_payloads(message):
                try:
                    result = await asyncio.wait_for(_do_send(payload), timeout=30)
                except TimeoutError:
                    return send_error("Matrix API timeout (30s)")
                if not result.get("success"):
                    return result
            return result
    except Exception as e:
        return send_error(f"Matrix send failed: {e}")


def _standalone_payloads(message: str) -> list[dict[str, Any]]:
    """One m.room.message content per chunk. The caller chunks on characters, and a chunk of
    non-Latin text can still exceed the homeserver's byte cap, so re-chunk on UTF-8 bytes."""
    from plugins.platforms.matrix.adapter import (
        MATRIX_MAX_MESSAGE_LENGTH_CEILING, _MAX_CONTENT_BYTES, _content_bytes, _utf8_len,
    )
    from plugins.platforms.matrix.rendering import _latex_to_tokens, _tokens_to_mx_maths

    _md = None
    with suppress(ImportError):
        import markdown as _md
    payloads = []
    for chunk in BasePlatformAdapter.truncate_message(message, MATRIX_MAX_MESSAGE_LENGTH_CEILING, _utf8_len):
        payload = {"msgtype": "m.text", "body": chunk}
        if _md is not None:
            tokenized, tex_store = _latex_to_tokens(chunk)
            html = _md.markdown(tokenized, extensions=["fenced_code", "tables"])
            formatted = {"format": "org.matrix.custom.html", "formatted_body": _tokens_to_mx_maths(
                re.sub(r"<h[1-6]>(.*?)</h[1-6]>", r"<strong>\1</strong>", html), tex_store)}
            if _content_bytes({**payload, **formatted}) <= _MAX_CONTENT_BYTES:
                payload.update(formatted)
        payloads.append(payload)
    return payloads
