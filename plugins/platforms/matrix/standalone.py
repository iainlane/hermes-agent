"""Standalone Matrix delivery through the Client-Server API."""

import asyncio
from contextlib import suppress
import os
import re
import time

from agent.secret_scope import get_secret
from gateway.platforms._shared import send_error


async def _resolve_matrix_room_alias(homeserver: str, token: str, alias: str):
    """Resolve a room alias to a room ID, or return a lookup error."""
    try:
        import aiohttp
    except ImportError:
        return None, "aiohttp not installed. Run: pip install aiohttp"
    from urllib.parse import quote
    encoded_alias = quote(alias, safe="")
    url = f"{homeserver}/_matrix/client/v3/directory/room/{encoded_alias}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with aiohttp.ClientSession() as session:
            async def _lookup():
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return None, f"alias resolution failed ({resp.status}): {body}"
                    return await resp.json(), None

            data, err = await asyncio.wait_for(_lookup(), timeout=15)
        if err:
            return None, err
        room_id = data.get("room_id")
        if not room_id:
            return None, f"alias resolution returned no room_id for {alias}"
        return room_id, None
    except Exception as e:
        return None, f"alias resolution failed: {e}"



async def _resolve_matrix_room_alias_target(homeserver: str, token: str, chat_id: str):
    """Return a concrete room ID for alias targets, or the original target."""
    if not chat_id.startswith("#"):
        return chat_id, None
    resolved, err = await _resolve_matrix_room_alias(homeserver, token, chat_id)
    if err:
        return chat_id, f"Matrix alias '{chat_id}': {err}"
    return resolved, None



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
        chat_id, err = await _resolve_matrix_room_alias_target(homeserver, token, chat_id)
        if err:
            return send_error(err)
        txn_id = f"hermes_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        from urllib.parse import quote
        url = f"{homeserver}/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{txn_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"msgtype": "m.text", "body": message}
        with suppress(ImportError):
            from plugins.platforms.matrix.rendering import _latex_to_tokens, _tokens_to_mx_maths
            import markdown as _md
            tokenized, tex_store = _latex_to_tokens(message)
            html = _md.markdown(tokenized, extensions=["fenced_code", "tables"])
            payload["format"] = "org.matrix.custom.html"
            payload["formatted_body"] = _tokens_to_mx_maths(
                re.sub(r"<h[1-6]>(.*?)</h[1-6]>", r"<strong>\1</strong>", html), tex_store)
        if thread_id:
            payload["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_id,
                "is_falling_back": True,
            }
        # asyncio.wait_for, not aiohttp.ClientTimeout: cron invokes this via
        # run_coroutine_threadsafe ("Timeout context manager should be used inside a task").
        async with aiohttp.ClientSession() as session:
            async def _do_send():
                async with session.put(url, headers=headers, json=payload) as resp:
                    if resp.status not in {200, 201}:
                        return send_error(f"Matrix API error ({resp.status}): {await resp.text()}")
                    data = await resp.json()
                    return {"success": True, "platform": "matrix", "chat_id": chat_id,
                            "message_id": data.get("event_id")}
            try:
                return await asyncio.wait_for(_do_send(), timeout=30)
            except asyncio.TimeoutError:
                return send_error("Matrix API timeout (30s)")
    except Exception as e:
        return send_error(f"Matrix send failed: {e}")
