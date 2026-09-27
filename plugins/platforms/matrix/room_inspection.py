"""Current Matrix room state, members, permissions and pinned messages."""

from __future__ import annotations

import asyncio
from typing import Any

from plugins.platforms.matrix.read_context import _read_access, _raw_event, _visible_event


def _content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        content = value.get("content", value)
        return content if isinstance(content, dict) else {}
    serialize = getattr(value, "serialize", None)
    if callable(serialize):
        return _content(serialize())
    return {}


async def _state(client: Any, room_id: str, event_type: str) -> dict[str, Any]:
    try:
        value = await asyncio.wait_for(client.get_state_event(room_id, event_type), timeout=10.0)
    except Exception as exc:
        if getattr(exc, "errcode", None) == "M_NOT_FOUND" or type(exc).__name__ == "MNotFound":
            return {}
        raise
    return _content(value)


def _text(content: dict[str, Any], field: str) -> str | None:
    value = content.get(field)
    return value[:1200] if isinstance(value, str) and value else None


def _level(content: dict[str, Any], key: str, default: int) -> int:
    value = content.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _user_level(content: dict[str, Any], user_id: str) -> int:
    users = content.get("users")
    users = users if isinstance(users, dict) else {}
    value = users.get(user_id)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _level(content, "users_default", 0)


async def _permissions(client: Any, room_id: str, requester: str, bot: str) -> dict[str, Any]:
    power = await _state(client, room_id, "m.room.power_levels")
    encryption = await _state(client, room_id, "m.room.encryption")
    create_event = await asyncio.wait_for(
        client.get_state_event(room_id, "m.room.create", format="event"), timeout=10.0,
    )
    create = _content(create_event)
    events = power.get("events")
    events = events if isinstance(events, dict) else {}
    pin_level = events.get("m.room.pinned_events")
    if not isinstance(pin_level, int) or isinstance(pin_level, bool):
        pin_level = _level(power, "state_default", 50)
    send_event_type = "m.room.encrypted" if _text(encryption, "algorithm") else "m.room.message"
    message_level = events.get(send_event_type)
    if not isinstance(message_level, int) or isinstance(message_level, bool):
        message_level = _level(power, "events_default", 0)
    bot_level = _user_level(power, bot)
    requester_level = _user_level(power, requester)
    room_version = _text(create, "room_version") or "1"
    creator_ids = {create.get("creator")}
    creator_sender = getattr(create_event, "sender", None)
    if isinstance(create_event, dict):
        creator_sender = create_event.get("sender")
    if isinstance(creator_sender, str):
        creator_ids.add(creator_sender)
    additional = create.get("additional_creators")
    if isinstance(additional, list):
        creator_ids.update(value for value in additional if isinstance(value, str))
    if not power and bot in creator_ids and room_version.isdecimal() and int(room_version) < 12:
        bot_level = 100
    if not power and requester in creator_ids and room_version.isdecimal() and int(room_version) < 12:
        requester_level = 100
    creator_override = room_version.isdecimal() and int(room_version) >= 12
    bot_is_creator = bot in creator_ids and creator_override
    return {
        "requester": {"user_id": requester, "level": requester_level,
                      "creator_override": requester in creator_ids and creator_override},
        "bot": {"user_id": bot, "level": bot_level, "creator_override": bot_is_creator},
        "required": {
            "send_message": message_level,
            "send_event_type": send_event_type,
            "edit_pins": pin_level,
            "invite": _level(power, "invite", 0),
            "kick": _level(power, "kick", 50),
            "ban": _level(power, "ban", 50),
            "redact_other": _level(power, "redact", 50),
        },
        "bot_can_edit_pins": bot_is_creator or bot_level >= pin_level,
    }


async def _pinned_event(
    adapter: Any, client: Any, room_id: str, chat_type: str, event_id: str,
) -> tuple[dict | None, dict | None]:
    try:
        raw = _raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
    except Exception as exc:
        return None, {"event_id": event_id, "error": f"Matrix event read failed: {type(exc).__name__}"}
    if raw.get("event_id") != event_id:
        return None, {"event_id": event_id, "error": "Matrix event was not returned"}
    visible, error = await _visible_event(adapter, raw, room_id, chat_type)
    if visible is None and error is None:
        error = {"event_id": event_id, "error": "event has no visible message"}
    return visible, error


async def inspect_matrix_room(
    adapter: Any, kind: str, room_id: str, limit: int, *, requester: str,
) -> dict[str, Any]:
    client, chat_type, error = await _read_access(adapter, room_id, requester)
    if error is not None:
        return error

    try:
        if kind == "state":
            fields = {
                "name": ("m.room.name", "name"),
                "topic": ("m.room.topic", "topic"),
                "canonical_alias": ("m.room.canonical_alias", "alias"),
                "join_rule": ("m.room.join_rules", "join_rule"),
                "history_visibility": ("m.room.history_visibility", "history_visibility"),
                "encryption": ("m.room.encryption", "algorithm"),
            }
            result = {"room_id": room_id}
            for key, (event_type, field) in fields.items():
                result[key] = _text(await _state(client, room_id, event_type), field)
            return result

        if kind == "members":
            profiles = await asyncio.wait_for(client.get_joined_members(room_id), timeout=10.0)
            members = []
            for user_id, profile in sorted(profiles.items(), key=lambda item: str(item[0]))[:limit]:
                content = _content(profile)
                display_name = _text(content, "displayname") or getattr(profile, "displayname", None)
                avatar_url = _text(content, "avatar_url") or getattr(profile, "avatar_url", None)
                members.append({
                    "user_id": str(user_id),
                    "display_name": str(display_name)[:1200] if display_name else None,
                    "avatar_url": str(avatar_url)[:1200] if avatar_url else None,
                })
            return {"members": members, "total": len(profiles), "truncated": len(profiles) > limit}

        if kind == "permissions":
            return await _permissions(client, room_id, requester, adapter._user_id)

        if kind == "pins":
            pinned = await _state(client, room_id, "m.room.pinned_events")
            event_ids = pinned.get("pinned")
            if not isinstance(event_ids, list) or not all(isinstance(value, str) for value in event_ids):
                event_ids = []
            selected = event_ids[:limit]
            semaphore = asyncio.Semaphore(10)

            async def fetch(event_id: str) -> tuple[dict | None, dict | None]:
                async with semaphore:
                    return await _pinned_event(adapter, client, room_id, chat_type, event_id)

            resolved = await asyncio.gather(*(fetch(event_id) for event_id in selected))
            return {
                "events": [visible for visible, _ in resolved if visible is not None],
                "total": len(event_ids), "truncated": len(event_ids) > limit,
                "errors": [failure for _, failure in resolved if failure is not None],
            }
    except Exception as exc:
        return {"error": f"Matrix room inspection failed: {type(exc).__name__}"}

    return {"error": "kind must be state, members, permissions, or pins"}
