"""Current Matrix room state, members, permissions and pinned messages."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from gateway.session_context import get_session_env, get_session_transport
from hermes_constants import hermes_home_key
from plugins.platforms.matrix.read_context import (
    MatrixReadEvent, Method, _current_read_access, _read_access, _raw_event, _visible_event,
)
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache


def _content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        content = value.get("content", value)
        return content if isinstance(content, dict) else {}
    serialize = getattr(value, "serialize", None)
    if callable(serialize):
        return _content(serialize())
    return {}


async def _state(context: _InspectionContext, event_type: str) -> dict[str, Any]:
    try:
        value = await context.request(lambda: context.client.get_state_event(context.room_id, event_type))
    except Exception as exc:
        if getattr(exc, "errcode", None) == "M_NOT_FOUND" or type(exc).__name__ == "MNotFound":
            return {}
        raise
    return _content(value)


def _text(content: dict[str, Any], field: str) -> str | None:
    value = content.get(field)
    return value[:1200] if isinstance(value, str) and value else None


def _numeric_level(value: Any, default: int, legacy_strings: bool) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if legacy_strings and isinstance(value, str) and re.fullmatch(r"\s*[+-]?[0-9]+\s*", value):
        return int(value)
    return default


def _level(content: dict[str, Any], key: str, default: int, legacy_strings: bool) -> int:
    return _numeric_level(content.get(key), default, legacy_strings)


def _user_level(content: dict[str, Any], user_id: str, legacy_strings: bool) -> int:
    users = content.get("users")
    users = users if isinstance(users, dict) else {}
    default = _level(content, "users_default", 0, legacy_strings)
    return _numeric_level(users.get(user_id), default, legacy_strings)


async def _permissions(context: _InspectionContext) -> dict[str, Any]:
    requester, bot = context.requester, context.owner.bot_id
    power = await _state(context, "m.room.power_levels")
    encryption = await _state(context, "m.room.encryption")
    create_event = await context.request(
        lambda: context.client.get_state_event(context.room_id, "m.room.create", format="event"),
    )
    create = _content(create_event)
    room_version = _text(create, "room_version") or "1"
    numeric_version = int(room_version) if room_version.isdecimal() else None
    legacy_strings = numeric_version is not None and numeric_version <= 9
    events = power.get("events")
    events = events if isinstance(events, dict) else {}
    pin_level = _numeric_level(
        events.get("m.room.pinned_events"),
        _level(power, "state_default", 50, legacy_strings), legacy_strings,
    )
    send_event_type = "m.room.encrypted" if _text(encryption, "algorithm") else "m.room.message"
    message_level = _numeric_level(
        events.get(send_event_type),
        _level(power, "events_default", 0, legacy_strings), legacy_strings,
    )
    bot_level = _user_level(power, bot, legacy_strings)
    requester_level = _user_level(power, requester, legacy_strings)
    creator_ids: set[str] = set()
    creator_sender = getattr(create_event, "sender", None)
    if isinstance(create_event, dict):
        creator_sender = create_event.get("sender")
    if isinstance(creator_sender, str):
        creator_ids.add(creator_sender)
    elif numeric_version is not None and numeric_version < 11:
        legacy_creator = create.get("creator")
        if isinstance(legacy_creator, str):
            creator_ids.add(legacy_creator)
    creator_override = numeric_version is not None and numeric_version >= 12
    if creator_override:
        additional = create.get("additional_creators")
        if isinstance(additional, list):
            creator_ids.update(value for value in additional if isinstance(value, str))
    if not power and bot in creator_ids and not creator_override:
        bot_level = 100
    if not power and requester in creator_ids and not creator_override:
        requester_level = 100
    bot_is_creator = bot in creator_ids and creator_override
    return {
        "requester": {"user_id": requester, "level": requester_level,
                      "creator_override": requester in creator_ids and creator_override},
        "bot": {"user_id": bot, "level": bot_level, "creator_override": bot_is_creator},
        "required": {
            "send_message": message_level,
            "send_event_type": send_event_type,
            "edit_pins": pin_level,
            "invite": _level(power, "invite", 0, legacy_strings),
            "kick": _level(power, "kick", 50, legacy_strings),
            "ban": _level(power, "ban", 50, legacy_strings),
            "redact_other": _level(power, "redact", 50, legacy_strings),
        },
        "bot_can_edit_pins": bot_is_creator or bot_level >= pin_level,
    }


class _InspectionRejected(Exception):
    def __init__(self, error: dict[str, str]) -> None:
        self.error = error
        super().__init__(error["error"])


def _session_identity() -> tuple[str, ...]:
    return tuple(str(get_session_env(key) or "") for key in (
        "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_PROFILE", "HERMES_SESSION_KEY", "HERMES_SESSION_ID",
    ))


@dataclass(frozen=True)
class _InspectionOwner:
    adapter: Any
    client: Any
    cache: MatrixEventContextCache | None
    home: str
    transport: tuple[Any, Any]
    session: tuple[str, ...]
    bot_id: str
    account_id: str
    device_id: str
    api: Any
    api_url: str
    api_token: str | None
    http_session: Any
    crypto: Any
    crypto_store: Any
    store_dir: Any

    @classmethod
    def capture(cls, adapter: Any) -> _InspectionOwner:
        client = adapter._client
        api = getattr(client, "api", None)
        crypto = getattr(client, "crypto", None)
        return cls(
            adapter, client, getattr(adapter, "_event_context_cache", None), hermes_home_key(),
            get_session_transport(), _session_identity(), str(adapter._user_id or ""),
            str(getattr(client, "mxid", "") or ""), str(getattr(client, "device_id", "") or ""),
            api, str(getattr(api, "base_url", "") or ""), getattr(api, "token", None),
            getattr(api, "session", None), crypto, getattr(crypto, "crypto_store", None),
            getattr(adapter, "_store_dir", None),
        )

    def check(self, room_id: str, requester: str) -> None:
        if getattr(self.adapter, "_closing", False) or self.adapter._client is None:
            raise _InspectionRejected({"error": "Matrix client is disconnected"})
        current = self.capture(self.adapter)
        if (
            current.client is not self.client or current.cache is not self.cache
            or current.api is not self.api or current.http_session is not self.http_session
            or current.crypto is not self.crypto or current.crypto_store is not self.crypto_store
            or current.transport[0] is not self.transport[0] or current.transport[1] is not self.transport[1]
            or current.home != self.home or current.session != self.session
            or current.bot_id != self.bot_id or current.account_id != self.account_id
            or current.device_id != self.device_id or current.api_url != self.api_url
            or current.api_token != self.api_token or current.store_dir != self.store_dir
            or (self.transport[0] is not None and (
                self.transport[0] is not self.adapter or self.session[:3] != ("matrix", room_id, requester)
            ))
        ):
            raise _InspectionRejected({"error": "Matrix room inspection context changed"})

    async def access(self, room_id: str, requester: str, chat_type: str | None = None) -> str:
        self.check(room_id, requester)
        client, current_chat_type, error = await asyncio.wait_for(
            _read_access(self.adapter, room_id, requester), timeout=10.0,
        )
        self.check(room_id, requester)
        if error is None:
            assert current_chat_type is not None
            client, current_chat_type, error = _current_read_access(
                self.adapter, room_id, requester, current_chat_type,
            )
        if error is not None:
            raise _InspectionRejected(error)
        if client is not self.client or (chat_type is not None and current_chat_type != chat_type):
            raise _InspectionRejected({"error": "Matrix room inspection context changed"})
        assert current_chat_type is not None
        return current_chat_type


@dataclass(frozen=True)
class _InspectionContext:
    adapter: Any
    client: Any
    room_id: str
    chat_type: str
    requester: str
    limit: int
    owner: _InspectionOwner
    dependencies: dict[str, MatrixEventContext] = field(default_factory=dict, compare=False)

    async def check_access(self) -> None:
        await self.owner.access(self.room_id, self.requester, self.chat_type)

    async def request(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        await self.check_access()
        try:
            return await asyncio.wait_for(operation(), timeout=10.0)
        finally:
            await self.check_access()


async def _pinned_event(context: _InspectionContext, event_id: str) -> MatrixReadEvent:
    cache = context.owner.cache
    assert cache is not None
    before = context.dependencies[event_id]
    raw = {"event_id": event_id, "room_id": context.room_id}
    try:
        await context.check_access()
        path = (
            f"/_matrix/client/v3/rooms/{quote(context.room_id, safe='')}"
            f"/event/{quote(event_id, safe='')}"
        )
        raw = _raw_event(await asyncio.wait_for(context.client.api.request(Method.GET, path), timeout=10.0))
        context.dependencies.update(cache.retain_events(context.room_id, [raw]))
        await context.check_access()
    except _InspectionRejected:
        raise
    except Exception as exc:
        return MatrixReadEvent(raw, None, {
            "event_id": event_id, "error": f"Matrix event read failed: {type(exc).__name__}",
        }, None, cache.history_entry(context.room_id, event_id))
    if raw.get("event_id") != event_id:
        return MatrixReadEvent({"event_id": event_id, "room_id": context.room_id}, None, {
            "event_id": event_id, "error": "Matrix event was not returned",
        }, None, cache.history_entry(context.room_id, event_id))
    visible, error, replacement_id = await _visible_event(
        context.adapter, raw, context.room_id, context.chat_type, before=before,
    )
    await context.check_access()
    if visible is None and error is None:
        error = {"event_id": event_id, "error": "event has no visible message"}
    return MatrixReadEvent(raw, visible, error, replacement_id, cache.history_entry(context.room_id, event_id))


async def _inspect_state(context: _InspectionContext) -> dict[str, Any]:
    fields = {
        "name": ("m.room.name", "name"),
        "topic": ("m.room.topic", "topic"),
        "canonical_alias": ("m.room.canonical_alias", "alias"),
        "join_rule": ("m.room.join_rules", "join_rule"),
        "history_visibility": ("m.room.history_visibility", "history_visibility"),
        "encryption": ("m.room.encryption", "algorithm"),
    }
    result = {"room_id": context.room_id}
    for key, (event_type, field) in fields.items():
        result[key] = _text(await _state(context, event_type), field)
    return result


async def _inspect_members(context: _InspectionContext) -> dict[str, Any]:
    profiles = await context.request(lambda: context.client.get_joined_members(context.room_id))
    members = []
    for user_id, profile in sorted(profiles.items(), key=lambda item: str(item[0]))[:context.limit]:
        content = _content(profile)
        display_name = _text(content, "displayname") or getattr(profile, "displayname", None)
        avatar_url = _text(content, "avatar_url") or getattr(profile, "avatar_url", None)
        members.append({
            "user_id": str(user_id),
            "display_name": str(display_name)[:1200] if display_name else None,
            "avatar_url": str(avatar_url)[:1200] if avatar_url else None,
        })
    return {"members": members, "total": len(profiles), "truncated": len(profiles) > context.limit}


async def _inspect_permissions(context: _InspectionContext) -> dict[str, Any]:
    return await _permissions(context)


async def _inspect_pins(context: _InspectionContext) -> dict[str, Any]:
    cache = context.owner.cache
    assert cache is not None
    cached = cache.snapshot(context.room_id)
    pinned = await _state(context, "m.room.pinned_events")
    event_ids = pinned.get("pinned")
    if not isinstance(event_ids, list) or not all(isinstance(value, str) for value in event_ids):
        event_ids = []
    selected = event_ids[:context.limit]
    for event_id in selected:
        context.dependencies[event_id] = cached.get(event_id) or cache.retain(context.room_id, event_id)
    semaphore = asyncio.Semaphore(10)

    async def fetch(event_id: str) -> MatrixReadEvent:
        async with semaphore:
            return await _pinned_event(context, event_id)

    resolved = await asyncio.gather(*(fetch(event_id) for event_id in selected))
    for snapshot in resolved:
        await context.check_access()
        await snapshot.refresh(context.adapter, context.room_id, context.chat_type)
    await context.check_access()
    for snapshot in resolved:
        snapshot.recheck(context.adapter, context.room_id, context.chat_type)
    return {
        "events": [snapshot.visible for snapshot in resolved if snapshot.visible is not None],
        "total": len(event_ids), "truncated": len(event_ids) > context.limit,
        "errors": [snapshot.error for snapshot in resolved if snapshot.error is not None],
    }


_INSPECTION_HANDLERS: dict[str, Callable[[_InspectionContext], Awaitable[dict[str, Any]]]] = {
    "state": _inspect_state,
    "members": _inspect_members,
    "permissions": _inspect_permissions,
    "pins": _inspect_pins,
}


async def inspect_matrix_room(
    adapter: Any, kind: str, room_id: str, limit: int, *, requester: str,
) -> dict[str, Any]:
    handler = _INSPECTION_HANDLERS.get(kind)
    if handler is None:
        return {"error": "kind must be state, members, permissions, or pins"}
    owner = _InspectionOwner.capture(adapter)
    try:
        chat_type = await owner.access(room_id, requester)
        context = _InspectionContext(adapter, owner.client, room_id, chat_type, requester, limit, owner)
        return await handler(context)
    except _InspectionRejected as exc:
        return exc.error
    except Exception as exc:
        return {"error": f"Matrix room inspection failed: {type(exc).__name__}"}
