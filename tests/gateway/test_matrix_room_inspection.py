"""Matrix room inspection reads current state through the receiving adapter."""

import asyncio
import gc
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.room_inspection import inspect_matrix_room
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache


@pytest.mark.asyncio
async def test_room_inspection_reports_state_members_permissions_and_pins():
    state = {
        "m.room.name": {"name": "Planning"},
        "m.room.topic": {"topic": "Release notes"},
        "m.room.canonical_alias": {"alias": "#planning:server"},
        "m.room.join_rules": {"join_rule": "invite"},
        "m.room.history_visibility": {"history_visibility": "shared"},
        "m.room.encryption": {"algorithm": "m.megolm.v1.aes-sha2"},
        "m.room.power_levels": {
            "users": {"@bot:server": 50, "@alice:server": 100},
            "users_default": 0,
            "state_default": 50,
            "events": {"m.room.pinned_events": 75},
            "invite": 50,
        },
        "m.room.create": {"creator": "@alice:server", "room_version": "10"},
        "m.room.pinned_events": {"pinned": ["$first", "$second"]},
    }

    async def get_state_event(room_id, event_type, **kwargs):
        return state[event_type]

    async def get_event(room_id, event_id):
        return {"event_id": event_id, "sender": "@alice:server", "type": "m.room.message",
                "content": {"msgtype": "m.text", "body": f"Pinned {event_id}"}}

    async def request(_method, path, **kwargs):
        return await get_event("!room:server", "$first")

    client = SimpleNamespace(
        get_state_event=AsyncMock(side_effect=get_state_event),
        get_joined_members=AsyncMock(return_value={
            "@bot:server": SimpleNamespace(displayname="Hermes", avatar_url=None),
            "@alice:server": SimpleNamespace(displayname="Alice", avatar_url="mxc://server/alice"),
        }),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        get_event=AsyncMock(side_effect=get_event), crypto=None,
    )
    adapter = SimpleNamespace(
        _event_context_cache=MatrixEventContextCache(),
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda user, **kw: user == "@alice:server",
    )

    result = {
        kind: await inspect_matrix_room(adapter, kind, "!room:server", 1, requester="@alice:server")
        for kind in ("state", "members", "permissions", "pins")
    }

    assert result == {
        "state": {"room_id": "!room:server", "name": "Planning", "topic": "Release notes",
                  "canonical_alias": "#planning:server", "join_rule": "invite",
                  "history_visibility": "shared", "encryption": "m.megolm.v1.aes-sha2"},
        "members": {"members": [{"user_id": "@alice:server", "display_name": "Alice",
                                 "avatar_url": "mxc://server/alice"}], "total": 2, "truncated": True},
        "permissions": {"requester": {"user_id": "@alice:server", "level": 100,
                                      "creator_override": False},
                        "bot": {"user_id": "@bot:server", "level": 50,
                                "creator_override": False},
                        "required": {"send_message": 0, "send_event_type": "m.room.encrypted",
                                     "edit_pins": 75, "invite": 50,
                                     "kick": 50, "ban": 50, "redact_other": 50},
                        "bot_can_edit_pins": False},
        "pins": {"events": [{"event_id": "$first", "sender": "@alice:server",
                             "body": "Pinned $first", "msgtype": "m.text", "thread_id": None,
                             "timestamp": None, "sender_authorized": True}],
                 "total": 2, "truncated": True, "errors": []},
    }


@pytest.mark.asyncio
async def test_room_inspection_rejects_unauthorized_requester_before_network():
    client = SimpleNamespace(get_state_event=AsyncMock(), get_joined_members=AsyncMock())
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: False,
    )

    result = await inspect_matrix_room(adapter, "state", "!room:server", 20,
                                       requester="@alice:server")

    assert result == {"error": "Matrix requester is not authorized for this room"}
    client.get_state_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_permissions_include_version_12_creator_override():
    async def get_state_event(room_id, event_type, **kwargs):
        if event_type == "m.room.create":
            assert kwargs == {"format": "event"}
            return {"sender": "@bot:server", "content": {"room_version": "12"}}
        if event_type == "m.room.encryption":
            return {"algorithm": "m.megolm.v1.aes-sha2"}
        return {"users_default": 0, "state_default": 50,
                "events": {"m.room.pinned_events": 75, "m.room.encrypted": 25}}

    adapter = SimpleNamespace(
        _client=SimpleNamespace(get_state_event=AsyncMock(side_effect=get_state_event)),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(adapter, "permissions", "!room:server", 20,
                                       requester="@alice:server")

    assert result == {
        "requester": {"user_id": "@alice:server", "level": 0, "creator_override": False},
        "bot": {"user_id": "@bot:server", "level": 0, "creator_override": True},
        "required": {"send_message": 25, "send_event_type": "m.room.encrypted",
                     "edit_pins": 75, "invite": 0,
                     "kick": 50, "ban": 50, "redact_other": 50},
        "bot_can_edit_pins": True,
    }


@pytest.mark.asyncio
async def test_old_room_power_levels_accept_numeric_strings():
    power = {
        "users": {"@bot:server": " +100 ", "@alice:server": "-5"},
        "events": {"m.room.pinned_events": " 075 ", "m.room.message": "+10"},
        "invite": " 25 ",
    }
    try:
        from mautrix.types import StateEvent
    except ImportError:
        pass
    else:
        power = StateEvent.deserialize_content({
            **power, "__mautrix_event_type": "m.room.power_levels",
        })

    async def get_state_event(room_id, event_type, **kwargs):
        if event_type == "m.room.create":
            return {"sender": "@alice:server", "content": {"room_version": "9"}}
        if event_type == "m.room.encryption":
            return {}
        return power

    adapter = SimpleNamespace(
        _client=SimpleNamespace(get_state_event=AsyncMock(side_effect=get_state_event)),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(
        adapter, "permissions", "!room:server", 20, requester="@alice:server"
    )

    assert result == {
        "requester": {"user_id": "@alice:server", "level": -5, "creator_override": False},
        "bot": {"user_id": "@bot:server", "level": 100, "creator_override": False},
        "required": {"send_message": 10, "send_event_type": "m.room.message",
                     "edit_pins": 75, "invite": 25,
                     "kick": 50, "ban": 50, "redact_other": 50},
        "bot_can_edit_pins": True,
    }


@pytest.mark.asyncio
async def test_version_12_ignores_legacy_creator_property():
    async def get_state_event(room_id, event_type, **kwargs):
        if event_type == "m.room.create":
            return {"sender": "@alice:server", "content": {
                "room_version": "12", "creator": "@bot:server",
            }}
        if event_type == "m.room.encryption":
            return {}
        return {"users_default": 0, "events": {"m.room.pinned_events": 75}}

    adapter = SimpleNamespace(
        _client=SimpleNamespace(get_state_event=AsyncMock(side_effect=get_state_event)),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(
        adapter, "permissions", "!room:server", 20, requester="@alice:server"
    )

    assert result == {
        "requester": {"user_id": "@alice:server", "level": 0, "creator_override": True},
        "bot": {"user_id": "@bot:server", "level": 0, "creator_override": False},
        "required": {"send_message": 0, "send_event_type": "m.room.message",
                     "edit_pins": 75, "invite": 0,
                     "kick": 50, "ban": 50, "redact_other": 50},
        "bot_can_edit_pins": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    "plain", "edited", "redacted", "encrypted", "encrypted-edited", "encrypted-redacted",
    "missing-key", "withdrawn-original", "withdrawn-edit",
])
async def test_pin_snapshots_expose_effective_state_after_sibling_await(state: str):
    mautrix_types = pytest.importorskip("mautrix.types")
    room, sender = "!room:server", "@alice:server"
    content = {"msgtype": "m.text", "body": "Original plan"}
    raw = {
        "room_id": room, "event_id": "$target", "sender": sender, "origin_server_ts": 1,
        "type": "m.room.message", "content": content,
    }
    encrypted = state.startswith("encrypted") or state == "missing-key"
    cipher = {
        "algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "original", "session_id": "session",
        "sender_key": "key", "device_id": "device",
    }
    if encrypted:
        raw.update(type="m.room.encrypted", content=cipher)
    edit_content = {
        "msgtype": "m.text", "body": "* Revised plan",
        "m.new_content": {"msgtype": "m.text", "body": "Revised plan"},
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
    }
    if state in {"edited", "encrypted-edited", "withdrawn-edit"}:
        replacement = {**raw, "event_id": "$edit", "content": edit_content}
        if encrypted:
            replacement["content"] = {**cipher, "ciphertext": "replacement",
                                      "m.relates_to": edit_content["m.relates_to"]}
        raw["unsigned"] = {"m.relations": {"m.replace": replacement}}
    if state in {"redacted", "encrypted-redacted"}:
        raw["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
    gate = {**raw, "event_id": "$gate", "type": "m.room.message",
            "content": {"msgtype": "m.text", "body": "Gate plan"}}
    gate.pop("unsigned", None)
    started, release = asyncio.Event(), asyncio.Event()

    async def request(_method, path, **_kwargs):
        event_id = "$gate" if path.endswith("%24gate") else "$target"
        if event_id == "$gate":
            started.set()
            await release.wait()
            return gate
        return raw

    class SessionNotFound(Exception):
        pass

    async def decrypt(event):
        if state == "missing-key":
            raise SessionNotFound()
        clear = edit_content if event.event_id == "$edit" else content
        return mautrix_types.Event.deserialize({
            **raw, "event_id": str(event.event_id), "type": "m.room.message",
            "content": json.loads(json.dumps(clear)),
        })

    plaintext = json.dumps({"room_id": room, "type": "m.room.message", "content": edit_content})
    store = SimpleNamespace(get_group_session=AsyncMock(return_value=SimpleNamespace(
        decrypt=lambda _ciphertext: (plaintext, 0),
    )))
    async def get_event(_room, event_id):
        return await request(None, "/event/" + event_id.replace("$", "%24"))

    client = SimpleNamespace(
        api=SimpleNamespace(request=request),
        get_state_event=AsyncMock(return_value={"pinned": ["$target", "$gate"]}),
        get_event=AsyncMock(side_effect=get_event),
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt, crypto_store=store),
    )
    cache = MatrixEventContextCache(max_entries=1)
    adapter = SimpleNamespace(
        _client=client, _event_context_cache=cache, _joined_rooms={room}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True), _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    pending = asyncio.create_task(inspect_matrix_room(adapter, "pins", room, 2, requester=sender))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        withdrawals = {"withdrawn-original": "$target", "withdrawn-edit": "$edit"}
        if state in withdrawals:
            cache.redact(room, withdrawals[state])
            cache.store(room, "$pressure", MatrixEventContext(sender, "Pressure"))
            gc.collect()
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    target = {
        "event_id": "$target", "sender": sender, "body": "Original plan", "msgtype": "m.text",
        "thread_id": None, "timestamp": 1, "sender_authorized": True,
    }
    if state in {"edited", "encrypted-edited"}:
        target.update(body="Revised plan", edited=True)
    if state in {"redacted", "encrypted-redacted", "withdrawn-original"}:
        target.update(body="[redacted]", msgtype=None, redacted=True)
    if state == "withdrawn-edit":
        target.update(body="[event content unavailable]", msgtype=None)
    errors = {
        "missing-key": [{"event_id": "$target", "error": "missing decryption keys"}],
        "withdrawn-edit": [{"event_id": "$target", "error": "replacement was redacted"}],
    }.get(state, [])
    assert result == {
        "events": ([] if state == "missing-key" else [target]) + [{
            "event_id": "$gate", "sender": sender, "body": "Gate plan", "msgtype": "m.text",
            "thread_id": None, "timestamp": 1, "sender_authorized": True,
        }],
        "total": 2, "truncated": False, "errors": errors,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "change"), [
    (kind, change)
    for kind in ("state", "members", "permissions", "pins")
    for change in (
        "unchanged", "closing", "client", "account", "api", "api-token", "http-session",
        "home", "transport", "actor", "room", "policy", "chat-type",
    )
] + [("state", "policy-last-missing-state")])
async def test_inspection_rechecks_owning_profile_session_and_policy_after_await(
    tmp_path, monkeypatch, kind: str, change: str,
):
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_constants import set_hermes_home_override
    from tools.matrix_read_tool import _matrix_read

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    room, user = "!room:server", "@alice:server"
    started, release = asyncio.Event(), asyncio.Event()
    waiting = True
    mutation = lambda: None

    def make_adapter(label):
        async def network_barrier(event_type=None):
            nonlocal waiting
            final_missing = change == "policy-last-missing-state"
            if label == "A" and waiting and (not final_missing or event_type == "m.room.encryption"):
                waiting = False
                started.set()
                await release.wait()
                mutation()

        async def state(_room, event_type, **_kwargs):
            await network_barrier(event_type)
            if change == "policy-last-missing-state" and event_type == "m.room.encryption":
                class MNotFound(Exception):
                    pass
                raise MNotFound()
            return {
                "m.room.name": {"name": label},
                "m.room.create": {"sender": user, "content": {"room_version": "10"}},
                "m.room.pinned_events": {"pinned": ["$pin"]},
            }.get(event_type, {})

        async def members(_room):
            await network_barrier()
            return {user: {"displayname": label}}

        async def request(*_args, **_kwargs):
            return {
                "room_id": room, "event_id": "$pin", "sender": user,
                "type": "m.room.message", "content": {"msgtype": "m.text", "body": label},
            }

        client = SimpleNamespace(
            api=SimpleNamespace(request=request, base_url="https://server", token=label, session=object()),
            mxid=f"@bot-{label}:server", device_id=f"device-{label}", crypto=None,
            get_state_event=state, get_joined_members=members,
            get_event=AsyncMock(return_value=awaitable_pin(label)),
        )
        adapter = SimpleNamespace(
            _client=client, _event_context_cache=MatrixEventContextCache(), _joined_rooms={room},
            _user_id=client.mxid, _closing=False, _allowed=True, _dm=False,
            _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        )
        adapter._is_dm_room = AsyncMock(side_effect=lambda _room: adapter._dm)
        adapter._is_sender_authorized = lambda actor, **_kw: actor == user and adapter._allowed
        adapter.inspect_matrix_room = lambda *args, **kwargs: inspect_matrix_room(adapter, *args, **kwargs)
        return adapter

    def awaitable_pin(label):
        return {"room_id": room, "event_id": "$pin", "sender": user,
                "type": "m.room.message", "content": {"msgtype": "m.text", "body": label}}

    first, second = make_adapter("A"), make_adapter("B")
    first_client = first._client
    first_api = first_client.api
    first_http_session = first_api.session
    home_a, home_b = tmp_path / "A", tmp_path / "B"
    home_a.mkdir()
    home_b.mkdir()

    async def run(adapter, home):
        with _profile_runtime_scope(home, {}):
            tokens = set_session_vars(
                platform="matrix", chat_id=room, user_id=user, profile=home.name,
                session_key=f"matrix-{home.name}", transport_adapter=adapter,
            )
            try:
                return json.loads(await _matrix_read({"kind": kind}))
            finally:
                clear_session_vars(tokens)

    changes = {
        "unchanged": lambda: None,
        "closing": lambda: setattr(first, "_closing", True),
        "client": lambda: setattr(first, "_client", second._client),
        "account": lambda: setattr(first_client, "mxid", "@different:server"),
        "api": lambda: setattr(first_client, "api", second._client.api),
        "api-token": lambda: setattr(first_client.api, "token", "changed"),
        "http-session": lambda: setattr(first_client.api, "session", object()),
        "home": lambda: set_hermes_home_override(home_b),
        "transport": lambda: set_session_vars(
            platform="matrix", chat_id=room, user_id=user, profile="A",
            session_key="matrix-A", transport_adapter=second,
        ),
        "actor": lambda: set_session_vars(
            platform="matrix", chat_id=room, user_id="@different:server", profile="A",
            session_key="matrix-A", transport_adapter=first,
        ),
        "room": lambda: first._joined_rooms.clear(),
        "policy": lambda: setattr(first, "_allowed", False),
        "policy-last-missing-state": lambda: setattr(first, "_allowed", False),
        "chat-type": lambda: setattr(first, "_dm", True),
    }
    mutation = changes[change]
    pending = asyncio.create_task(run(first, home_a))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        result_b = await asyncio.wait_for(run(second, home_b), timeout=2)
    finally:
        release.set()
    result_a = await asyncio.wait_for(pending, timeout=2)
    first._client = first_client
    first._closing = False
    first._allowed = True
    first._dm = False
    first._joined_rooms.add(room)
    first_client.mxid = "@bot-A:server"
    first_client.api = first_api
    first_api.token = "A"
    first_api.session = first_http_session
    again_a = await asyncio.wait_for(run(first, home_a), timeout=2)

    def expected(label):
        return {
            "state": {"room_id": room, "name": label, "topic": None, "canonical_alias": None,
                      "join_rule": None, "history_visibility": None, "encryption": None},
            "members": {"members": [{"user_id": user, "display_name": label, "avatar_url": None}],
                        "total": 1, "truncated": False},
            "permissions": {
                "requester": {"user_id": user, "level": 100, "creator_override": False},
                "bot": {"user_id": f"@bot-{label}:server", "level": 0, "creator_override": False},
                "required": {"send_message": 0, "send_event_type": "m.room.message", "edit_pins": 50,
                             "invite": 0, "kick": 50, "ban": 50, "redact_other": 50},
                "bot_can_edit_pins": False,
            },
            "pins": {"events": [{"event_id": "$pin", "sender": user, "body": label,
                                 "msgtype": "m.text", "thread_id": None, "timestamp": None,
                                 "sender_authorized": True}],
                     "total": 1, "truncated": False, "errors": []},
        }[kind]

    refusals = {
        "closing": {"error": "Matrix client is disconnected"},
        "room": {"error": "Matrix room is not allowed or joined"},
        "policy": {"error": "Matrix requester is not authorized for this room"},
        "policy-last-missing-state": {"error": "Matrix requester is not authorized for this room"},
    }
    expected_a = expected("A") if change == "unchanged" else refusals.get(change, {
        "error": "Matrix room inspection context changed",
    })
    assert (result_a, result_b, again_a) == (expected_a, expected("B"), expected("A"))
