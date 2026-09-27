"""Matrix room inspection reads current state through the receiving adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.room_inspection import inspect_matrix_room


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

    client = SimpleNamespace(
        get_state_event=AsyncMock(side_effect=get_state_event),
        get_joined_members=AsyncMock(return_value={
            "@bot:server": SimpleNamespace(displayname="Hermes", avatar_url=None),
            "@alice:server": SimpleNamespace(displayname="Alice", avatar_url="mxc://server/alice"),
        }),
        get_event=AsyncMock(side_effect=get_event), crypto=None,
    )
    adapter = SimpleNamespace(
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
