"""A separate Matrix client verifies model-visible reads through the live gateway."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

from nio import RoomMessageText, RoomPutStateResponse, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


def test_model_reads_an_event_from_its_live_matrix_session(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=15)

            async def send_and_wait(body: str) -> str:
                sent = await client.room_send(
                    live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent

                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if not joined:
                        continue
                    for event in joined.timeline.events:
                        if (
                            isinstance(event, RoomMessageText)
                            and event.sender == live_room.bot.user_id
                            and event.event_id not in seen
                        ):
                            seen.add(event.event_id)
                            return sent.event_id

            target = await asyncio.wait_for(send_and_wait("Read target [history:blue]"), timeout=15)
            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix read event"]}),
                ToolCall("tool_call", {"calls": [{
                    "name": "matrix_read", "arguments": {"kind": "event", "event_id": target},
                }]}),
                Text("Read complete"),
            )
            await asyncio.wait_for(send_and_wait("Read the earlier Matrix event"), timeout=15)

            requests = gateway.model.main_requests()
            assert len(requests) == 4
            search_messages = [message for message in requests[2]["messages"] if message["role"] == "tool"]
            assert len(search_messages) == 1
            assert "matrix_read" in json.loads(search_messages[0]["content"])["tools"]
            tool_messages = [message for message in requests[3]["messages"] if message["role"] == "tool"]
            assert len(tool_messages) == 2
            result = json.loads(tool_messages[1]["content"])
            assert isinstance(result["events"][0]["timestamp"], int)
            assert {**result, "events": [{**result["events"][0], "timestamp": None}]} == {
                "events": [{
                    "event_id": target,
                    "sender": live_room.observer.user_id,
                    "body": "Read target [history:blue]",
                    "msgtype": "m.text",
                    "thread_id": None,
                    "timestamp": None,
                    "sender_authorized": True,
                }],
                "errors": [],
            }
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_model_inspects_live_room_state_members_permissions_and_pins(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        bot_client = live_room.bot.client(live_room.homeserver)
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=15)

            async def send_and_wait(body: str, expected_reply: str) -> None:
                sent = await client.room_send(
                    live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent
                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined and any(
                        isinstance(event, RoomMessageText)
                        and event.sender == live_room.bot.user_id
                        and event.body == expected_reply
                        for event in joined.timeline.events
                    ):
                        return

            await asyncio.wait_for(send_and_wait("Start room inspection", "Matrix live reply"), timeout=15)
            topic = await client.room_put_state(
                live_room.room_id, "m.room.topic", {"topic": "Release notes"},
            )
            assert isinstance(topic, RoomPutStateResponse), topic
            pinned = await bot_client.room_send(
                live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": "Pinned plan"},
            )
            assert isinstance(pinned, RoomSendResponse), pinned
            pin_state = await client.room_put_state(
                live_room.room_id, "m.room.pinned_events", {"pinned": [pinned.event_id]},
            )
            assert isinstance(pin_state, RoomPutStateResponse), pin_state

            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix room state members permissions pins"]}),
                ToolCall(
                    "tool_call", {"calls": [{"name": "matrix_read", "arguments": {"kind": "state"}}]},
                    parallel=[
                        ("tool_call", {"calls": [{"name": "matrix_read", "arguments": {"kind": kind}}]})
                        for kind in ("members", "permissions", "pins")
                    ],
                ),
                Text("Inspection complete"),
            )
            await asyncio.wait_for(send_and_wait("Inspect this room", "Inspection complete"), timeout=20)

            requests = gateway.model.main_requests()
            assert len(requests) == 4
            results = [
                json.loads(message["content"])
                for message in requests[3]["messages"] if message["role"] == "tool"
            ]
            assert len(results) == 5
            inspected = results[1:]
            assert inspected[0]["topic"] == "Release notes"
            assert {member["user_id"] for member in inspected[1]["members"]} == {
                live_room.observer.user_id, live_room.bot.user_id,
            }
            assert inspected[1]["total"] == 2
            assert inspected[2]["bot"]["user_id"] == live_room.bot.user_id
            assert inspected[3]["events"][0]["body"] == "Pinned plan"
            assert inspected[3]["events"][0]["event_id"] == pinned.event_id
        finally:
            await client.close()
            await bot_client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=45))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
