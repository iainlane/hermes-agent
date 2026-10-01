"""Reaction feedback stays on the original Matrix control card."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", ["$root", ""])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize(("card", "reactor", "key", "expired", "outcome"), [
    pytest.param("approval", "@owner:example.org", "\U0001f44d", False, None, id="approval-invalid-key"),
    pytest.param("approval", "@stranger:example.org", "✅", False, None, id="approval-unauthorized"),
    pytest.param("approval", "@other:example.org", "✅", False, None, id="approval-not-requester"),
    pytest.param("model", "@owner:example.org", "1️⃣", True, None, id="model-picker-expired"),
    pytest.param("choice", "@owner:example.org", "1️⃣", False, "Applied", id="choice-picker-confirmed"),
    pytest.param("choice", "@owner:example.org", "1️⃣", False, RuntimeError("boom"),
                 id="choice-picker-failed"),
])
async def test_feedback_on_a_threaded_card_stays_in_its_thread(
        monkeypatch, tmp_path, thread_id, interleaved, card, reactor, key, expired, outcome):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token", extra={
            "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
        },
    ))
    adapter._allowed_user_ids = {"@owner:example.org", "@other:example.org"}
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter._schedule_reaction_redaction = lambda *args, **kwargs: None
    adapter.redact_message = AsyncMock(return_value=True)
    sent = []

    async def send_message_event(room_id, event_type, content):
        sent.append(content)
        return f"$event-{len(sent)}"

    adapter._client = MagicMock()
    adapter._client.send_message_event = send_message_event

    async def on_selected(*_args):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    from plugins.platforms.matrix.followup_mixin import _MatrixFollowupChoice
    from plugins.platforms.matrix.reaction_followups import PendingFollowupReaction, PendingFollowupReactions

    action = _MatrixFollowupChoice(
        "turn", (), "!room:example.org", "@owner:example.org", thread_id, "", "s",
        pending=PendingFollowupReactions(clock=lambda: 0),
    )
    adapter._store_dir = tmp_path / "store"
    adapter._reaction_followup_actions = {"s": action}
    metadata = {"thread_id": thread_id, "requester_user_id": "@owner:example.org"}
    entry = None
    if card == "approval":
        from tools import approval
        from tools.approval_gateway_wait import _ApprovalEntry
        entry = _ApprovalEntry({"command": "rm -rf /tmp/card"})
        monkeypatch.setitem(approval._gateway_queues, "s", [entry])
        metadata.update(entry.data)
        await adapter.send_exec_approval(
            chat_id="!room:example.org", command="rm -rf /tmp/card", session_key="s", metadata=metadata)
        registry = adapter._approval_prompts_by_event
    elif card == "model":
        await adapter.send_model_picker(
            "!room:example.org", [{"slug": "p", "name": "P", "models": ["m"]}], "m", "p", "s",
            on_selected, metadata=metadata)
        registry = adapter._model_picker_prompts_by_event
    else:
        await adapter.send_choice_picker(
            "!room:example.org", "Pick", [{"value": "v", "label": "V"}], "s", on_selected, metadata=metadata)
        registry = adapter._choice_picker_prompts_by_event
    if expired:
        registry["$event-1"].expires_at = 0

    if interleaved:
        await adapter.send("!room:example.org", "Later output", metadata={"thread_id": thread_id})
    feedback_start = len(sent)

    await adapter._on_reaction(types.SimpleNamespace(
        sender=reactor, event_id="$reaction", room_id="!room:example.org",
        content={"m.relates_to": {"event_id": "$event-1", "key": key}}))

    relations = [content["m.relates_to"] for content in sent[feedback_start:]]
    expected_relation = {"m.in_reply_to": {"event_id": "$event-1"}}
    if thread_id:
        expected_relation.update(rel_type="m.thread", event_id=thread_id, is_falling_back=True)
    assert (relations, action.pending.events) == ([expected_relation], {})

    await adapter._on_reaction(types.SimpleNamespace(
        sender="@owner:example.org", event_id="$ordinary-reaction", room_id="!room:example.org",
        content={"m.relates_to": {"event_id": "$ordinary", "key": "👍"}},
    ))
    assert action.pending.events == {
        "$ordinary-reaction": PendingFollowupReaction(
            "$ordinary", "👍", "@owner:example.org", "$ordinary-reaction", 10,
        ),
    }
    if entry is not None:
        approval._gateway_queues.pop("s", None)
