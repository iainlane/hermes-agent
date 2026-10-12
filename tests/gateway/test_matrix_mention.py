"""Tests for Matrix require-mention gating and auto-thread features."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

# The matrix adapter module is importable without mautrix installed
# (module-level imports use try/except with stubs).  No need for
# module-level mock installation — tests that call adapter methods
# needing real mautrix APIs mock them individually.


def _make_adapter(tmp_path=None):
    """Create a MatrixAdapter with mocked config."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10  # avoid startup grace filter
    return adapter


def _set_dm(adapter, room_id="!room1:example.org", is_dm=True):
    """Mark a room as DM (or not) in the adapter's cache."""
    adapter._dm_rooms[room_id] = is_dm


def _make_event(
    body,
    sender="@alice:example.org",
    event_id="$evt1",
    room_id="!room1:example.org",
    formatted_body=None,
    thread_id=None,
    mention_user_ids=None,
):
    """Create a fake room message event.

    The mautrix adapter reads ``event.room_id``, ``event.sender``,
    ``event.event_id``, ``event.timestamp``, and ``event.content``
    (a dict with ``msgtype``, ``body``, etc.).
    """
    content = {"body": body, "msgtype": "m.text"}
    if formatted_body:
        content["formatted_body"] = formatted_body
        content["format"] = "org.matrix.custom.html"

    if mention_user_ids is not None:
        content["m.mentions"] = {"user_ids": mention_user_ids}

    relates_to = {}
    if thread_id:
        relates_to["rel_type"] = "m.thread"
        relates_to["event_id"] = thread_id
    if relates_to:
        content["m.relates_to"] = relates_to

    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id=room_id,
        timestamp=int(time.time() * 1000),
        content=content,
    )


# ---------------------------------------------------------------------------
# Mention detection helpers
# ---------------------------------------------------------------------------


class TestIsBotMentioned:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_full_user_id_in_body(self):
        assert self.adapter._is_bot_mentioned("hey @hermes:example.org help")

    def test_localpart_in_body(self):
        assert self.adapter._is_bot_mentioned("hermes can you help?")


    def test_matrix_pill_in_formatted_body(self):
        html = '<a href="https://matrix.to/#/@hermes:example.org">Hermes</a> help'
        assert self.adapter._is_bot_mentioned("Hermes help", html)


    # m.mentions.user_ids — MSC3952 / Matrix v1.7 authoritative mentions
    # Ported from openclaw/openclaw#64796



class TestStripMention:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_strip_full_user_id(self):
        result = self.adapter._strip_mention("@hermes:example.org help me")
        assert result == "help me"

    def test_localpart_preserved(self):
        """Bare localpart (no @) is preserved — avoids false positives in paths."""
        result = self.adapter._strip_mention("hermes help me")
        assert result == "hermes help me"


# ---------------------------------------------------------------------------
# Outbound mention payloads
# ---------------------------------------------------------------------------


class TestOutboundMentions:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mock_client = MagicMock()
        self.mock_client.send_message_event = AsyncMock(return_value="$evt1")
        self.adapter._client = self.mock_client

    @staticmethod
    def _sent_content(mock_client):
        call_args = mock_client.send_message_event.call_args
        return call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_send_adds_matrix_mentions_and_formatted_body(self):
        result = await self.adapter.send(
            "!room1:example.org",
            "Hello @alice:example.org, please check this.",
        )

        assert result.success is True
        content = self._sent_content(self.mock_client)
        assert content["m.mentions"] == {"user_ids": ["@alice:example.org"]}
        assert content["formatted_body"] == (
            'Hello <a href="https://matrix.to/#/@alice:example.org">'
            "@alice:example.org</a>, please check this."
        )


# ---------------------------------------------------------------------------
# Require-mention gating in _on_room_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_mention_default_processes_mentioned(monkeypatch):
    """Default: messages with mention are processed, mention stripped."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "help me"


@pytest.mark.asyncio
async def test_require_mention_m_mentions_user_ids(monkeypatch):
    """m.mentions.user_ids is authoritative per MSC3952 — no body mention needed.

    Ported from openclaw/openclaw#64796.
    """
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    # Body has NO mention, but m.mentions.user_ids includes the bot.
    event = _make_event(
        "please reply",
        mention_user_ids=["@hermes:example.org"],
    )

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_m_mentions_other_user_ignored(monkeypatch):
    """m.mentions.user_ids mentioning another user should NOT activate the bot."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event(
        "hey alice check this",
        mention_user_ids=["@alice:example.org"],
    )

    await adapter._on_room_message(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_strips_full_mxid(monkeypatch):
    """DMs strip the full MXID from body when require_mention is on (default)."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "help me"


@pytest.mark.asyncio
async def test_bare_mention_passes_empty_string(monkeypatch):
    """A message that is only a mention should pass through as empty, not be dropped."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("@hermes:example.org")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mention_room, mention_body, claims, same_sync_batch", [
    ("!room1:example.org", "@hermes:example.org", True, False),
    ("!room2:example.org", "@hermes:example.org", False, False),
    ("!room1:example.org", "@hermes:example.org hi", False, False),
    ("!room1:example.org", "@hermes:example.org", True, True),
    ("!room1:example.org", "@hermes:example.org", True, "two_voices"),
])
async def test_bare_mention_claims_parked_voice_only_in_same_room(
        monkeypatch, mention_room, mention_body, claims, same_sync_batch):
    """An unmentioned MSC3245 voice (empty m.mentions) is answered by the sender's bare @mention
    typed right after it in the SAME room; a bare mention in another room never pulls it across,
    and a mention carrying text is answered as that text. mautrix runs one /sync batch's events as
    concurrent tasks, so the claim must also win while the voice still awaits a room-identity fetch.
    ``two_voices``: batch [voice (slow gate), mention, voice2 (fast)] then mention2 -- each mention
    answers the voice sent before it, even though voice2 parks first, and nothing stays parked."""
    import asyncio

    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/voice.ogg")
    adapter._background_read_receipt = MagicMock()
    voice = _make_event("voice message", event_id="$voice")
    voice.content.update({"msgtype": "m.audio", "url": "mxc://example.org/v", "info": {"mimetype": "audio/ogg"},
                          "org.matrix.msc3245.voice": {}, "m.mentions": {}})
    mention = _make_event(mention_body, event_id="$text", room_id=mention_room,
                          mention_user_ids=["@hermes:example.org"])

    if same_sync_batch:
        resolve_identity = adapter._resolve_room_identity
        delays = [0.1] if same_sync_batch == "two_voices" else []

        async def slow_identity(room_id):  # stale 60s cache -> homeserver round-trip
            await asyncio.sleep(delays.pop(0) if delays else 0.01)
            return await resolve_identity(room_id)
        adapter._resolve_room_identity = slow_identity
        batch = [voice, mention]
        if same_sync_batch == "two_voices":
            voice2 = _make_event("voice message", event_id="$voice2")
            voice2.content.update({k: voice.content[k] for k in (
                "msgtype", "url", "info", "org.matrix.msc3245.voice", "m.mentions")})
            batch.append(voice2)
        await asyncio.gather(*(adapter._on_room_message(e) for e in batch))
        if same_sync_batch == "two_voices":
            await adapter._on_room_message(_make_event(
                "@hermes:example.org", event_id="$text2", mention_user_ids=["@hermes:example.org"]))
            dispatched = [m.args[0].message_id for m in adapter.handle_message.await_args_list]
            assert dispatched == ["$voice", "$voice2"]
            assert not adapter._parked_voices._parked and not adapter._parked_voices._inflight
            return
    else:
        await adapter._on_room_message(voice)
        adapter.handle_message.assert_not_awaited()
        adapter._download_and_cache_media.assert_not_awaited()  # parked voice is never downloaded
        await adapter._on_room_message(mention)

    dispatched = [(m.args[0].source.chat_id, m.args[0].message_id) for m in adapter.handle_message.await_args_list]
    assert dispatched == ([("!room1:example.org", "$voice")] if claims else [(mention_room, "$text")])
    if claims:  # the bare mention is the newest event; the read marker must reach it
        adapter._background_read_receipt.assert_any_call("!room1:example.org", "$text")


# ---------------------------------------------------------------------------
# Auto-thread in _on_room_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_thread_preserves_existing_thread(monkeypatch):
    """If message is already in a thread, thread_id is not overridden."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    adapter._threads.mark("$thread_root")
    event = _make_event("reply in thread", thread_id="$thread_root")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$thread_root"


@pytest.mark.asyncio
async def test_auto_thread_skips_dm(monkeypatch):
    """DMs should not get auto-threaded."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("hello dm", event_id="$dm1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


# ---------------------------------------------------------------------------
# Thread persistence
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# DM mention-thread feature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_mention_thread_creates_thread(monkeypatch):
    """MATRIX_DM_MENTION_THREADS=true: DM with @mention creates a thread."""
    monkeypatch.setenv("MATRIX_DM_MENTION_THREADS", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help me", event_id="$dm1")

    with patch.object(adapter._threads, "_save"):
        await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$dm1"
    assert msg.text == "help me"


@pytest.mark.parametrize("existing", [None, "operator"])
@pytest.mark.parametrize("secondary", [False, True])
def test_yaml_bridge_respects_scope_and_existing_env(monkeypatch, existing, secondary):
    import os
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from plugins.platforms.matrix.adapter import _apply_yaml_config

    expected = {"MATRIX_REQUIRE_MENTION": "false", "MATRIX_AUTO_THREAD": "false",
                "MATRIX_FREE_RESPONSE_ROOMS": "!one:example.org,!two:example.org"}
    config = {"require_mention": False, "auto_thread": False,
              "free_response_rooms": ["!one:example.org", "!two:example.org"]}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
        if existing is not None:
            monkeypatch.setenv(key, existing)
    token = set_secret_scope({}) if secondary else None
    set_multiplex_active(secondary)
    try:
        assert _apply_yaml_config({"matrix": config}, config) == config
        assert {key: os.environ.get(key) for key in expected} == (
            dict.fromkeys(expected, existing) if secondary or existing else expected
        )
    finally:
        if token is not None:
            reset_secret_scope(token)
        set_multiplex_active(False)


_OUTBOUND_RENDER_CASES = [
    pytest.param(
        "[Do not eat eggs stamped `3UA1012066`](https://example.org/recall)",
        "[Do not eat eggs stamped `3UA1012066`](https://example.org/recall)",
        '<a href="https://example.org/recall">Do not eat eggs stamped <code>3UA1012066</code></a>',
        None,
        None,
        False,
        id="egg-recall",
    ),
    pytest.param(
        "[`feat/minimal-rewrite`](https://github.com/example/project/tree/feat/minimal-rewrite)",
        "[`feat/minimal-rewrite`](https://github.com/example/project/tree/feat/minimal-rewrite)",
        '<a href="https://github.com/example/project/tree/feat/minimal-rewrite"><code>feat/minimal-rewrite</code></a>',
        None,
        None,
        False,
        id="original-link-label",
    ),
    pytest.param(
        "[`code`](https://example.com)",
        "[`code`](https://example.com)",
        '<a href="https://example.com"><code>code</code></a>',
        None,
        None,
        False,
        id="original-fallback-label",
    ),
    pytest.param(
        "[`one`](https://example.org/one) and [two `three`](https://example.org/two) plus `four`",
        "[`one`](https://example.org/one) and [two `three`](https://example.org/two) plus `four`",
        '<a href="https://example.org/one"><code>one</code></a> and <a href="https://example.org/two">two <code>three</code></a> plus <code>four</code>',
        None,
        None,
        False,
        id="multiple-nested-spans",
    ),
    pytest.param("bare text", "bare text", None, None, None, False, id="plain"),
    pytest.param(
        "`standalone`",
        "`standalone`",
        "<code>standalone</code>",
        None,
        None,
        False,
        id="code",
    ),
    pytest.param(
        "[ordinary](https://example.org)",
        "[ordinary](https://example.org)",
        '<a href="https://example.org">ordinary</a>',
        None,
        None,
        False,
        id="ordinary-link",
    ),
    pytest.param(
        "@bob:example.org [`@alice:example.org`](https://example.org) `@carol:example.org` @bob:example.org @dave:example.org",
        "[@bob:example.org](https://matrix.to/#/@bob:example.org) [`@alice:example.org`](https://example.org) `@carol:example.org` [@bob:example.org](https://matrix.to/#/@bob:example.org) [@dave:example.org](https://matrix.to/#/@dave:example.org)",
        '<a href="https://matrix.to/#/@bob:example.org">@bob:example.org</a> <a href="https://example.org"><code>@alice:example.org</code></a> <code>@carol:example.org</code> <a href="https://matrix.to/#/@bob:example.org">@bob:example.org</a> <a href="https://matrix.to/#/@dave:example.org">@dave:example.org</a>',
        None,
        {"user_ids": ["@bob:example.org", "@dave:example.org"]},
        False,
        id="users-and-deduplication",
    ),
    pytest.param(
        "`@alice:example.org @room` [@carol:example.org @room](https://example.org)",
        "`@alice:example.org @room` [@carol:example.org @room](https://example.org)",
        '<code>@alice:example.org @room</code> <a href="https://example.org">@carol:example.org @room</a>',
        None,
        None,
        True,
        id="protected-users-and-room",
    ),
    *[
        pytest.param(
            "@room `@room` [@room](https://example.org)",
            "@room `@room` [@room](https://example.org)",
            '@room <code>@room</code> <a href="https://example.org">@room</a>',
            None,
            {"room": True} if enabled else None,
            enabled,
            id="room-enabled" if enabled else "room-disabled",
        )
        for enabled in (False, True)
    ],
    pytest.param(
        "```\n@alice:example.org @room\n```\n@bob:example.org",
        "```\n@alice:example.org @room\n```\n[@bob:example.org](https://matrix.to/#/@bob:example.org)",
        '<pre><code>@alice:example.org @room\n</code></pre>\n<a href="https://matrix.to/#/@bob:example.org">@bob:example.org</a>',
        '<pre><code>@alice:example.org @room\n</code></pre><br>\n<a href="https://matrix.to/#/@bob:example.org">@bob:example.org</a>',
        {"user_ids": ["@bob:example.org"]},
        True,
        id="fenced-mentions",
    ),
]


@pytest.mark.parametrize(
    "text,linked,html,fallback_html,mentions,allow_room", _OUTBOUND_RENDER_CASES
)
def test_outbound_mention_injection_preserves_protected_regions(
    text, linked, html, fallback_html, mentions, allow_room
):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    assert adapter._inject_outbound_mention_links(text) == linked


@pytest.mark.parametrize("fallback", [False, True], ids=["markdown", "fallback"])
@pytest.mark.parametrize(
    "text,linked,html,fallback_html,mentions,allow_room", _OUTBOUND_RENDER_CASES
)
def test_outbound_payload_preserves_rendering_and_notification_scope(
    text, linked, html, fallback_html, mentions, allow_room, fallback, monkeypatch
):
    import sys

    from plugins.platforms.matrix.adapter import MatrixAdapter

    if fallback:
        monkeypatch.setitem(sys.modules, "markdown", None)
    else:
        import markdown

        assert markdown.Markdown is not None

    adapter = object.__new__(MatrixAdapter)
    adapter._allow_room_mentions = allow_room
    expected = {"msgtype": "m.text", "body": text}
    if mentions is not None:
        expected["m.mentions"] = mentions
    formatted = fallback_html if fallback and fallback_html is not None else html
    if formatted is not None:
        expected.update(format="org.matrix.custom.html", formatted_body=formatted)
    assert adapter._build_text_message_content(text) == expected
