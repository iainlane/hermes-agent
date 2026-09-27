"""Matrix room and thread context for gateway turns."""

from __future__ import annotations

from urllib.parse import quote, urlencode

from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Optional
from contextlib import suppress
import asyncio
import hashlib
import json
import logging
import time

from gateway.inbound_context import InboundContextSnapshot
from gateway.platforms.event import MessageEvent
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContextCache
from plugins.platforms.matrix.room_context import MatrixHistoryContext, MatrixRoomIdentity, MatrixRoomState, fetch_room_entries
from plugins.platforms.matrix.discovery import discover_matrix
from plugins.platforms.matrix.unread import ReadProgress, mark_matrix_read, read_matrix_unread
from plugins.platforms.matrix.room_access import (
    LiveRoomClient, RoomClassificationUnavailable, RoomClientChanged, RoomClientOwner, RoomClientScope,
)
from plugins.platforms.matrix.thread_context import PreviousTurnCheck, fetch_thread_entries
from plugins.platforms.matrix.turn_context import MatrixTurnContext


if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")
_ROOM_STATE_READ_TIMEOUT_SECONDS = 10.0
_ROOM_NAME_STATE_KEYS = {"m.room.name": "name", "m.room.topic": "topic", "m.room.canonical_alias": "alias"}


class MatrixContextMixin:
    @staticmethod
    def _build_source_permalink(
        room_id: str, event_id: str | None, server_name: str | None = None
    ) -> str | None:
        """Canonical matrix.to permalink for a room/event pair.

        ``event_id`` should be the thread root when the message lives in a thread (stable anchor
        that opens the thread), else the triggering event. The ``via`` parameter carries the
        server name, derived from the room ID when the caller has none. None when no event.
        """
        if not event_id:
            return None
        via = server_name or (room_id.rsplit(":", 1)[-1].strip() or None if ":" in room_id else None)
        permalink = (
            f"https://matrix.to/#/{quote(room_id, safe='!$:@')}/"
            f"{quote(event_id, safe='!$:@')}"
        )
        return f"{permalink}?{urlencode({'via': via})}" if via else permalink

    _client: Any
    _user_id: str
    _event_context_cache: MatrixEventContextCache
    _thread_backfill_limit: int
    _room_backfill_limit: int
    _content_mentions_bot: Callable[[str, dict], bool]
    _is_sender_authorized: Callable[..., bool | None]

    async def fetch_inbound_context(self, event: MessageEvent) -> InboundContextSnapshot:
        from plugins.platforms.matrix.rich_content import MatrixRichContentSnapshot

        if any(
            isinstance(dependency, MatrixRichContentSnapshot)
            and dependency.context.adapter is self
            for dependency in event._inbound_context_dependencies
        ):
            return MatrixRichContentSnapshot.capture(self, event)
        return MatrixTurnContext.capture(self, event)

    async def fetch_thread_history(
        self,
        chat_id: str,
        thread_id: str,
        *,
        before_event_id: str | None = None,
        exclude_event_ids: Collection[str] = (),
        is_previous_turn: PreviousTurnCheck | None = None,
    ) -> MatrixHistoryContext | None:
        entries = await fetch_thread_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            thread_id,
            limit=self._thread_backfill_limit,
            before_event_id=before_event_id,
            exclude_event_ids=exclude_event_ids,
            is_previous_turn=is_previous_turn,
        )
        if not entries:
            return None
        return await MatrixHistoryContext.prepare(
            self, chat_id, entries, "Earlier messages in this thread"
        )

    async def fetch_room_history(
        self,
        chat_id: str,
        event_id: str,
        *,
        is_previous_turn: PreviousTurnCheck | None = None,
        exclude_event_ids: Collection[str] = (),
    ) -> MatrixHistoryContext | None:
        entries = await fetch_room_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            event_id,
            limit=self._room_backfill_limit,
            is_previous_turn=is_previous_turn, exclude_event_ids=exclude_event_ids,
        )
        if not entries:
            return None
        return await MatrixHistoryContext.prepare(
            self, chat_id, entries, "Recent room messages"
        )

    async def fetch_mention_history(
        self, event: MessageEvent
    ) -> MatrixHistoryContext | None:
        """Read the messages that the mention gate dropped in this room or thread since the
        previous turn. Returns None when the room or thread does not require a mention,
        because every message there has already started a turn.

        The scan stops at the bot's own last reply or the last mention that the gate
        admitted, whichever is later. That event belongs to the previous turn in this room
        or thread, and an earlier catch-up covered the messages before it. The previous turn
        can belong to another session, for example after `/new` or when each mention starts
        its own automatic thread. The scan still stops there, so in the main timeline the
        first turn after a reset does not receive the conversation that the reset discarded.
        The first turn of a thread session uses the thread history instead. The bot's status
        notices are not replies, so the scan continues past them and leaves them out."""
        source = event.source
        content = event.raw_message
        if event.internal or source.chat_type == "dm" or not isinstance(content, dict):
            return None
        if not event.metadata.get("matrix_requires_mention") or not event.message_id:
            return None
        if not event.metadata.get(
            "matrix_mention_claimed"
        ) and not self._content_mentions_bot(
            str(content.get("body") or ""),
            content,
        ):
            return None

        room_id = source.chat_id

        def is_previous_turn(sender: str, original_content: dict) -> bool:
            if sender == self._user_id:
                return True
            return self._content_mentions_bot(
                str(original_content.get("body") or ""), original_content,
            ) and self._is_sender_authorized(sender, chat_type="group", chat_id=room_id) is not False

        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.thread_root:
            return await self.fetch_thread_history(
                room_id,
                relation.thread_root,
                before_event_id=event.message_id,
                is_previous_turn=is_previous_turn, exclude_event_ids=event.merged_message_ids,
            )
        return await self.fetch_room_history(
            room_id, event.message_id, is_previous_turn=is_previous_turn, exclude_event_ids=event.merged_message_ids
        )

    @staticmethod
    def _state_event_value(event: Any, key: str) -> Optional[str]:
        """Extract a simple value from a Matrix state event object or dict (top-level, then .content)."""
        if event is None:
            return None
        for obj in (event, event.get("content") if isinstance(event, dict) else getattr(event, "content", None)):
            value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            if value:
                return str(value)
        return None

    async def _get_room_members(
        self: MatrixAdapter, room_id: str, *, force_refresh: bool = False,
        owner: RoomClientScope | None = None,
    ) -> Optional[set[str]]:
        """Read the complete joined member list from the store or homeserver."""
        from plugins.platforms.matrix.adapter import RoomID, Membership

        owner = owner or LiveRoomClient(self)
        owner.check()
        client = owner.client
        if client is None:
            return None

        state_store = getattr(client, "state_store", None)
        if state_store is not None and not force_refresh:
            with suppress(Exception):
                full_list = await state_store.has_full_member_list(RoomID(room_id))
                owner.check()
                if full_list:
                    members = await state_store.get_members(
                        RoomID(room_id), memberships=(Membership.JOIN,)
                    )
                    owner.check()
                    if members is not None:
                        return {str(member) for member in members}

        owner.check()
        with suppress(Exception):
            members = await asyncio.wait_for(client.get_joined_members(RoomID(room_id)), timeout=10)
            owner.check()
            if isinstance(members, dict):
                return {str(member) for member in members}
        owner.check()
        return None

    async def _get_room_member_profiles(
        self: MatrixAdapter, room_id: str, *, owner: RoomClientScope | None = None,
    ) -> Optional[Dict[Any, Any]]:
        from plugins.platforms.matrix.adapter import RoomID, Membership

        owner = owner or LiveRoomClient(self)
        owner.check()
        client = owner.client
        state_store = getattr(client, "state_store", None) if client else None
        if state_store:
            with suppress(Exception):
                profiles = await state_store.get_member_profiles(
                    RoomID(room_id), memberships=(Membership.JOIN,)
                )
                owner.check()
                if profiles:
                    return dict(profiles)

        owner.check()
        if client is not None and hasattr(client, "get_joined_members"):
            with suppress(Exception):
                profiles = await asyncio.wait_for(
                    client.get_joined_members(RoomID(room_id)), _ROOM_STATE_READ_TIMEOUT_SECONDS,
                )
                owner.check()
                if profiles:
                    return dict(profiles)
        owner.check()
        return None

    def _compute_room_display_name(self: MatrixAdapter, profiles: Optional[Dict[Any, Any]]) -> Optional[str]:
        if not profiles:
            return None

        own_user_id = (self._user_id or "").strip().lower()
        names = []
        for user_id, member in profiles.items():
            if str(user_id).strip().lower() == own_user_id:
                continue
            display_name = getattr(member, "displayname", None)
            if display_name and display_name.strip():
                names.append(display_name.strip())
            elif str(user_id).startswith("@") and ":" in str(user_id):
                names.append(str(user_id)[1:].split(":", 1)[0])
            else:
                names.append(str(user_id))

        if not names:
            return None

        names.sort()
        if len(names) == 1:
            return names[0]
        if len(names) <= 3:
            return f"{', '.join(names[:-1])} and {names[-1]}"
        remaining = len(names) - 3
        noun = "other" if remaining == 1 else "others"
        return f"{', '.join(names[:3])} and {remaining} {noun}"

    async def _read_room_state_event(
        self: MatrixAdapter, room_id: str, event_type: str, *, owner: RoomClientScope | None = None,
    ) -> Any:
        """The content of a room state event, or None when the room has no such event (``M_NOT_FOUND``).
        Any other failure, including the read deadline, raises."""
        from plugins.platforms.matrix.adapter import RoomID, MNotFound

        owner = owner or LiveRoomClient(self)
        owner.check()
        client = owner.client
        if not client or not hasattr(client, "get_state_event"):
            return None
        try:
            event = await asyncio.wait_for(
                client.get_state_event(RoomID(room_id), event_type), _ROOM_STATE_READ_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            owner.check()
            if isinstance(exc, MNotFound) or getattr(exc, "errcode", None) == "M_NOT_FOUND":
                return None
            raise
        owner.check()
        return event

    async def _read_room_member_profiles(
        self: MatrixAdapter, room_id: str, *, force_refresh: bool = False,
        owner: RoomClientScope | None = None,
    ) -> tuple[Optional[set[str]], Optional[Dict[Any, Any]]]:
        members = await self._get_room_members(room_id, force_refresh=force_refresh, owner=owner)
        profiles = await self._get_room_member_profiles(room_id, owner=owner) if members is not None else None
        return members, profiles

    def _remember_room_names(
        self: MatrixAdapter, room_id: str, reads: Dict[str, Any], profiles: Optional[Dict[Any, Any]],
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """The room's name, topic, canonical alias and member-derived name. *reads* maps each event type
        in ``_ROOM_NAME_STATE_KEYS`` to its content or to the exception that its read raised, and
        *profiles* is None when the member read failed. A failed read gives the last value read for the
        room, so a timeout does not rename the room; a successful read replaces that value, so a removed
        name or topic applies."""
        values = self._room_state_values.pop(room_id, {})
        for event_type, key in _ROOM_NAME_STATE_KEYS.items():
            event = reads[event_type]
            if not isinstance(event, Exception):
                values[event_type] = (self._state_event_value(event, key) or "").strip() or None
        if profiles is not None:
            values["m.room.member"] = self._compute_room_display_name(profiles)
        if len(self._room_state_values) >= self._room_identity_cache_max:
            del self._room_state_values[next(iter(self._room_state_values))]
        self._room_state_values[room_id] = values
        return (
            values.get("m.room.name"), values.get("m.room.topic"), values.get("m.room.canonical_alias"),
            values.get("m.room.member"),
        )

    def _invalidate_room_identities(self: MatrixAdapter, room_id: str | None = None) -> None:
        """Drop one cached room identity (or all when *room_id* is None)."""
        if room_id is None:
            self._room_identities.clear()
            self._room_identity_cached_at.clear()
        else:
            self._room_identities.pop(room_id, None)
            self._room_identity_cached_at.pop(room_id, None)

    async def _resolve_room_identity(
        self: MatrixAdapter, room_id: str, *, force_refresh: bool = False, owner: RoomClientScope | None = None,
    ) -> MatrixRoomIdentity:
        """Resolve room identity from joined membership and room metadata."""
        owner = owner or LiveRoomClient(self)
        owner.check()
        cached = self._room_identities.get(room_id)
        ttl = self._room_identity_ttl_seconds
        cache_fresh = ttl <= 0 or time.monotonic() - self._room_identity_cached_at.get(room_id, 0.0) <= ttl
        if cached is not None and cache_fresh and not force_refresh:
            return cached
        (
            name_event, topic_event, alias_event, join_rules_event, history_event, encryption_event,
            tombstone_event, member_read,
        ) = reads = await asyncio.gather(
            *(
                self._read_room_state_event(room_id, event_type, owner=owner) for event_type in (
                    "m.room.name", "m.room.topic", "m.room.canonical_alias", "m.room.join_rules",
                    "m.room.history_visibility", "m.room.encryption", "m.room.tombstone",
                )
            ),
            self._read_room_member_profiles(room_id, force_refresh=force_refresh, owner=owner),
            return_exceptions=True,
        )
        owner.check()
        failed_reads = [result for result in reads if isinstance(result, Exception)]
        members, profiles = (None, None) if isinstance(member_read, Exception) else member_read
        if failed_reads:
            logger.debug("Matrix: room state read failed for %s: %r", room_id, failed_reads[0])

        def state_value(event: Any, key: str) -> Optional[str]:
            if isinstance(event, Exception):
                return None
            return (self._state_event_value(event, key) or "").strip() or None

        room_name, room_topic, canonical_alias, member_name = self._remember_room_names(
            room_id, dict(zip(_ROOM_NAME_STATE_KEYS, (name_event, topic_event, alias_event))), profiles,
        )
        member_count = len(members) if members is not None else None
        members_digest = None
        if members is not None and profiles is not None:
            profile_names = {
                str(user_id): str(getattr(profile, "displayname", None) or "")
                for user_id, profile in profiles.items()
            }
            member_rows = [(user_id, profile_names.get(user_id, "")) for user_id in sorted(members)]
            members_digest = hashlib.sha256(
                json.dumps(member_rows, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        has_explicit_name = bool(room_name)
        is_direct = bool(self._dm_rooms.get(room_id, False))
        is_likely_dm = bool(members is not None and len(members) == 2 and self._user_id in members)
        display_name = room_name or canonical_alias or member_name or room_id
        room_state = (
            None if failed_reads or members_digest is None
            else MatrixRoomState(
                display_name, room_topic, members_digest,
                join_rule=state_value(join_rules_event, "join_rule"),
                history_visibility=state_value(history_event, "history_visibility"),
                encrypted=encryption_event is not None, tombstoned=tombstone_event is not None,
            )
        )
        identity = MatrixRoomIdentity(
            room_id=room_id, room_name=room_name, room_topic=room_topic, canonical_alias=canonical_alias,
            server_name=(room_id.rsplit(":", 1)[-1].strip() or None) if ":" in room_id else None,
            joined_member_count=member_count, room_state=room_state,
            is_direct_account_data=is_direct, display_name=display_name,
            has_explicit_name=has_explicit_name, chat_type="dm" if is_likely_dm else "room",
            conflict=bool(is_direct and not is_likely_dm),
            joined_members=frozenset(members) if members is not None else None)
        owner.check()
        if len(self._room_identities) >= self._room_identity_cache_max:
            oldest = min(self._room_identity_cached_at, key=self._room_identity_cached_at.get, default=None)
            if oldest:
                self._invalidate_room_identities(oldest)
        self._room_identities[room_id] = identity
        self._room_identity_cached_at[room_id] = time.monotonic()
        return identity

    async def _is_dm_room(
        self: MatrixAdapter, room_id: str, *, owner: RoomClientOwner | None = None, require_classification: bool = False,
    ) -> bool:
        identity = await self._resolve_room_identity(room_id, owner=owner)
        if require_classification and identity.joined_member_count is None:
            raise RoomClassificationUnavailable
        return identity.chat_type == "dm"

    def _is_allowed_matrix_room(self: MatrixAdapter, room_id: str, chat_type: str) -> bool:
        return not self._allowed_room_ids or room_id in self._allowed_room_ids or chat_type == "dm"

    async def _is_allowed_matrix_room_event(
        self: MatrixAdapter, room_id: str, *, owner: RoomClientOwner | None = None, require_classification: bool = False,
    ) -> bool:
        """MATRIX_ALLOWED_ROOMS gate; DMs are exempt so personal chats survive a project allowlist."""
        if self._is_allowed_matrix_room(room_id, "group"):
            return True
        try:
            is_dm = await self._is_dm_room(room_id, owner=owner, require_classification=require_classification)
            return self._is_allowed_matrix_room(room_id, "dm" if is_dm else "group")
        except (RoomClientChanged, RoomClassificationUnavailable):
            raise
        except Exception as exc:
            logger.debug("Matrix: could not resolve room identity for allowlist check in %s: %s", room_id, exc)
            return False

    async def discover_matrix(
        self: MatrixAdapter, kind: str, room_id: str, limit: int, *, requester: str,
        search_term: str | None = None,
    ) -> dict:
        return await discover_matrix(self, kind, room_id, limit, requester=requester, search_term=search_term)

    async def read_matrix_unread(
        self: MatrixAdapter, room_id: str, thread_id: str, *, requester: str,
    ) -> dict[str, Any]:
        return await read_matrix_unread(self, room_id, thread_id, requester=requester)

    async def mark_matrix_read(
        self: MatrixAdapter, room_id: str, event_id: object, thread_id: object, visibility: object, *, requester: str,
        interrupt_check: Callable[[], bool] = lambda: False,
        before_write: Callable[[], None] = lambda: None,
        progress: ReadProgress | None = None,
    ) -> dict[str, Any]:
        return await mark_matrix_read(
            self, room_id, event_id, thread_id, visibility, requester=requester,
            interrupt_check=interrupt_check, before_write=before_write, progress=progress,
        )
