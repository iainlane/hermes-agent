"""Matrix gateway admission and deferred text-batch receipts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Dict, Optional
from contextlib import suppress
import inspect
import logging
import time

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.location import format_location_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.adapter_feedback import ReadReceiptMode
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache, _split_reply_fallback
from plugins.platforms.matrix.sync_transport import (
    DurableSyncStore, SyncCheckpoints, SyncDispatch, create_sync_client, is_invalid_sync_cursor,
)
from plugins.platforms.matrix.voice_mention import ParkedVoices
from plugins.platforms.matrix.unread import SYNC_FILTER


if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixIntakeMixin(BasePlatformAdapter):
    _client: Any
    _require_mention: bool
    _read_receipts_mode: ReadReceiptMode
    _event_context_cache: MatrixEventContextCache
    _parked_voices: ParkedVoices
    _text_batch_intakes: dict[int, list[tuple[str, asyncio.Future[bool]]]]
    _buffered_intakes: dict[str, asyncio.Future[bool]]
    _strip_mention: Callable[[str], str]
    _content_mentions_bot: Callable[[str, dict], bool]
    _handle_media_message: Callable[..., Awaitable[bool | None]]
    _background_read_receipt: Callable[[str, str], None]
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]

    async def handle_message(self, event: MessageEvent) -> None:
        if getattr(event, "_hermes_startup_restore_replay", False) is True:
            await super().handle_message(event)
            return
        store = getattr(self._client, "sync_store", None)
        event_id = str(event.message_id or "")
        receipts = self._text_batch_intakes.get(id(event), [])
        event_ids = tuple(receipt_id for receipt_id, _receipt in receipts) or (event_id,)
        if not isinstance(store, DurableSyncStore) or not event_id.startswith("$"):
            await super().handle_message(event)
            return
        if not store.reserve_intake(event_id):
            event._gateway_accepted = True
            return
        consumed = self._message_handler is not None
        try:
            await super().handle_message(event)
            if consumed:
                if receipts:
                    await store.accept_intakes(event_ids)
                else:
                    await store.accept_intake(event_id)
        finally:
            store.release_intake(event_id)


    async def _admit(self, event: MessageEvent) -> bool:
        """Pass *event* to the gateway and return whether the gateway consumed it.

        The gateway consumes an event when it starts or queues a turn, and also when it handles
        the event inline or drops it on purpose: a command or redirect while the session is
        busy, a clarify answer, an unauthorised sender or an unresolved profile route. A sync
        retry would run any of those again, so only an event that found no message handler is
        left unacknowledged.
        """
        consumed = getattr(self, "_message_handler", None) is not None
        await self.handle_message(event)
        return consumed


    async def _handle_text_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict, *, reply_parent: MatrixEventContext | None = None) -> asyncio.Future[bool] | bool | None:
        from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command

        body = source_content.get("body", "") or ""
        location_text = None
        if source_content.get("msgtype") == "m.location":
            if not isinstance(body, str):
                body = ""
            location_body = body
            if relates_to.get("m.in_reply_to"):
                _, location_body = _split_reply_fallback(body)
            if self._require_mention and self._content_mentions_bot(body, source_content):
                location_body = self._strip_mention(location_body)
            location_text = format_location_content({**source_content, "body": location_body})
            if location_text is None:
                logger.debug("Matrix: ignoring invalid location %s in %s", event_id, room_id)
                return
        if not body and location_text is None:
            return
        reply_target = MatrixRelation.from_content(relates_to).reply_target
        reply_parent = reply_parent or (self._event_context_cache.retain(room_id, reply_target) if reply_target else None)
        # Dict lookup first: the mention regexes only run when a voice is parked or being gated
        # (both only happen under require_mention).
        if (location_text is None and self._parked_voices.pending(room_id, sender)
                and not self._strip_mention(body).strip() and self._content_mentions_bot(body, source_content)):
            limit = self._parked_voices.mark()  # never claim a voice sent after this mention
            await self._parked_voices.settle(room_id, sender)  # same-/sync-batch voice still gating
            parked = self._parked_voices.claim(room_id, sender, before=limit)
            if parked:  # answer the voice this bare mention was typed for, not an empty text
                voice_id, voice_ts, voice_content, voice_relates = parked
                await self._handle_media_message(
                    room_id, sender, voice_id, voice_ts, voice_content, voice_relates, "m.audio",
                    mention_claimed=True, receipt_event_id=event_id)
                if self._read_receipts_mode == ReadReceiptMode.IMMEDIATE:
                    self._background_read_receipt(room_id, event_id)
                return
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, _normalize_matrix_bang_command(body), source_content, relates_to,
            reply_parent=reply_parent, event_ts=event_ts)
        if msg_event is None:
            return
        if location_text is not None:
            msg_event.text = location_text
            msg_event.message_type = MessageType.TEXT
        self._event_context_cache.store(room_id, event_id, MatrixEventContext(sender, msg_event.text))
        return await self._admit_text_event(msg_event)

    async def _admit_text_event(self, event: MessageEvent) -> asyncio.Future[bool] | bool:
        if event.message_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(event)
            pending = self._pending_text_batches.get(self._text_batch_key(event))
            if pending is None:
                return True
            event_id = str(event.message_id or "")
            receipt: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._text_batch_intakes.setdefault(id(pending), []).append((event_id, receipt))
            self._buffered_intakes[event_id] = receipt
            receipt.add_done_callback(lambda _: self._buffered_intakes.pop(event_id, None))
            # The flush task reports a dispatch error; the receipt only marks the batch as failed.
            receipt.add_done_callback(lambda done: done.cancelled() or done.exception())
            return receipt
        return await self._admit(event)


    async def _dispatch_text_batch(self, event: MessageEvent) -> None:
        dispatch = getattr(self._client, "hermes_sync", None)
        task = asyncio.current_task()
        if isinstance(dispatch, SyncDispatch) and task is not None:
            dispatch.own_background_task(task)
        receipts = self._text_batch_intakes.get(id(event), [])
        try:
            consumed = await self._admit(event)
        except BaseException as exc:
            for _event_id, receipt in receipts:
                if not receipt.done():
                    if isinstance(exc, asyncio.CancelledError):
                        receipt.cancel()
                    else:
                        receipt.set_exception(exc)
            raise
        else:
            for _event_id, receipt in receipts:
                if not receipt.done():
                    receipt.set_result(consumed)
        finally:
            self._text_batch_intakes.pop(id(event), None)

    async def _connect_initial_sync(self: MatrixAdapter, client: Any) -> None:
        """Full initial sync: seed joined rooms, DM cache, and dispatch queued to-device events."""
        try:
            since = await client.sync_store.get_next_batch()
            self._resuming_sync = bool(since)
            try:
                sync_data = await client.sync(since=since, timeout=10000, full_state=True, filter_id=SYNC_FILTER)
            except Exception as exc:
                if not since or not is_invalid_sync_cursor(exc):
                    raise
                logger.warning("Matrix: saved sync cursor was rejected; refreshing full state")
                # A full sync returns recent history that was handled before the restart.
                self._resuming_sync = False
                sync_data = await client.sync(timeout=10000, full_state=True, filter_id=SYNC_FILTER)
            if isinstance(sync_data, dict):
                self._joined_rooms.clear()
                await self._absorb_sync(client, sync_data, initial=True)
            else:
                raise TypeError(f"Matrix: initial sync returned unexpected type {type(sync_data).__name__}")
        except Exception as exc:
            logger.warning("Matrix: initial sync error: %s", exc)
            raise

    async def _connect_matrix(self: MatrixAdapter, *, is_reconnect: bool = False) -> bool:
        from plugins.platforms.matrix.adapter import _create_matrix_session, UserID, EventType

        self._device_id_unverified = False
        if self._client is not None:
            try:
                await self.disconnect()
            except Exception as exc:
                logger.warning("Matrix: error disconnecting before reconnect: %s", exc)
        from mautrix.api import HTTPAPI
        from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
        if not self._homeserver:
            logger.error("Matrix: homeserver URL not configured")
            return False
        # Resolved here, inside the profile scope, so multiplexed profiles never share it.
        store_dir = self._resolve_store_dir()
        store_dir.mkdir(parents=True, exist_ok=True)
        if self._followup_store_path().exists():
            self._purge_expired_watches()
        client_session = _create_matrix_session(self._proxy_url)
        api = HTTPAPI(base_url=self._homeserver, token=self._access_token or "", client_session=client_session)
        state_store = MemoryStateStore()
        sync_store = MemorySyncStore()
        client = create_sync_client(
            mxid=UserID(self._user_id) if self._user_id else UserID(""), device_id=self._device_id or None,
            api=api, state_store=state_store, sync_store=sync_store)
        self._client = client
        self._unread.reset()
        if not await self._connect_authenticate(client, api):
            return False
        sync_store = DurableSyncStore(
            store_dir, self._homeserver, str(client.mxid), str(client.device_id or ""), str(api.token))
        try:
            await sync_store.load()
        except OSError as exc:
            logger.error("Matrix: could not read sync cursor: %s", exc)
            await self.disconnect()
            return False
        client.sync_store = sync_store
        dispatch = getattr(client, "hermes_sync", None)
        self._sync_checkpoints = SyncCheckpoints(
            sync_store, dispatch.dispatching_intakes if isinstance(dispatch, SyncDispatch) else frozenset)
        self._sync_position = None
        if self._encryption and not await self._connect_setup_e2ee(client, api, state_store):
            return False
        if self._encryption and getattr(client, "crypto", None) and isinstance(getattr(client, "hermes_sync", None), SyncDispatch):
            from mautrix.client.encryption_manager import DecryptionDispatcher
            client.remove_dispatcher(DecryptionDispatcher)
            client.add_event_handler(EventType.ROOM_ENCRYPTED, client.hermes_sync.decrypt_sync_event, wait_sync=True)
        from mautrix.client import InternalEventType as IntEvt
        from mautrix.client.dispatcher import MembershipEventDispatcher
        client.add_dispatcher(MembershipEventDispatcher)  # without this INVITE never fires
        client.add_event_handler(EventType.ROOM_MESSAGE, self._on_room_message, wait_sync=True)
        sticker_type = getattr(EventType, "STICKER", None)
        if sticker_type is not None:
            client.add_event_handler(sticker_type, self._on_room_message, wait_sync=True)
        client.add_event_handler(EventType.REACTION, self._on_reaction, wait_sync=True)
        client.add_event_handler(IntEvt.INVITE, self._on_invite, wait_sync=True)
        redaction_type = getattr(EventType, "ROOM_REDACTION", None)
        if redaction_type is not None:
            client.add_event_handler(redaction_type, self._on_redaction, wait_sync=True)
        for state_type_name in (
            "ROOM_TOPIC", "ROOM_NAME", "ROOM_CANONICAL_ALIAS", "ROOM_MEMBER",
            "ROOM_TOMBSTONE", "ROOM_ENCRYPTION", "ROOM_JOIN_RULES",
            "ROOM_HISTORY_VISIBILITY",
        ):
            state_type = getattr(EventType, state_type_name, None)
            if state_type is not None:
                client.add_event_handler(state_type, self._on_room_state, wait_sync=True)
        client.hermes_sync.intake_handlers = {self._on_room_message, self._on_reaction}
        self._startup_ts = time.time()
        self._reset_clock_skew_detector()  # a reconnect after an NTP fix starts clean
        self._closing = False
        self._wire_plugin_handlers(client)
        try:
            await self._connect_initial_sync(client)
        except Exception:
            await self.disconnect()
            return False
        if self._encryption and getattr(client, "crypto", None):
            try:
                await client.crypto.share_keys()
            except Exception as exc:
                logger.warning("Matrix: initial key share failed: %s", exc)
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._mark_connected()
        logger.info("Matrix: connected after initial dispatch checkpoint")
        return True

    async def _disconnect_matrix(self: MatrixAdapter) -> None:
        self._unread.reset()
        self._closing = True
        purge = getattr(self, "_watch_purge_handle", None)
        if purge is not None:
            purge.cancel()
            self._watch_purge_handle = None
        for session_key in tuple(self._reaction_followup_actions):
            self._discard_followup_action(session_key)
        await self._close_matrix_approvals()
        await self.close_rtc_calls()
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):
                pass
        dispatch = getattr(self._client, "hermes_sync", None)
        if isinstance(dispatch, SyncDispatch):
            await dispatch.cancel()
        batch_tasks = tuple(self._pending_text_batch_tasks.values())
        for task in batch_tasks:
            task.cancel()
        await asyncio.gather(*batch_tasks, return_exceptions=True)
        for event_id, receipt in tuple(self._buffered_intakes.items()):
            # A reconnect resumes from the saved cursor and must hand these events over again.
            self._forget_processed_event(event_id)
            receipt.cancel()
        self._pending_text_batches.clear()
        self._text_batch_intakes.clear()
        if self._sync_checkpoints is not None:
            await self._sync_checkpoints.cancel()
        for tasks in (self._invite_join_tasks.values(), self._reaction_redaction_tasks):
            pending = list(tasks)
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._invite_join_tasks.clear()
        self._reaction_redaction_tasks.clear()
        if getattr(self, "_crypto_db", None):
            try:
                await self._crypto_db.stop()
            except Exception as exc:
                logger.debug("Matrix: could not close crypto DB on disconnect: %s", exc)
        if self._client:
            with suppress(Exception):
                await self._client.api.session.close()
            self._client = None
        logger.info("Matrix: disconnected")

    async def _sync_loop(self: MatrixAdapter) -> None:
        from plugins.platforms.matrix.adapter import _is_permanent_matrix_auth_error

        client = self._client
        next_batch = self._sync_position or await client.sync_store.get_next_batch()
        while not self._closing:
            try:
                if await self._rewind_failed_intake(client):
                    next_batch = self._sync_position
                    await asyncio.sleep(5)
                    continue
                # 45s outer cap guards TCP-level hangs the 30s long-poll timeout cannot catch.
                # mautrix raises on every non-2xx, so a non-dict here is never an error object.
                sync_data = await asyncio.wait_for(client.sync(since=next_batch, timeout=30000, filter_id=SYNC_FILTER), timeout=45.0)
                if isinstance(sync_data, dict):
                    next_batch = await self._absorb_sync(client, sync_data) or next_batch
                    await asyncio.sleep(0)  # let fresh invite joins start before the next sync
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures. Transient 5xx outages must retry.
                if _is_permanent_matrix_auth_error(exc):
                    logger.error("Matrix: permanent auth error, stopping sync: %s", exc)
                    return
                if next_batch and is_invalid_sync_cursor(exc):
                    try:
                        await self._connect_initial_sync(client)
                        next_batch = self._sync_position
                        continue
                    except Exception:
                        if self._closing:
                            return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _rewind_failed_intake(self: MatrixAdapter, client: Any) -> bool:
        """Return to the persisted cursor after a buffered text batch failed to reach the gateway."""
        checkpoints = self._sync_checkpoints
        failed = checkpoints.take_failure() if checkpoints is not None else None
        if failed is None:
            return False
        for event_id in failed:
            self._forget_processed_event(event_id)
        self._sync_position = await client.sync_store.get_next_batch()
        self._resuming_sync = bool(self._sync_position)
        logger.warning("Matrix: retrying buffered intake from the saved sync cursor in 5s")
        return True

    async def _absorb_sync(self: MatrixAdapter, client: Any, sync_data: Dict[str, Any], *, initial: bool = False) -> Optional[str]:
        """Apply one sync response: joined rooms, next_batch, event dispatch, pending invites. Returns next_batch.
        The initial (full-state) sync also seeds the DM cache and dispatches so the OlmMachine sees
        to-device key shares queued while offline."""
        self._permalink_routing.observe_sync(client, sync_data)
        self._last_sync_ts = time.time()
        self.update_rtc_call_state(sync_data)
        rooms_join = sync_data.get("rooms", {}).get("join", {})
        if rooms_join or initial:
            self._joined_rooms.update(rooms_join.keys())
            self._invalidate_room_identities()
        nb = sync_data.get("next_batch")  # incremental syncs resume from here
        if initial:
            await self._refresh_dm_cache()
        if client is self._client:
            self._unread.observe(client, sync_data, initial=initial)
            self._joined_rooms.difference_update(sync_data.get("rooms", {}).get("leave", {}))
        await self._dispatch_sync(sync_data)
        self._schedule_pending_invite_joins(sync_data)
        if nb:
            dispatch = getattr(client, "hermes_sync", None)
            store = client.sync_store
            if isinstance(dispatch, SyncDispatch) and isinstance(store, DurableSyncStore):
                seen, buffered = dispatch.take_intakes()
                checkpoints = self._sync_checkpoints
                if checkpoints is None or checkpoints.store is not store:
                    checkpoints = self._sync_checkpoints = SyncCheckpoints(store, dispatch.dispatching_intakes)
                await checkpoints.commit(nb, seen, buffered)
            else:
                await store.put_next_batch(nb)
            if isinstance(dispatch, SyncDispatch):
                dispatch.acknowledge()
            self._sync_position = nb
            self._resuming_sync = True
        if initial:
            logger.info("Matrix: initial dispatch checkpoint complete, joined %d rooms", len(self._joined_rooms))
        return nb

    async def _dispatch_sync(self: MatrixAdapter, sync_data: Dict[str, Any]) -> None:
        """Dispatch a sync response through the mautrix event machinery."""
        client = self._client
        if not client or not hasattr(client, "handle_sync"):
            return
        dispatch = getattr(client, "hermes_sync", None)
        if isinstance(dispatch, SyncDispatch):
            try:
                await dispatch.dispatch_sync(sync_data)
            finally:
                for handler, event_id in dispatch.failed_sync_handlers:
                    if handler in {self._on_room_message, self._on_reaction}:
                        self._forget_processed_event(event_id)
            return
        tasks = client.handle_sync(sync_data)
        if inspect.isawaitable(tasks):
            tasks = await tasks
        if tasks:
            # return_exceptions=True: one failing handler must not drop its SIBLING events.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Matrix: event handler failed during sync dispatch: %s", result)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
