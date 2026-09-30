"""Effective event invalidation for Matrix redactions."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _redacted_event_id(event: Any) -> str:
    """The event ID that an ``m.room.redaction`` event redacts, or an empty string."""
    content = getattr(event, "content", None)
    # Room version 11 moved ``redacts`` into the content.
    return str(getattr(event, "redacts", None) or (content.get("redacts") if content else None) or "")


class MatrixRedactionMixin:
    """Redaction handling for the Matrix adapter."""

    async def _on_redaction(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", "") or "")
        target = _redacted_event_id(event)
        if room_id and target:
            self._event_context_cache.redact(room_id, target)
            for action in self._reaction_followup_actions.values():
                if action.room_id == room_id:
                    action.pending.discard(target)

        if room_id and target:
            self._withdraw_redacted_message(room_id, str(getattr(event, "sender", "") or ""), target)

    async def _withdraw_redacted_message(
        self, room_id: str, sender: str, target: str
    ) -> bool:
        """Drop ``target`` if it is still waiting for its turn and ``sender`` wrote it. A
        redaction by anyone else, such as a moderator, leaves the message queued."""
        batches = tuple(self._pending_text_batches.items())
        withdrawn = self._parked_voices.discard(room_id, sender, target)
        withdrawn = (
            self.withdraw_pending_message(target, chat_id=room_id, sender_id=sender)
            or withdrawn
        )
        removed = []
        for key, previous in batches:
            remaining = self._pending_text_batches.get(key)
            if remaining is previous:
                continue
            receipts = self._text_batch_intakes.pop(id(previous), [])
            retained_ids = (
                {remaining.message_id, *remaining.merged_message_ids}
                if remaining is not None
                else set()
            )
            retained = []
            for event_id, receipt in receipts:
                if event_id in retained_ids:
                    retained.append((event_id, receipt))
                    continue
                removed.append((event_id, receipt))
            if retained and remaining is not None:
                self._text_batch_intakes.setdefault(id(remaining), []).extend(retained)
        if removed:
            store = getattr(self._client, "sync_store", None)
            try:
                if isinstance(store, DurableSyncStore):
                    await store.accept_intakes(
                        tuple(event_id for event_id, _ in removed)
                    )
            except BaseException as exc:
                for _event_id, receipt in removed:
                    if receipt.done():
                        continue
                    if isinstance(exc, asyncio.CancelledError):
                        receipt.cancel()
                        continue
                    receipt.set_exception(exc)
                raise
            for _event_id, receipt in removed:
                if not receipt.done():
                    receipt.set_result(True)
        if withdrawn:
            logger.info(
                "Matrix: dropped queued message %s in %s; its sender redacted it",
                target,
                room_id,
            )

        return bool(withdrawn)
