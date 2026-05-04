"""Resolve native Matrix delivery destinations."""

from collections.abc import Awaitable, Callable
import logging
from typing import Any

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixDeliveryMixin:
    _client: Any
    _joined_rooms: set[str]
    _join_room_by_id: Callable[[str], Awaitable[bool]]

    async def _resolve_send_target(self, chat_id: str) -> str:
        """Resolve aliases to room IDs and join the resolved room before sending.

        The Client-Server send endpoint accepts room IDs. On lookup failure,
        return the original target so the send reports the homeserver error.
        """
        if not chat_id:
            return chat_id
        target = chat_id.split("/", 1)[0]
        if not target.startswith("#"):
            return chat_id
        try:
            info = await self._client.resolve_room_alias(target)
            room_id = str(info.room_id) if info and info.room_id else ""
        except Exception as exc:
            logger.warning("Matrix: failed to resolve alias %s: %s", target, exc)
            return chat_id
        if not room_id:
            logger.warning(
                "Matrix: alias %s did not resolve to a room ID; the alias "
                "must be published as a Local Address on the target room. "
                "Either add it in Element (Room Settings, General, Local "
                "Addresses) or target by room ID instead.",
                target,
            )
            return chat_id
        if room_id not in self._joined_rooms:
            await self._join_room_by_id(room_id)
        return room_id

