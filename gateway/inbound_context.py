"""Refreshable external context for a new gateway input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class InboundContextSnapshot(Protocol):
    async def refresh(self) -> None: ...

    def prepend_history(self, text: str) -> str: ...

    def reply_event(self, event: MessageEvent) -> MessageEvent: ...

    def reply_image_paths(self) -> list[str]: ...

    def media_event(self, event: MessageEvent) -> MessageEvent:
        """Return *event* with only the authored attachments that are still current."""
        ...


@dataclass(frozen=True)
class ImageEnrichment:
    path: str
    text: str

    @classmethod
    async def enrich_each(
        cls, runner: Any, source: SessionSource, session_key: str, paths: list[str]
    ) -> tuple[ImageEnrichment, ...]:
        """Enrich each image separately so that withdrawing one image removes only its own
        description. The session's native image buffer keeps its paths and adds each path
        routed natively here."""
        native_images = runner._consume_pending_native_image_paths(session_key)
        enrichments = []
        for path in paths:
            text = await runner._enrich_inbound_images(source, session_key, "", [path])
            enrichments.append(cls(path, text))
            native_images.extend(runner._consume_pending_native_image_paths(session_key))
        state = runner._peek_session_state(session_key)
        if state is not None:
            state.persistent.native_image_paths = list(dict.fromkeys(native_images))
        return tuple(enrichments)


@dataclass
class PreparedInboundMessage:
    snapshot: InboundContextSnapshot
    event: MessageEvent
    text: str
    channel_context: str | None = None
    quoted_images: tuple[ImageEnrichment, ...] = ()
    authored_images: tuple[ImageEnrichment, ...] = ()
    message_text: str | None = None
    persist_user_message: str | None = None
    persist_user_timestamp: float | None = None

    def retained_image_paths(self, paths: list[str]) -> list[str]:
        current = self.snapshot.reply_image_paths()
        authored = self.event.authored_media().media_urls
        current_authored = self.snapshot.media_event(self.event).media_urls
        quoted = {image.path for image in self.quoted_images}
        return [
            path
            for path in paths
            if (path not in quoted or path in current or path in current_authored)
            and (path not in authored or path in current_authored or path in current)
        ]

    def revalidate_native_input(
        self, runner: Any, message: str, paths: list[str]
    ) -> tuple[str, list[str]]:
        previous = self.message_text
        current = self.render(runner, timestamps=True)
        if previous is not None and previous in message:
            current = message.replace(previous, current, 1)
        return current, self.retained_image_paths(paths)

    def render(self, runner: Any, *, timestamps: bool = False) -> str:
        text = self.text
        authored = self.snapshot.media_event(self.event).media_urls
        quoted = self.snapshot.reply_image_paths()
        descriptions = [
            *(image.text for image in self.authored_images if image.path in authored and image.text),
            *(image.text for image in self.quoted_images if image.path in quoted and image.text),
        ]
        if descriptions:
            text = "\n\n".join([*descriptions, text])
        text = self.snapshot.prepend_history(text)
        reply = self.snapshot.reply_event(self.event)
        text = runner._prepend_inbound_reply_context(reply, self.event.source, text)
        if self.channel_context:
            text = f"{self.channel_context}\n\n[New message]\n{text}"
        if timestamps:
            text, self.persist_user_message, self.persist_user_timestamp = (
                runner._hmwa_apply_message_timestamp(self.event, text)
            )
        self.message_text = text
        return text
