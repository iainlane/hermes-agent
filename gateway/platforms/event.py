"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gateway.session import SessionSource

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"

    @classmethod
    def from_agent_result(cls, result: Dict[str, Any]) -> "ProcessingOutcome":
        """Classify a returned agent result before reply delivery."""
        if result.get("interrupted"):
            return cls.CANCELLED
        if result.get("failed"):
            return cls.FAILURE
        return cls.SUCCESS


class _ProcessingPhase(Enum):
    PENDING = "pending"
    DEFERRED = "deferred"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class _ProcessingInput:
    event: "MessageEvent"
    text: Optional[str]


@dataclass
class _ProcessingCompletion:
    adapter: "BasePlatformAdapter"
    event: "MessageEvent"


@dataclass
class _VoicePreparation:
    text: Optional[str] = None
    transcripts: List[str] = field(default_factory=list)
    echoed: int = 0


@dataclass
class _ProcessingState:
    phase: _ProcessingPhase = _ProcessingPhase.PENDING
    outcome: Optional[ProcessingOutcome] = None
    receipt_message_id: Optional[str] = None
    consumed_receipt_message_id: Optional[str] = None
    receipt_inputs: List[_ProcessingInput] = field(default_factory=list)
    pending_completion: Optional[_ProcessingCompletion] = None

    def defer(self) -> None:
        self.phase = _ProcessingPhase.DEFERRED

    def take_pending_input(self, pending_text: str) -> Optional["MessageEvent"]:
        pending_indices: set[int] = set()
        remaining = pending_text
        # A later correction can quote an earlier input's complete payload.
        for index in range(len(self.receipt_inputs) - 1, -1, -1):
            text = self.receipt_inputs[index].text
            if text is not None and text in remaining:
                pending_indices.add(index)
                remaining = remaining.replace(text, "", 1)
        pending_input = None
        for index, incoming in enumerate(self.receipt_inputs):
            if index in pending_indices:
                pending_input = incoming.event
                continue
            self.consumed_receipt_message_id = incoming.event.receipt_message_id
        self.receipt_inputs.clear()
        self.receipt_message_id = self.consumed_receipt_message_id
        return pending_input

    def start(self) -> None:
        self.phase = _ProcessingPhase.RUNNING
        self.outcome = None
        self.consumed_receipt_message_id = self.receipt_message_id
        self.receipt_inputs.clear()

    def complete(self) -> bool:
        if self.phase in {_ProcessingPhase.DEFERRED, _ProcessingPhase.COMPLETED}:
            return False
        self.phase = _ProcessingPhase.COMPLETED
        return True

    def complete_inline(self) -> bool:
        if self.phase is not _ProcessingPhase.PENDING:
            return False
        return self.complete()


@dataclass
class MessageEvent:
    """Incoming message from a platform — the normalized shape all adapters produce."""
    text: str
    message_type: MessageType = MessageType.TEXT
    # Author, mirrored from ``source`` for per-message prompt builders; None for non-IM sources.
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    # None only in isolated unit tests; production always sets it. Typing it Optional
    # exposes ~60 unguarded ``.source.<attr>`` reads, so that is a separate change.
    source: SessionSource = None
    raw_message: Any = None
    message_id: Optional[str] = None
    # Delivery-ledger identity for the final send, when it differs from ``message_id``. A queued
    # (/queue) chain answers the LAST message of the chain, so its final send has to be ledgered
    # under that message's id. Keyed on the opening event's id instead, two chained turns carrying
    # the same text collide on one obligation id and the earlier turn's row is overwritten (a
    # refused first reply then reads as delivered). Reply routing is unaffected: the reply anchor
    # still comes from this event.
    ledger_message_id: Optional[str] = None
    # Reply anchor for the final send when the answer is to a DIFFERENT message than the one that
    # opened the turn: a successful busy redirect turns the running turn onto the redirecting
    # message, so its reply must quote that message (#115001). ``_reply_anchor_for_event``
    # honours this over ``message_id``; None = derive from the event as usual.
    reply_anchor_override: Optional[str] = None
    # Platform update id (Telegram ``update_id``): ``/restart`` records it so the new gateway
    # advances past it even if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: List[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay only): {prompt_id, option_id, label?,
    # prompt_message_id?}; routed to the approval/slash-confirm/clarify resolvers BEFORE dispatch.
    prompt_response: Optional[Dict[str, Any]] = None
    # Auto-loaded skill(s) for topic/channel bindings; a single name or ordered list.
    auto_skill: Optional[str | list[str]] = None
    # Per-channel ephemeral system prompt; applied at API call time, never persisted to transcript.
    channel_prompt: Optional[str] = None
    # History-backfilled channel context (missed under require_mention); kept out of ``text`` so
    # run.py's sender-prefix logic sees only the trigger message.
    channel_context: Optional[str] = None
    # Set for synthetic events (e.g. background-process notifications) that must bypass user authorization.
    internal: bool = False
    # Free-form per-event metadata (e.g. ``whatsapp_from_owner=True``); plugins must ``.get()``.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / control prompts? Proactive plugin events set False
    # so untrusted payload text stays conversational. New fields append after it (positional compat).
    allow_gateway_control: bool = True
    # Was this message addressed to this bot? False lets a bare silence marker stand (the adapter
    # knows the message was meant for someone else); None means unknown and keeps the visible
    # fallback, like True.
    reply_expected: Optional[bool] = None
    reply_to_author_authorized: Optional[bool] = None
    # dataclasses.replace shares this mutable state with the adapter's original event.
    _processing_state: _ProcessingState = field(default_factory=_ProcessingState, repr=False, compare=False)
    _voice_preparation: _VoicePreparation = field(default_factory=_VoicePreparation, repr=False, compare=False)

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)
    # Run-owned final presentation snapshot; never deserialized from ingress metadata.
    _notification_reply_muted: Optional[bool] = field(default=None, init=False, repr=False, compare=False)

    def absorb_reply_expected(self, other: "MessageEvent") -> None:
        """One turn now answers *other* too: an addressed message wins, then an unknown one."""
        if self.reply_expected is not True and other.reply_expected is not False:
            self.reply_expected = other.reply_expected

    def append_channel_context(self, context: Optional[str]) -> None:
        if not context:
            return
        self.channel_context = f"{self.channel_context}\n{context}" if self.channel_context else context

    def absorb_reply_context(self, other: "MessageEvent") -> None:
        if self.reply_to_text or not other.reply_to_text:
            return

        self.reply_to_message_id = other.reply_to_message_id
        self.reply_to_text = other.reply_to_text
        self.reply_to_author_id = other.reply_to_author_id
        self.reply_to_author_name = other.reply_to_author_name
        self.reply_to_is_own_message = other.reply_to_is_own_message
        self.reply_to_author_authorized = other.reply_to_author_authorized
    def absorb_turn_input(self, other: "MessageEvent", *, input_text: Optional[str] = None) -> None:
        self.absorb_reply_expected(other)
        receipt_id = other.receipt_message_id
        if receipt_id:
            self._processing_state.receipt_message_id = receipt_id
            if self._processing_state is not other._processing_state:
                self._processing_state.receipt_inputs.append(_ProcessingInput(other, input_text))

    @property
    def receipt_message_id(self) -> Optional[str]:
        return self._processing_state.receipt_message_id or self.message_id

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and (self.text or "").lstrip().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = (self.text or "").lstrip().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = (self.text or "").lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
