"""Prepared gateway turns, execution and final transcript publication."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from gateway.response_filters import reply_expected_metadata
from gateway.run_inbound_turn_context import channel_state_metadata
from gateway.warning_notifications import diagnostic_metadata

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


class GatewayTurnExecutionMixin:
    async def _handle_message_with_agent(self: GatewayRunner, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", (event.text or "")[:80].replace("\n", " "),
            getattr(event, "reply_to_message_id", None),
            (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " "),
        )

        resolved = await self._hmwa_resolve_session(event, source)
        if resolved is None:
            return
        source, session_entry, session_key = resolved
        prepared, _session_env_tokens = await self._hmwa_prepare_turn(
            event, source, session_entry, session_key, _quick_key, run_generation,
        )
        if not isinstance(prepared, self._PreparedTurn):
            return prepared
        history, message_text = prepared.history, prepared.message_text

        try:
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(source.thread_id) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Capture the launch session id so post-run compression publication is identity-guarded
            # (a /new may move session_entry.session_id while the old run is still unwinding).
            from gateway.run_heartbeat_acceptance import heartbeat_owner_is_current
            if not heartbeat_owner_is_current(self, event, session_key):
                return
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            # Admission/typing is not execution. All routing, authorization and
            # turn preparation gates have passed when the agent runner is entered.
            event._heartbeat_execution_started = True
            # Internal events reuse the last human turn's channel inputs (see _pinned_channel_inputs).
            _turn_channel_prompt, _turn_source = self._pinned_channel_inputs(
                session_key, event.channel_prompt, source, internal=event.internal,
            )
            if not event.internal:
                # Persist the coherent context+channel pair before execution: a crash during the
                # human turn may be followed by an internal startup-resume on the next process.
                await self._persist_prompt_pins(session_key, _run_start_session_id)
            agent_result = await self._run_agent(
                message=message_text, context_prompt=prepared.context_prompt, history=history, source=_turn_source,
                session_id=_run_start_session_id, session_key=session_key,
                run_generation=run_generation, event_message_id=self._reply_anchor_for_event(event),
                inbound_message_id=str(event.message_id) if event.message_id else None,
                channel_prompt=_turn_channel_prompt, moa_config=getattr(event, "_moa_config", None),
                title_user_message=prepared.title_user_message,
                persist_user_message=prepared.persist_user_message,
                persist_user_timestamp=prepared.persist_user_timestamp,
                persist_user_display_kind=prepared.persist_user_display_kind,
                reply_expected=event.reply_expected,
                persist_user_display_metadata={
                    "gateway_input_owner": prepared.persistence_owner, **channel_state_metadata(event),
                    **reply_expected_metadata(event.reply_expected), **diagnostic_metadata(event)},
                message_type=event.message_type,
                scheduled_heartbeat=bool(getattr(event, "_heartbeat_session_id", None)),
                input_snapshot=getattr(event, "_prepared_inbound", None),
            )
            if getattr(event, "_prepared_inbound", None) is not None:
                prepared.message_text = event._prepared_inbound.message_text
                prepared.persist_user_message = event._prepared_inbound.persist_user_message
                prepared.persist_user_timestamp = event._prepared_inbound.persist_user_timestamp
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            # A queued (/queue) chain answered the LAST message of the chain, so the outer final
            # send (bracketed by the adapter against this event) must be ledgered under that
            # message's id or it collides with an earlier turn's row carrying the same text. Reply
            # routing is untouched: the anchor still comes from this event.
            if isinstance(agent_result, dict):
                _terminal_inbound = agent_result.get("queued_terminal_inbound_id")
                if _terminal_inbound:
                    event.ledger_message_id = str(_terminal_inbound)
                if "queued_terminal_notification_category" in agent_result:
                    event.metadata["notification_category"] = agent_result["queued_terminal_notification_category"]
                if isinstance(agent_result.get("_notification_reply_muted"), bool):
                    event._notification_reply_muted = agent_result["_notification_reply_muted"]

            await self._hmwa_stop_typing_for_turn(event, source)

            if not self._is_session_run_current(_quick_key, run_generation):
                self._hmwa_discard_stale_result(source, _quick_key, run_generation)
                return None

            response, _intentional_silence, agent_messages = await self._hmwa_shape_agent_response(
                agent_result, source, history, session_entry, session_key,
                _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
                persist_user_display_kind=prepared.persist_user_display_kind,
                reply_expected=event.reply_expected,
            )
            response = self._hmwa_prepend_reasoning(agent_result, response, source, _intentional_silence)
            _footer_line = self._hmwa_runtime_footer_line(agent_result, source, _turn_seconds)
            # Streaming already delivered the body: the footer goes out as a trailing send instead.
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"
            await self._hmwa_post_turn_hooks(hook_ctx, agent_result, response)

            agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure = (
                self._hmwa_classify_turn_failure(agent_result, history, session_entry)
            )
            if agent_failed_early and not is_context_overflow_failure:
                response = self._hmwa_add_failed_turn_notice(response, self._hmwa_failed_turn_notice(agent_result))
            response, session_entry = await self._hmwa_compression_exhaustion_reset(
                agent_result, response, session_entry, session_key, source, internal=event.internal,
            )
            await self._hmwa_persist_turn_transcript(
                event=event, source=source, session_entry=session_entry, session_key=session_key,
                agent_result=agent_result, agent_messages=agent_messages, prepared=prepared,
                response=response, agent_failed_early=agent_failed_early,
                hidden_reasoning_incomplete=hidden_reasoning_incomplete,
                is_context_overflow_failure=is_context_overflow_failure,
            )
            return await self._hmwa_deliver_turn_response(
                event, source, session_entry, session_key, run_generation,
                agent_result, agent_messages, response, _footer_line, _intentional_silence,
            )

        except Exception as e:
            return await self._hmwa_agent_error_reply(e, event, source, session_entry, session_key, prepared)
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)
