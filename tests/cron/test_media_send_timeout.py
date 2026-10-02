"""Cron media-send timeout resolution and failure-reason formatting.

Covers two salvaged fixes:

- PR #87965 (@AiwendilInTheWoods): an argument-less exception (notably
  TimeoutError from ``future.result(timeout=...)``) has an empty ``str()``,
  which used to render "failed to send media <path>: " with no reason at
  all — in both the log line and the delivery error recorded on the run.
- PR #87967 (@AiwendilInTheWoods): the per-attachment send timeout was a
  hardcoded 30s; large attachments (long TTS audio, big exports) failed on
  slow uplinks with no way to raise it. Now resolved via
  HERMES_CRON_MEDIA_SEND_TIMEOUT → cron.media_send_timeout_seconds → 300s.
"""

import asyncio
import contextlib
import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest

from cron.scheduler_script import _DEFAULT_MEDIA_SEND_TIMEOUT, _get_media_send_timeout
from cron.scheduler_delivery import _send_media_via_adapter


class TestMediaSendTimeoutResolution:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: {})
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT

    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "45")
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": 900}},
        )
        assert _get_media_send_timeout() == 45

    def test_config_value(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": 900}},
        )
        assert _get_media_send_timeout() == 900

    @pytest.mark.parametrize("bad", ["abc", "-5", "0", ""])
    def test_invalid_env_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", bad)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: {})
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT

    def test_invalid_config_falls_back(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": "nope"}},
        )
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT


class TestEmptyReasonFallback:
    def _run(self, tmp_path, monkeypatch, exc):
        """Drive _send_media_via_adapter into its generic except handler."""
        media = tmp_path / "clip.mp3"
        media.write_bytes(b"x")

        monkeypatch.setattr(
            "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
            staticmethod(lambda files, session_key="": [(str(media), False)]),
        )

        def boom(coro, loop):
            coro.close()
            raise exc

        monkeypatch.setattr("agent.async_utils.WithdrawableDispatch.schedule", boom)

        class _Adapter:
            async def send_voice(self, **kw):  # pragma: no cover - never awaited
                pass

        errors = _send_media_via_adapter(
            _Adapter(), "C123", [(str(media), False)], None, loop=object(),
            job={"id": "job-x"},
        )
        assert len(errors) == 1
        return errors[0]

    def test_timeout_error_names_the_class(self, tmp_path, monkeypatch):
        # TimeoutError() has an empty str() — the recorded reason must not
        # be blank (the trailing-colon-nothing log from the field report).
        err = self._run(tmp_path, monkeypatch, TimeoutError())
        assert err.rstrip() != f"failed to send media {tmp_path / 'clip.mp3'}:"
        assert "TimeoutError" in err

    def test_exception_with_message_keeps_it(self, tmp_path, monkeypatch):
        err = self._run(tmp_path, monkeypatch, RuntimeError("bridge closed"))
        assert "bridge closed" in err


@pytest.mark.parametrize("started", [False, True], ids=["unstarted-retries", "started-observed"])
def test_media_confirmation_timeout_preserves_execution_ownership(
    tmp_path, monkeypatch, caplog, started
):
    from cron.scheduler_delivery import _TargetDelivery, _live_send_media
    from gateway.config import Platform

    path = tmp_path / "media-cache" / "image.png"
    path.parent.mkdir()
    path.write_bytes(b"image")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (path.parent,))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    entered, wedged, unwedge = threading.Event(), threading.Event(), threading.Event()
    observed = threading.Event()
    release = asyncio.Event()
    calls = []
    inner = []
    futures = []
    callbacks = []
    completions = []

    async def send_image(**kwargs):
        entered.set()
        calls.append((kwargs["chat_id"], kwargs["image_path"], kwargs["metadata"]))
        await release.wait()
        raise RuntimeError("late media transport failure")

    def create_send(**kwargs):
        coro = send_image(**kwargs)
        inner.append(coro)
        return coro

    def wedge():
        wedged.set()
        assert unwedge.wait(10), "test did not release the blocked gateway loop"

    if not started:
        loop.call_soon_threadsafe(wedge)

    real_schedule = asyncio.run_coroutine_threadsafe

    class ConfirmationTimeout:
        def __init__(self, future):
            self.future = future

        def result(self, timeout=None):
            assert (entered if started else wedged).wait(10)
            raise TimeoutError

        def cancel(self):
            return self.future.cancel()

        def add_done_callback(self, callback):
            callbacks.append(callback)
            def complete(future):
                try:
                    callback(future)
                finally:
                    completions.append(callback)
                    if len(completions) == len(callbacks):
                        observed.set()
            self.future.add_done_callback(complete)

    def schedule(coro, target_loop):
        future = real_schedule(coro, target_loop)
        futures.append(future)
        return ConfirmationTimeout(future)

    adapter = SimpleNamespace(platform=Platform.TELEGRAM, send_image_file=create_send)
    target = _TargetDelivery(
        job={"id": "media-confirmation"}, platform=Platform.TELEGRAM,
        platform_name="telegram", chat_id="chat", thread_id=None, transport=None,
        pconfig=None, runtime_adapter=adapter, target_adapters={}, config=None,
        loop=loop, notify_delivery=True, origin={}, origin_target=False,
        origin_user_id=None, is_dm_target=False, mirror_text="",
        mirror_this_target=False, in_channel_surface=False,
        inchannel_continuable=False, opened_thread_id=None,
    )
    metadata = {"message_thread_id": "thread", "scope_id": "tenant"}
    media = [(str(path), False)]
    media_errors, delivery_errors = [], []
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr("asyncio.run_coroutine_threadsafe", schedule)
            retry = _live_send_media(target, metadata, media, media_errors, delivery_errors)
        closed_before_loop_resumes = inner[0].cr_frame is None
        unwedge.set()
        loop.call_soon_threadsafe(release.set)
        with contextlib.suppress(CancelledError, RuntimeError):
            futures[0].result(timeout=10)
        if started and callbacks:
            assert observed.wait(10), "late media completion was not observed"
        real_schedule(asyncio.sleep(0), loop).result(timeout=10)
        error = f"failed to send media {path}: TimeoutError (target telegram:chat)"
        late_warning = any(
            "late media transport failure" in record.message
            and "after confirmation timeout" in record.message
            for record in caplog.records
        )
        actual = {
            "retry": retry,
            "calls": calls,
            "media_errors": media_errors,
            "delivery_errors": delivery_errors,
            "closed_before_loop_resumes": closed_before_loop_resumes,
            "future_cancelled": futures[0].cancelled(),
            "late_warning": late_warning,
        }
        expected = {
            "retry": [] if started else media,
            "calls": [("chat", str(path), metadata)] if started else [],
            "media_errors": [] if started else [error],
            "delivery_errors": [error] if started else [],
            "closed_before_loop_resumes": not started,
            "future_cancelled": False,
            "late_warning": started,
        }
        assert actual == expected
    finally:
        unwedge.set()
        loop.call_soon_threadsafe(release.set)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive(), "gateway loop thread did not stop"
        loop.close()
