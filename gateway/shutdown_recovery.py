"""Recover profile-owned shutdown spools at gateway startup."""

import logging
import json
from pathlib import Path

from gateway.input_owner import recorded_gateway_input_owner
from gateway.session import SessionStore
from gateway.session_transcript import TranscriptReadError
from gateway.shutdown_pending_lock import pending_snapshot_lock
from gateway.shutdown_pending import PENDING_SCHEMA, PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_source

logger = logging.getLogger("gateway.run")


def consume_executed_pending(store: SessionStore) -> int:
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    home = get_hermes_home().resolve()
    consumed = 0
    for path in (home / "pending_messages").glob("*.json"):
        try:
            with pending_snapshot_lock(path):
                original = path.read_bytes()
                payload = json.loads(original)
                if not isinstance(payload, dict) or payload.get("schema") != PENDING_SCHEMA:
                    continue
                snapshot = PendingQueueSnapshot.from_payload(payload)
                if Path(snapshot.runtime_home).resolve() != home:
                    continue
                resolved = store.resolve_session_id_for_key(snapshot.session_key, not_after=snapshot.ts)
                if resolved is None:
                    continue
                session_id, db = resolved
                if Path(db.db_path).resolve().parent != home:
                    raise ValueError(f"pending execution store {db.db_path} belongs outside profile home {home}")
                remaining = []
                for record in snapshot.events:
                    if "input_owner" not in record:
                        remaining.append(record)
                        continue
                    source = decode_pending_source(record)
                    if store._generate_session_key(source) != snapshot.session_key:
                        raise ValueError("pending owner differs from its session namespace")
                    owner = recorded_gateway_input_owner(source, record["uid"], record["input_owner"])
                    if not store.has_input_owner(session_id, owner):
                        remaining.append(record)
                count = len(snapshot.events) - len(remaining)
                if not count or path.read_bytes() != original:
                    continue
                if remaining:
                    atomic_json_write(path, {**payload, "events": remaining}, mode=0o600)
                else:
                    path.unlink()
                consumed += count
        except (OSError, ValueError, TypeError, KeyError, TranscriptReadError):
            logger.warning("Could not verify pending execution in %s; preserving its records", path, exc_info=True)
    return consumed


def _recover_pending_flushes(runner) -> int:
    """Recover transcripts and consume executed pending inputs in each served home.

    Pending files and the canonical state.db are profile-owned. Recovery enters
    each home's scope before it reads either path.
    """
    from gateway.run import _multiplex_profile_homes
    from gateway.shutdown_flush import recover_pending_to_db
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    resolver = runner.session_store.resolve_session_id_for_key
    recovered = consume_executed_pending(runner.session_store)
    recovered += recover_pending_to_db(session_resolver=resolver, project_pending=False)
    if not getattr(runner.config, "multiplex_profiles", False):
        return recovered
    launch_home = Path(get_hermes_home()).resolve()
    for name, home in _multiplex_profile_homes(runner.config):
        if Path(home).resolve() == launch_home or not (Path(home) / "pending_messages").is_dir():
            continue
        token = set_hermes_home_override(str(home))
        try:
            recovered += consume_executed_pending(runner.session_store)
            recovered += recover_pending_to_db(session_resolver=resolver, project_pending=False)
        except Exception:  # one profile's unreadable spool must not strand the others'
            logger.warning("Pending-message recovery failed for profile %s", name, exc_info=True)
        finally:
            reset_hermes_home_override(token)
    return recovered
