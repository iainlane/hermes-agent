"""Recover profile-owned shutdown spools at gateway startup."""

import logging
from pathlib import Path

logger = logging.getLogger("gateway.run")


def _recover_pending_flushes(runner) -> int:
    """Replay every ``pending_messages`` spool this gateway owns into state.db; return the count.

    ``_get_flush_dir`` follows the active HERMES_HOME, so a routed turn on a multiplexed gateway spools
    its stalled transcript backlog under ``profiles/<name>/`` and the runtime drain forgets it on
    restart. After the launch home, replay each served profile inside its own home so the default
    store ``recover_pending_to_db`` opens is that profile's state.db (#123584).
    """
    from gateway.run import _multiplex_profile_homes
    from gateway.shutdown_flush import recover_pending_to_db
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    resolver = runner.session_store.resolve_session_id_for_key
    recovered = recover_pending_to_db(session_resolver=resolver)
    if not getattr(runner.config, "multiplex_profiles", False):
        return recovered
    launch_home = Path(get_hermes_home()).resolve()
    for name, home in _multiplex_profile_homes(runner.config):
        if Path(home).resolve() == launch_home or not (Path(home) / "pending_messages").is_dir():
            continue
        token = set_hermes_home_override(str(home))
        try:
            recovered += recover_pending_to_db(session_resolver=resolver)
        except Exception:  # one profile's unreadable spool must not strand the others'
            logger.warning("Pending-message recovery failed for profile %s", name, exc_info=True)
        finally:
            reset_hermes_home_override(token)
    return recovered
