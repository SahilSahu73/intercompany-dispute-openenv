"""Audit service: record actions and detect exploit patterns."""

from datetime import datetime, timezone

from domain.ledger_models import AuditEvent
from domain.scenario_models import EpisodeContext


def record_event(
    ctx: EpisodeContext,
    actor: str,
    action_type: str,
    status: str,
    detail: str = "",
    reference_id: str | None = None,
) -> None:
    """Append an audit event to the episode log."""
    event = AuditEvent(
        timestamp=datetime.now(timezone.utc),
        actor=actor,
        action_type=action_type,
        status=status,
        detail=detail,
        reference_id=reference_id,
    )
    ctx.audit_log.append(event)


def detect_loops(ctx: EpisodeContext, window: int = 5) -> bool:
    """Check if the recent actions are repetitive (loop detection).

    Returns True if:
    1. Last `window` actions are identical (same type + same detail), OR
    2. Alternating between exactly 2 actions, OR
    3. No write actions in the last 10 steps (stuck in reads)
    """
    if len(ctx.audit_log) < window:
        return False

    recent = ctx.audit_log[-window:]

    # Pattern 1: All recent actions are identical
    first = recent[0]
    if all(
        e.action_type == first.action_type and e.detail == first.detail
        for e in recent
    ):
        return True

    # Pattern 2: Alternating A-B-A-B pattern
    if window >= 4:
        types = [e.action_type for e in recent]
        if len(set(types)) == 2:
            if all(types[i] == types[i % 2] for i in range(len(types))):
                return True

    # Pattern 3: No write actions for 10 consecutive steps
    read_only_window = 10
    if len(ctx.audit_log) >= read_only_window:
        last_n = ctx.audit_log[-read_only_window:]
        write_actions = {"execute_match", "post_adjustment", "execute_elimination"}
        if not any(e.action_type in write_actions for e in last_n):
            return True

    return False


def count_action_type(ctx: EpisodeContext, action_type: str) -> int:
    """Count how many times a specific action type has been called."""
    return sum(1 for e in ctx.audit_log if e.action_type == action_type)
