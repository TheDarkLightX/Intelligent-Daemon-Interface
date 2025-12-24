"""
IAN Coordinator Hooks - Observer pattern for contribution events.

Provides a lightweight hook interface for the IANCoordinator to notify
external systems (e.g., WebSocket, logging, metrics) of contribution events
without breaking determinism.

Design Principles:
- Hooks are optional (None by default)
- Hook exceptions are caught and logged, never propagate
- Coordinator remains pure and deterministic
- Hooks receive immutable event data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ContributionMeta, Metrics
    from .coordinator import RejectionReason

logger = logging.getLogger(__name__)


# =============================================================================
# Event Dataclasses
# =============================================================================

@dataclass(frozen=True)
class ContributionAcceptedEvent:
    """Event emitted when a contribution is accepted."""
    
    goal_id: str
    pack_hash: bytes
    contributor_id: str
    score: float
    log_index: int
    metrics: "Metrics"
    leaderboard_position: Optional[int] = None
    is_new_leader: bool = False


@dataclass(frozen=True)
class ContributionRejectedEvent:
    """Event emitted when a contribution is rejected."""
    
    goal_id: str
    pack_hash: bytes
    contributor_id: str
    rejection_reason: "RejectionReason"
    reason_detail: str


@dataclass(frozen=True)
class LeaderboardUpdatedEvent:
    """Event emitted when leaderboard changes."""
    
    goal_id: str
    entries: List["ContributionMeta"]
    active_policy_hash: Optional[bytes] = None


# =============================================================================
# Hooks Protocol
# =============================================================================

class CoordinatorHooks(Protocol):
    """
    Protocol for coordinator event hooks.
    
    Implementations must be exception-safe. The coordinator will catch
    and log any exceptions to preserve determinism.
    
    Hooks are called synchronously after state updates complete.
    For async operations (e.g., WebSocket), implementations should
    use thread-safe queuing or `asyncio.run_coroutine_threadsafe`.
    """
    
    def on_contribution_accepted(self, event: ContributionAcceptedEvent) -> None:
        """Called after a contribution is successfully processed."""
        ...
    
    def on_contribution_rejected(self, event: ContributionRejectedEvent) -> None:
        """Called when a contribution is rejected."""
        ...
    
    def on_leaderboard_updated(self, event: LeaderboardUpdatedEvent) -> None:
        """Called when the leaderboard changes."""
        ...


# =============================================================================
# Null Implementation (for testing)
# =============================================================================

class NullHooks:
    """No-op hooks implementation for testing."""
    
    def on_contribution_accepted(self, event: ContributionAcceptedEvent) -> None:
        pass
    
    def on_contribution_rejected(self, event: ContributionRejectedEvent) -> None:
        pass
    
    def on_leaderboard_updated(self, event: LeaderboardUpdatedEvent) -> None:
        pass


# =============================================================================
# Logging Hooks (for debugging)
# =============================================================================

class LoggingHooks:
    """Hooks that log all events for debugging."""
    
    def __init__(self, log_level: int = logging.INFO):
        self._level = log_level
    
    def on_contribution_accepted(self, event: ContributionAcceptedEvent) -> None:
        logger.log(
            self._level,
            f"[HOOK] Contribution accepted: goal={event.goal_id}, "
            f"hash={event.pack_hash.hex()[:16]}..., score={event.score:.4f}, "
            f"position={event.leaderboard_position}, is_leader={event.is_new_leader}"
        )
    
    def on_contribution_rejected(self, event: ContributionRejectedEvent) -> None:
        logger.log(
            self._level,
            f"[HOOK] Contribution rejected: goal={event.goal_id}, "
            f"hash={event.pack_hash.hex()[:16]}..., reason={event.rejection_reason.name}"
        )
    
    def on_leaderboard_updated(self, event: LeaderboardUpdatedEvent) -> None:
        logger.log(
            self._level,
            f"[HOOK] Leaderboard updated: goal={event.goal_id}, "
            f"entries={len(event.entries)}"
        )
