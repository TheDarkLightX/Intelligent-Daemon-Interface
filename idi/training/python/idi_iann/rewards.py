"""Reward shaping utilities for trading + communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class CommunicationRewardShaper:
    """Heuristic shaping for communication actions.

    This is intentionally lightweight: it rewards aligned alerts and polite/steady
    communication, and penalizes obvious mismatches. For richer setups, plug in a
    learned reward model or preference data.
    """

    alert_bonus: float = 0.2
    alert_mismatch_penalty: float = -0.25
    positive_bonus: float = 0.02
    persistence_bonus: float = 0.02

    def shape(
        self,
        base_reward: float,
        comm_action: str,
        next_state: Tuple[int, ...],
        features: Dict[str, float] | None = None,
    ) -> float:
        """Return shaped reward."""
        shaped = base_reward
        features = features or {}
        # Example: use trend/vol proxies (indices) from state; caller can pass richer features.
        trend_bucket = next_state[2] if len(next_state) > 2 else 0
        risk_signal = features.get("risk_signal", 0.0)

        if comm_action == "alert":
            if risk_signal > 0 or trend_bucket > 0:
                shaped += self.alert_bonus
            else:
                shaped += self.alert_mismatch_penalty
        elif comm_action == "positive":
            shaped += self.positive_bonus
        elif comm_action == "persist":
            shaped += self.persistence_bonus
        # 'silent' leaves reward unchanged
        return shaped

