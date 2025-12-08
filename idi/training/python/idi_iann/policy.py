"""Lookup-table policy container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable, Sequence
import json

from .domain import Action, StateKey


@dataclass
class PolicyEntry:
    """Q-value entry for a single state."""
    q_values: Dict[str, float]  # Use str for serialization compatibility

    def best_action(self) -> str:
        """Return best action as string (for compatibility)."""
        if not self.q_values:
            return Action.HOLD.value
        return max(self.q_values.items(), key=lambda item: item[1])[0]


class LookupPolicy:
    """In-memory policy with export helpers."""

    ACTIONS: Sequence[Action] = (Action.HOLD, Action.BUY, Action.SELL)
    
    @property
    def action_strings(self) -> Sequence[str]:
        """Return action strings for compatibility."""
        return tuple(a.value for a in self.ACTIONS)

    def __init__(self) -> None:
        self._table: Dict[StateKey, PolicyEntry] = {}

    def q_value(self, state: StateKey, action: Action) -> float:
        """Get Q-value for state-action pair."""
        action_str = action.value if isinstance(action, Action) else action
        default_entry = PolicyEntry({a.value: 0.0 for a in self.ACTIONS})
        return self._table.setdefault(state, default_entry).q_values.get(action_str, 0.0)

    def update(self, state: StateKey, action: Action, delta: float) -> None:
        """Update Q-value for state-action pair."""
        action_str = action.value if isinstance(action, Action) else action
        default_entry = PolicyEntry({a.value: 0.0 for a in self.ACTIONS})
        entry = self._table.setdefault(state, default_entry)
        entry.q_values[action_str] = entry.q_values.get(action_str, 0.0) + delta

    def best_action(self, state: StateKey) -> Action:
        """Get best action for state."""
        default_entry = PolicyEntry({a.value: 0.0 for a in self.ACTIONS})
        best_str = self._table.setdefault(state, default_entry).best_action()
        # Convert string back to Action enum
        for action in Action:
            if action.value == best_str:
                return action
        return Action.HOLD  # Default fallback

    def serialize_manifest(self, target: Path) -> None:
        payload = {
            "states": len(self._table),
            "actions": [a.value for a in self.ACTIONS],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_trace(
        self,
        episodes: Iterable[Sequence[Dict[str, int]]],
        target_dir: Path,
    ) -> None:
        """Export policy execution trace for debugging (Tau spec inputs)."""

        target_dir.mkdir(parents=True, exist_ok=True)
        streams: Dict[str, list[int]] = {
            "q_buy": [],
            "q_sell": [],
            "risk_budget_ok": [],
            "q_emote_positive": [],
            "q_emote_alert": [],
            "q_emote_persistence": [],
        }
        regimes: list[int] = []
        for episode in episodes:
            for tick in episode:
                streams["q_buy"].append(tick.get("q_buy", 0))
                streams["q_sell"].append(tick.get("q_sell", 0))
                streams["risk_budget_ok"].append(tick.get("risk_budget_ok", 1))
                streams["q_emote_positive"].append(tick.get("q_emote_positive", 0))
                streams["q_emote_alert"].append(tick.get("q_emote_alert", 0))
                streams["q_emote_persistence"].append(tick.get("q_emote_persistence", 0))
                regimes.append(tick.get("q_regime", 0))
        for name, values in streams.items():
            (target_dir / f"{name}.in").write_text("\n".join(str(v) for v in values), encoding="utf-8")
        (target_dir / "q_regime.in").write_text("\n".join(str(v) for v in regimes), encoding="utf-8")

    def to_entries(self) -> Dict[str, Dict[str, float]]:
        """Return a serializable view of the policy table."""
        return {str(state): dict(entry.q_values) for state, entry in self._table.items()}
