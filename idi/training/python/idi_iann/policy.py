"""Lookup-table policy container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable, Sequence
import json

StateKey = Tuple[int, int, int, int, int]
Action = str


@dataclass
class PolicyEntry:
    q_values: Dict[Action, float]

    def best_action(self) -> Action:
        return max(self.q_values.items(), key=lambda item: item[1])[0]


class LookupPolicy:
    """In-memory policy with export helpers."""

    ACTIONS: Sequence[Action] = ("hold", "buy", "sell")

    def __init__(self) -> None:
        self._table: Dict[StateKey, PolicyEntry] = {}

    def q_value(self, state: StateKey, action: Action) -> float:
        return self._table.setdefault(state, PolicyEntry({a: 0.0 for a in self.ACTIONS})).q_values[action]

    def update(self, state: StateKey, action: Action, delta: float) -> None:
        entry = self._table.setdefault(state, PolicyEntry({a: 0.0 for a in self.ACTIONS}))
        entry.q_values[action] += delta

    def best_action(self, state: StateKey) -> Action:
        return self._table.setdefault(
            state, PolicyEntry({a: 0.0 for a in self.ACTIONS})
        ).best_action()

    def serialize_manifest(self, target: Path) -> None:
        payload = {
            "states": len(self._table),
            "actions": list(self.ACTIONS),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_trace(
        self,
        episodes: Iterable[Sequence[Dict[str, int]]],
        target_dir: Path,
    ) -> None:
        """Write sbf/bv traces for Tau specs."""

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

