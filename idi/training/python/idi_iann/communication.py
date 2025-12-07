"\"\"\"Auxiliary Q-table governing communication/emotive outputs.\"\"\""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

Action = str
CommStateKey = Tuple[int, ...]


@dataclass
class CommunicationEntry:
    q_values: Dict[Action, float]

    def best_action(self) -> Action:
        return max(self.q_values.items(), key=lambda item: item[1])[0]


class CommunicationPolicy:
    """Small Q-table dedicated to expressive communication."""

    def __init__(self, actions: Sequence[Action]):
        if not actions:
            raise ValueError("CommunicationPolicy requires at least one action")
        self._actions = tuple(actions)
        self._table: Dict[CommStateKey, CommunicationEntry] = {}

    def q_value(self, state: CommStateKey, action: Action) -> float:
        return self._table.setdefault(state, CommunicationEntry({a: 0.0 for a in self._actions})).q_values[action]

    def update(self, state: CommStateKey, action: Action, delta: float) -> None:
        entry = self._table.setdefault(state, CommunicationEntry({a: 0.0 for a in self._actions}))
        entry.q_values[action] += delta

    def best_action(self, state: CommStateKey) -> Action:
        return self._table.setdefault(state, CommunicationEntry({a: 0.0 for a in self._actions})).best_action()

    @property
    def actions(self) -> Sequence[Action]:
        return self._actions

