"""Episodic memory for k-NN style Q estimates (lookup-friendly)."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from .domain import Action, StateKey


class EpisodicQMemory:
    """Simple episodic memory: (state, action) -> recent TD targets."""

    def __init__(self, capacity: int = 1024, k: int = 8, decay: float = 0.99):
        self.capacity = capacity
        self.k = k
        self.decay = decay
        self._store: Dict[Tuple[StateKey, str], Deque[float]] = defaultdict(deque)
        self._counts: Deque[Tuple[StateKey, str]] = deque()

    def write(self, state: StateKey, action: Action | str, target_q: float) -> None:
        key = (state, action.value if isinstance(action, Action) else action)
        buf = self._store[key]
        buf.append(target_q)
        if len(buf) > self.k:
            buf.popleft()
        self._counts.append(key)
        # Evict oldest if over capacity
        if len(self._counts) > self.capacity:
            oldest = self._counts.popleft()
            if oldest in self._store and not self._store[oldest]:
                self._store.pop(oldest, None)

    def query(self, state: StateKey, action: Action | str) -> float | None:
        key = (state, action.value if isinstance(action, Action) else action)
        if key not in self._store or not self._store[key]:
            return None
        buf = list(self._store[key])
        # Exponential decay weighting
        weights = [self.decay ** i for i in range(len(buf))]
        numerator = sum(w * v for w, v in zip(weights, reversed(buf)))
        denom = sum(weights)
        return numerator / denom if denom > 0 else None

    def freeze(self) -> Dict[str, Dict[str, float]]:
        """Export as a table compatible with policy commitments."""
        table: Dict[str, Dict[str, float]] = {}
        for (state, action), values in self._store.items():
            if not values:
                continue
            table.setdefault(str(state), {})[action] = sum(values) / len(values)
        return table
