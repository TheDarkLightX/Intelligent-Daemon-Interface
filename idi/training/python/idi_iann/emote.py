"""Emotive signal helpers for Tau art outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import EmoteConfig


@dataclass
class EmotionState:
    """Tracks mood persistence for positive/alert channels."""

    positive_active: bool = False
    alert_active: bool = False
    linger_ticks: int = 0


class EmotionEngine:
    """Turns mood buckets into textual/emoji cues."""

    def __init__(self, config: EmoteConfig):
        self._config = config
        self._state = EmotionState()

    def reset(self) -> None:
        self._state = EmotionState()

    def render(self, mood_bucket: int) -> Dict[str, int]:
        palette_value = self._config.palette.get(mood_bucket, "🙂 steady")
        is_positive = "🙂" in palette_value or "🚀" in palette_value
        is_alert = "⚠️" in palette_value or "😐" in palette_value

        positive_bit = 1 if is_positive else 0
        alert_bit = 1 if is_alert else 0
        if alert_bit:
            self._state.alert_active = True
            self._state.linger_ticks = self._config.linger_ticks
        elif is_positive:
            self._state.positive_active = True
            self._state.linger_ticks = self._config.linger_ticks
        else:
            self._state.linger_ticks = max(0, self._state.linger_ticks - 1)
            if self._state.linger_ticks == 0:
                self._state = EmotionState()
        persistence = 1 if self._state.linger_ticks > 0 else 0
        return {
            "positive": positive_bit,
            "alert": alert_bit,
            "persistence": persistence,
        }

