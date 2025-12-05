"""High-level Q-learning loop for IDI tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import random

from .config import TrainingConfig
from .env import SyntheticMarketEnv
from .policy import LookupPolicy, StateKey
from .emote import EmotionEngine
from .strategies import ActionStrategy, EpsilonGreedyStrategy


Action = str


@dataclass
class TraceBatch:
    """Serializable trace data for Tau inputs."""

    ticks: List[Dict[str, int]] = field(default_factory=list)

    def append(self, payload: Dict[str, int]) -> None:
        self.ticks.append(payload)

    def export(self, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        streams = {
            "q_buy": "q_buy.in",
            "q_sell": "q_sell.in",
            "risk_budget_ok": "risk_budget_ok.in",
            "q_emote_positive": "q_emote_positive.in",
            "q_emote_alert": "q_emote_alert.in",
            "q_emote_persistence": "q_emote_persistence.in",
            "q_regime": "q_regime.in",
        }
        for key, filename in streams.items():
            data = [str(tick.get(key, 0)) for tick in self.ticks]
            (target_dir / filename).write_text("\n".join(data), encoding="utf-8")


class QTrainer:
    """Runs tabular Q-learning and emits Tau-ready traces."""

    def __init__(self, config: TrainingConfig, strategy: ActionStrategy | None = None):
        self._config = config
        self._env = SyntheticMarketEnv(config.quantizer, config.rewards)
        self._policy = LookupPolicy()
        self._emotion = EmotionEngine(config.emote)
        self._exploration = 1.0
        self._rng = random.Random(0)
        self._strategy = strategy or EpsilonGreedyStrategy()

    def run(self) -> Tuple[LookupPolicy, TraceBatch]:
        for episode in range(self._config.episodes):
            obs = self._env.reset()
            self._emotion.reset()
            for _ in range(self._config.episode_length):
                state = obs.as_state()
                action = self._choose_action(state)
                new_obs, reward = self._env.step(action)
                self._update_policy(state, action, reward, new_obs.as_state())
                obs = new_obs
            self._exploration *= self._config.exploration_decay

        trace = self._rollout()
        return self._policy, trace

    def _choose_action(self, state: StateKey) -> Action:
        return self._strategy.select(
            state=state,
            policy=self._policy,
            actions=self._env.ACTIONS,
            exploration_rate=self._exploration,
            prng=self._rng,
        )

    def _update_policy(self, state: StateKey, action: Action, reward: float, next_state: StateKey) -> None:
        best_next = self._policy.best_action(next_state)
        td_target = reward + self._config.discount * self._policy.q_value(next_state, best_next)
        td_error = td_target - self._policy.q_value(state, action)
        self._policy.update(state, action, self._config.learning_rate * td_error)

    def _rollout(self) -> TraceBatch:
        """Greedy pass to produce Tau input traces."""
        trace = TraceBatch()
        obs = self._env.reset()
        self._emotion.reset()
        for _ in range(self._config.episode_length):
            state = obs.as_state()
            action = self._policy.best_action(state)
            payload = self._build_payload(action, state)
            trace.append(payload)
            obs, _ = self._env.step(action)
        return trace

    def _build_payload(self, action: Action, state: StateKey) -> Dict[str, int]:
        emote_bits = self._emotion.render(state[-1])
        return {
            "q_buy": 1 if action == "buy" else 0,
            "q_sell": 1 if action == "sell" else 0,
            "risk_budget_ok": 1,
            "q_emote_positive": emote_bits["positive"],
            "q_emote_alert": emote_bits["alert"],
            "q_emote_persistence": emote_bits["persistence"],
            "q_regime": state[3],  # reuse scarcity bucket as regime
        }

