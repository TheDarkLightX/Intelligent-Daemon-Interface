"""High-level Q-learning loop for IDI tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import random
import secrets

from .config import TrainingConfig
from .env import SyntheticMarketEnv
from .crypto_env import CryptoMarket, MarketParams
from .policy import LookupPolicy
from .domain import Action, StateKey
from .emote import EmotionEngine
from .strategies import ActionStrategy, EpsilonGreedyStrategy
from .abstraction import TileCoder
from .communication import CommunicationPolicy
from .rewards import CommunicationRewardShaper


CommAction = str


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
            "price_up": "price_up.in",
            "price_down": "price_down.in",
            "weight_momentum": "weight_momentum.in",
            "weight_contra": "weight_contra.in",
            "weight_trend": "weight_trend.in",
        }
        for key, filename in streams.items():
            data = [str(tick.get(key, 0)) for tick in self.ticks]
            (target_dir / filename).write_text("\n".join(data), encoding="utf-8")


@dataclass(frozen=True)
class StepResult:
    """Typed wrapper for an environment transition."""

    obs: object
    reward: float
    aux: Dict[str, Any]


class QTrainer:
    """Runs tabular Q-learning and emits Tau-ready traces."""

    def __init__(
        self,
        config: TrainingConfig,
        strategy: ActionStrategy | None = None,
        use_crypto_env: bool = False,
        seed: Optional[int] = 0,
        env_factory: Optional[Callable[[TrainingConfig, bool, int], object]] = None,
    ):
        self._config = config
        self._seed = seed if seed is not None else secrets.randbelow(2**32)
        factory = env_factory or self._default_env_factory
        self._env = factory(config, use_crypto_env, self._seed)
        self._policy = LookupPolicy()
        self._communication = CommunicationPolicy(config.communication.actions)
        self._emotion = EmotionEngine(config.emote)
        self._comm_shaper = CommunicationRewardShaper()
        self._exploration = 1.0
        self._rng = random.Random(self._seed)
        self._strategy = strategy or EpsilonGreedyStrategy()
        self._tile_coder = TileCoder(config.tile_coder) if config.tile_coder else None
        self._episode_rewards: List[float] = []
        self._comm_action_counts: Dict[str, int] = {a: 0 for a in self._communication.actions}

    def run(self) -> Tuple[LookupPolicy, TraceBatch]:
        """Train across episodes and return the learned policy and generated traces."""
        for _ in range(self._config.episodes):
            self._run_episode()
        trace = self._rollout()
        return self._policy, trace

    def _run_episode(self) -> None:
        """Run a single training episode."""
        episode_reward = 0.0
        obs = self._env.reset()
        self._emotion.reset()
        base_state = self._as_state(obs)
        state = self._state_key(base_state)
        for _ in range(self._config.episode_length):
            reward, next_base, next_state = self._step_once(state, base_state)
            episode_reward += reward
            base_state = next_base
            state = next_state
        self._exploration *= self._config.exploration_decay
        self._episode_rewards.append(episode_reward)

    def _step_once(
        self, state: StateKey, base_state: Tuple[int, ...]
    ) -> Tuple[float, Tuple[int, ...], StateKey]:
        """Execute one training step and return (reward, next_base_state, next_state)."""
        action = self._choose_action(state)
        comm_action = self._choose_comm_action(state)
        step = self._step_env(action)
        new_obs, reward, aux = step.obs, step.reward, step.aux
        next_base = self._as_state(new_obs)
        next_state = self._state_key(next_base)
        comm_reward = self._comm_shaper.shape(
            base_reward=reward,
            comm_action=comm_action,
            next_state=next_state,
            features=self._comm_features(next_base, aux),
        )
        self._update_q(state, action, reward, next_state)
        self._update_comm_q(state, comm_action, comm_reward, next_state)
        self._comm_action_counts[comm_action] = self._comm_action_counts.get(comm_action, 0) + 1
        return reward, next_base, next_state

    def _choose_action(self, state: StateKey) -> Action:
        return self._strategy.select(
            state=state,
            policy=self._policy,
            actions=self._env.ACTIONS,
            exploration_rate=self._exploration,
            prng=self._rng,
        )

    def _choose_comm_action(self, state: StateKey) -> CommAction:
        epsilon = max(self._strategy.minimum_exploration, self._exploration)
        if self._rng.random() < epsilon:
            return self._rng.choice(list(self._communication.actions))
        return self._communication.best_action(state)

    def _update_q(self, state: StateKey, action: Action, reward: float, next_state: StateKey) -> None:
        """Update Q-table using TD learning."""
        td_target = self._compute_td_target(state, action, reward, next_state)
        td_error = td_target - self._policy.q_value(state, action)
        self._policy.update(state, action, self._config.learning_rate * td_error)

    def _update_comm_q(
        self, state: StateKey, action: CommAction, reward: float, next_state: StateKey
    ) -> None:
        """Update communication Q-table using TD learning."""
        best_next = self._communication.best_action(next_state)
        td_target = reward + self._config.discount * self._communication.q_value(next_state, best_next)
        td_error = td_target - self._communication.q_value(state, action)
        self._communication.update(state, action, self._config.learning_rate * td_error)

    def _compute_td_target(
        self, state: StateKey, action: Action, reward: float, next_state: StateKey
    ) -> float:
        """Compute TD target for Q-update."""
        best_next = self._policy.best_action(next_state)
        return reward + self._config.discount * self._policy.q_value(next_state, best_next)

    def _rollout(self) -> TraceBatch:
        """Greedy pass to produce Tau input traces."""
        trace = TraceBatch()
        obs = self._env.reset()
        self._emotion.reset()
        prev_base = self._as_state(obs)
        state = self._state_key(prev_base)
        for _ in range(self._config.episode_length):
            action = self._policy.best_action(state)
            comm_action = self._communication.best_action(state)
            step = self._step_env(action)
            next_obs, aux = step.obs, step.aux
            next_base = self._as_state(next_obs)
            weights = self._layer_gates(next_base)
            payload = self._build_payload(action, comm_action, prev_base, next_base, weights, aux)
            trace.append(payload)
            prev_base = next_base
            state = self._state_key(next_base)
        return trace

    def _build_payload(
        self,
        action: Action,
        comm_action: CommAction,
        prev_base: Tuple[int, ...],
        next_base: Tuple[int, ...],
        weights: Dict[str, int],
        aux: Dict[str, Any],
    ) -> Dict[str, int]:
        emote_bits = self._emotion.render(next_base[-1], comm_action)
        price_up = int(next_base[0] > prev_base[0])
        price_down = int(next_base[0] < prev_base[0])
        risk_event = int(bool(aux.get("risk_event"))) if aux else 0
        action_str = action.value if isinstance(action, Action) else action
        return {
            "q_buy": 1 if action_str == "buy" or action == Action.BUY else 0,
            "q_sell": 1 if action_str == "sell" or action == Action.SELL else 0,
            "risk_budget_ok": 1,
            "q_emote_positive": emote_bits["positive"],
            "q_emote_alert": emote_bits["alert"],
            "q_emote_persistence": emote_bits["persistence"],
            "q_regime": next_base[3],
            "price_up": price_up,
            "price_down": price_down,
            "weight_momentum": weights["momentum"],
            "weight_contra": weights["contrarian"],
            "weight_trend": weights["trend"],
            "risk_event": risk_event,
        }

    def _layer_gates(self, base_state: Tuple[int, ...]) -> Dict[str, int]:
        if not self._config.layers.emit_weight_streams:
            return {"momentum": 1, "contrarian": 1, "trend": 1}
        trend_bucket = base_state[2]
        max_trend = max(1, self._config.quantizer.trend_buckets - 1)
        trend_ratio = trend_bucket / max_trend

        momentum_gate = int(trend_ratio >= self._config.layers.momentum_threshold)
        contrarian_gate = int(trend_ratio <= self._config.layers.contrarian_threshold)
        price_bucket = base_state[0]
        price_parity = price_bucket % 2 == 0
        trend_gate = int(price_parity == self._config.layers.trend_favors_even)

        if momentum_gate + contrarian_gate + trend_gate == 0:
            momentum_gate = 1

        return {
            "momentum": momentum_gate,
            "contrarian": contrarian_gate,
            "trend": trend_gate,
        }

    def _state_key(self, base_state: Tuple[int, ...]) -> StateKey:
        if not self._tile_coder:
            return base_state
        return base_state + self._tile_coder.encode(base_state)

    def _as_state(self, obs: object) -> Tuple[int, ...]:
        """Normalize observation to tuple form."""
        if hasattr(obs, "as_state"):
            return obs.as_state()  # type: ignore[return-value]
        if hasattr(obs, "price") and hasattr(obs, "regime"):
            regime_idx = {"bull": 0, "bear": 1, "chop": 2, "panic": 3}.get(getattr(obs, "regime", "chop"), 2)
            pos = getattr(obs, "position", 0)
            pnl = getattr(obs, "pnl", 0.0)
            ret = getattr(obs, "last_return", 0.0)
            return (int(pos + 1), regime_idx, int(abs(ret) > 0.01), int(pnl >= 0))
        raise ValueError(f"Unknown observation type: {type(obs)}")

    def _comm_features(self, next_state: Tuple[int, ...], aux: Dict[str, Any]) -> Dict[str, float]:
        risk_signal = float(aux.get("risk_event", 0.0)) if aux else 0.0
        if not risk_signal and len(next_state) > 2:
            risk_signal = float(next_state[2] > 0)
        return {"risk_signal": risk_signal}

    def _step_env(self, action: Action) -> Tuple[object, float, Dict[str, Any]]:
        # Convert Action enum to string for environment compatibility
        action_str = action.value if isinstance(action, Action) else action
        step_out = self._env.step(action_str)
        if isinstance(step_out, tuple) and len(step_out) == 3:
            obs, reward, info = step_out
            aux: Dict[str, Any] = info if isinstance(info, dict) else {}
        else:
            obs, reward = step_out
            aux = {}
        return StepResult(obs=obs, reward=float(reward), aux=aux)

    def _default_env_factory(self, cfg: TrainingConfig, use_crypto: bool, seed: int) -> object:
        return self._default_env_factory_static(cfg, use_crypto, seed, None)

    @staticmethod
    def _default_env_factory_static(
        cfg: TrainingConfig, use_crypto: bool, seed: int, market_params: MarketParams | None
    ) -> object:
        if use_crypto:
            mp = market_params or MarketParams()
            mp.seed = seed
            return CryptoMarket(mp)
        return SyntheticMarketEnv(cfg.quantizer, cfg.rewards, seed=seed)

    def stats(self) -> Dict[str, float]:
        mean_reward = sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0.0
        return {
            "episodes": len(self._episode_rewards),
            "mean_reward": mean_reward,
            "comm_action_counts": dict(self._comm_action_counts),
        }

