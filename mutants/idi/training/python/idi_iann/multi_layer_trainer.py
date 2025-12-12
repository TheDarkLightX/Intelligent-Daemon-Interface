"""Multi-layer Q-learning trainer with fractal abstraction and layer coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import random
import numpy as np

from .config import TrainingConfig, MultiLayerConfig, FractalConfig, FractalLevelConfig
from .env import SyntheticMarketEnv
from .crypto_env import CryptoMarket, MarketParams
from .domain import Action, StateKey
from .emote import EmotionEngine
from .strategies import ActionStrategy, EpsilonGreedyStrategy
from .fractal_abstraction import (
    FractalStateEncoder, HierarchicalTileCoder, FractalQTable, FractalPolicy, FractalConfig as FractalAbstractionConfig,
    FractalLevelConfig as FractalAbstractionLevelConfig
)
from .communication import CommunicationPolicy
from .rewards import CommunicationRewardShaper
from .policy import LookupPolicy


@dataclass
class LayerPerformance:
    """Tracks performance metrics for a layer."""
    name: str
    recent_rewards: List[float] = field(default_factory=list)
    win_count: int = 0
    loss_count: int = 0
    total_actions: int = 0
    
    def update(self, reward: float, action_taken: bool):
        """Update performance metrics."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        self.total_actions += 1
        if action_taken:
            if reward > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
    
    def win_rate(self) -> float:
        """Compute win rate."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    def recent_performance(self) -> float:
        """Compute recent average reward."""
        return np.mean(self.recent_rewards) if self.recent_rewards else 0.0


class LayerCoordinator:
    """
    Coordinates multiple trading layers with weighted voting.
    
    Uses uncertainty estimation and confidence scores to weight layer votes.
    """
    
    def __init__(self, layers: Dict[str, FractalPolicy], coordination_method: str):
        self.layers = layers
        self.coordination_method = coordination_method  # "weighted_voting", "ensemble", "hierarchical"
        self._performance: Dict[str, LayerPerformance] = {
            name: LayerPerformance(name) for name in layers.keys()
        }
    
    def select_action(self, state: Dict[str, float]) -> Tuple[Action, Dict[str, float]]:
        """
        Select action using coordination method.
        
        Returns:
            (action, layer_weights) where layer_weights shows contribution of each layer
        """
        if self.coordination_method == "weighted_voting":
            return self._weighted_voting(state)
        elif self.coordination_method == "ensemble":
            return self._ensemble_selection(state)
        elif self.coordination_method == "hierarchical":
            return self._hierarchical_selection(state)
        else:
            return self._weighted_voting(state)  # Default
    
    def _weighted_voting(self, state: Dict[str, float]) -> Tuple[Action, Dict[str, float]]:
        """Weighted voting based on layer confidence and recent performance."""
        votes: Dict[Action, float] = {}
        confidences: Dict[str, float] = {}
        
        for layer_name, layer in self.layers.items():
            action = layer.choose_action(state, exploration=0.0)
            confidence = self._compute_confidence(layer_name, state)
            confidences[layer_name] = confidence
            votes[action] = votes.get(action, 0.0) + confidence
        
        best_action = max(votes.items(), key=lambda x: x[1])[0] if votes else Action.HOLD
        return best_action, confidences
    
    def _compute_confidence(self, layer_name: str, state: Dict[str, float]) -> float:
        """Compute layer confidence based on uncertainty and recent performance."""
        perf = self._performance[layer_name]
        
        # Recent performance (0.0 to 1.0, normalized)
        recent_perf = perf.recent_performance()
        normalized_perf = (recent_perf + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Win rate
        win_rate = perf.win_rate()
        
        # Confidence = weighted combination
        confidence = 0.6 * normalized_perf + 0.4 * win_rate
        
        return max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
    
    def _ensemble_selection(self, state: Dict[str, float]) -> Tuple[Action, Dict[str, float]]:
        """Ensemble selection: average Q-values across layers."""
        q_sums: Dict[Action, float] = {}
        counts: Dict[Action, int] = {}
        
        for layer_name, layer in self.layers.items():
            for action in Action:
                q_val = layer.fractal_q_table.q_value(state, action)
                q_sums[action] = q_sums.get(action, 0.0) + q_val
                counts[action] = counts.get(action, 0) + 1
        
        # Average Q-values
        avg_q: Dict[Action, float] = {
            action: q_sums[action] / counts[action] 
            for action in q_sums.keys()
        }
        
        best_action = max(avg_q.items(), key=lambda x: x[1])[0] if avg_q else Action.HOLD
        confidences = {name: 1.0 / len(self.layers) for name in self.layers.keys()}
        
        return best_action, confidences
    
    def _hierarchical_selection(self, state: Dict[str, float]) -> Tuple[Action, Dict[str, float]]:
        """Hierarchical selection: use regime-aware layer first, fall back to others."""
        # Check if we can determine regime
        regime = state.get("regime", 0.0)
        
        # Regime-based layer selection
        if abs(regime) > 0.01:  # Strong regime signal
            if regime > 0:
                primary_layer = "momentum" if "momentum" in self.layers else list(self.layers.keys())[0]
            else:
                primary_layer = "mean_reversion" if "mean_reversion" in self.layers else list(self.layers.keys())[0]
        else:
            primary_layer = "regime_aware" if "regime_aware" in self.layers else list(self.layers.keys())[0]
        
        # Use primary layer if available
        if primary_layer in self.layers:
            action = self.layers[primary_layer].choose_action(state, exploration=0.0)
            confidences = {name: 1.0 if name == primary_layer else 0.0 for name in self.layers.keys()}
            return action, confidences
        
        # Fallback to weighted voting
        return self._weighted_voting(state)
    
    def update_performance(self, layer_name: str, reward: float, action_taken: bool):
        """Update performance metrics for a layer."""
        if layer_name in self._performance:
            self._performance[layer_name].update(reward, action_taken)


class MultiLayerTrainer:
    """
    Trainer for multi-layer Q-learning with fractal abstraction.
    
    Trains multiple layers (momentum, mean-reversion, regime-aware) independently
    with specialized rewards, then coordinates them for final action selection.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        use_crypto_env: bool = False,
        seed: Optional[int] = 0,
        env_factory: Optional[Callable[[TrainingConfig, bool, int], object]] = None,
    ):
        if not config.multi_layer:
            raise ValueError("MultiLayerTrainer requires multi_layer config")
        if not config.fractal:
            raise ValueError("MultiLayerTrainer requires fractal config")
        
        self._config = config
        self._multi_layer_config = config.multi_layer
        self._fractal_config = config.fractal
        self._seed = seed if seed is not None else random.randint(0, 2**32)
        
        # Create environment
        factory = env_factory or self._default_env_factory
        self._env = factory(config, use_crypto_env, self._seed)
        
        # Create fractal encoder
        fractal_levels = []
        for level in config.fractal.levels:
            fractal_levels.append(
                FractalAbstractionLevelConfig(
                    features=tuple(level.features),
                    scale_factor=level.scale_factor,
                    visit_threshold=level.visit_threshold
                )
            )
        fractal_abstraction_config = FractalAbstractionConfig(
            levels=tuple(fractal_levels),
            backoff_enabled=config.fractal.backoff_enabled,
            hierarchical_updates=config.fractal.hierarchical_updates
        )
        self._fractal_encoder = FractalStateEncoder(fractal_abstraction_config)
        
        # Create hierarchical tile coder if tile_coder config exists
        self._tile_coder = None
        if config.tile_coder:
            scale_ratios = [level.scale_factor for level in config.fractal.levels]
            self._tile_coder = HierarchicalTileCoder(
                config.tile_coder,
                len(config.fractal.levels),
                scale_ratios
            )
        
        # Create layers
        self._layers: Dict[str, FractalPolicy] = {}
        for layer_name in self._multi_layer_config.layers:
            fractal_q_table = FractalQTable(self._fractal_encoder, self._tile_coder)
            policy = FractalPolicy(fractal_q_table)
            self._layers[layer_name] = policy
        
        # Create coordinator
        self._coordinator = LayerCoordinator(self._layers, self._multi_layer_config.coordination)
        
        # Communication layer
        self._communication = CommunicationPolicy(config.communication.actions)
        self._emotion = EmotionEngine(config.emote)
        self._comm_shaper = CommunicationRewardShaper()
        
        # Training state
        self._exploration = 1.0
        self._rng = random.Random(self._seed)
        self._episode_rewards: List[float] = []
        self._comm_action_counts: Dict[str, int] = {a: 0 for a in self._communication.actions}
    
    @staticmethod
    def _default_env_factory(config: TrainingConfig, use_crypto: bool, seed: int) -> object:
        """Default environment factory."""
        if use_crypto:
            mp = MarketParams()
            mp.seed = seed
            return CryptoMarket(mp)
        return SyntheticMarketEnv(config.quantizer, config.rewards, seed=seed)
    
    def run(self) -> Tuple[Dict[str, FractalPolicy], Any]:
        """Train across episodes and return learned policies and trace."""
        for _ in range(self._config.episodes):
            self._run_episode()
        trace = self._rollout()
        return self._layers, trace
    
    def _run_episode(self) -> None:
        """Run a single training episode."""
        episode_reward = 0.0
        obs = self._env.reset()
        self._emotion.reset()
        
        for _ in range(self._config.episode_length):
            reward, obs = self._step_once(obs)
            episode_reward += reward
        
        self._exploration *= self._config.exploration_decay
        self._episode_rewards.append(episode_reward)
    
    def _step_once(self, obs: Any) -> Tuple[float, Any]:
        """Execute one training step."""
        # Convert observation to state dict
        state_dict = self._obs_to_state_dict(obs)
        
        # Get actions from all layers
        layer_actions = {}
        for layer_name, layer in self._layers.items():
            action = layer.choose_action(state_dict, self._exploration)
            layer_actions[layer_name] = action
        
        # Coordinate actions
        coordinated_action, layer_weights = self._coordinator.select_action(state_dict)
        
        # Step environment
        next_obs, base_reward, aux = self._step_env(coordinated_action)
        
        # Compute layer-specific rewards
        for layer_name, layer_action in layer_actions.items():
            layer_reward = self._compute_layer_reward(
                base_reward, layer_action, coordinated_action,
                layer_weights[layer_name], aux, layer_name
            )
            
            # Update layer Q-table
            next_state_dict = self._obs_to_state_dict(next_obs)
            td_target = self._compute_td_target(state_dict, layer_action, layer_reward, next_state_dict, layer_name)
            current_q = self._layers[layer_name].fractal_q_table.q_value(state_dict, layer_action)
            td_error = td_target - current_q
            self._layers[layer_name].fractal_q_table.update(state_dict, layer_action, self._config.learning_rate * td_error)
            
            # Update performance
            self._coordinator.update_performance(
                layer_name, layer_reward, layer_action != Action.HOLD
            )
        
        # Communication layer update
        comm_action = self._communication.best_action(self._comm_state_key(state_dict))
        comm_reward = self._comm_shaper.shape(
            base_reward=base_reward,
            comm_action=comm_action,
            next_state=self._comm_state_key(self._obs_to_state_dict(next_obs)),
            features=self._comm_features(state_dict, aux),
        )
        comm_td_target = comm_reward + self._config.discount * 0.0  # Simplified
        comm_current_q = self._communication.q_value(self._comm_state_key(state_dict), comm_action)
        comm_td_error = comm_td_target - comm_current_q
        self._communication.update(self._comm_state_key(state_dict), comm_action, self._config.learning_rate * comm_td_error)
        self._comm_action_counts[comm_action] = self._comm_action_counts.get(comm_action, 0) + 1
        
        return base_reward, next_obs
    
    def _obs_to_state_dict(self, obs: Any) -> Dict[str, float]:
        """Convert observation to state dictionary."""
        # Handle different observation types
        state_dict = {}
        
        # SyntheticMarketEnv returns EnvObservation with quantized buckets
        if hasattr(obs, "price_bucket"):
            state_dict["price"] = float(obs.price_bucket) / 4.0  # Normalize
            state_dict["volume"] = float(obs.volume_bucket) / 4.0
            state_dict["trend"] = float(obs.trend_bucket) / 4.0
            state_dict["momentum"] = float(obs.trend_bucket) / 4.0  # Use trend as momentum proxy
            state_dict["regime"] = float(obs.trend_bucket) / 4.0  # Use trend as regime proxy
        # CryptoMarket returns MarketState
        elif hasattr(obs, "price"):
            state_dict["price"] = float(obs.price) / 1000.0  # Normalize
            state_dict["volume"] = 0.5  # Default
            state_dict["trend"] = float(obs.last_return) if hasattr(obs, "last_return") else 0.0
            state_dict["momentum"] = float(obs.last_return) if hasattr(obs, "last_return") else 0.0
            # Map regime string to float
            if hasattr(obs, "regime"):
                regime_map = {"bull": 0.75, "bear": 0.25, "chop": 0.5, "panic": 0.0}
                state_dict["regime"] = regime_map.get(obs.regime, 0.5)
            else:
                state_dict["regime"] = 0.5
        else:
            # Default values
            state_dict = {
                "price": 0.5,
                "volume": 0.5,
                "trend": 0.5,
                "momentum": 0.5,
                "regime": 0.5
            }
        
        return state_dict
    
    def _step_env(self, action: Action) -> Tuple[Any, float, Dict[str, Any]]:
        """Step environment and return (next_obs, reward, aux)."""
        # Convert Action enum to string for environment compatibility
        action_str = action.value if isinstance(action, Action) else action
        
        if hasattr(self._env, "step"):
            result = self._env.step(action_str)
            if isinstance(result, tuple) and len(result) >= 2:
                next_obs = result[0]
                reward = result[1]
                aux = result[2] if len(result) > 2 else {}
                return next_obs, reward, aux
        # Fallback
        return self._env.reset(), 0.0, {}
    
    def _compute_layer_reward(
        self, 
        base_reward: float, 
        layer_action: Action, 
        coordinated_action: Action,
        layer_weight: float,
        aux: Dict[str, Any],
        layer_name: str
    ) -> float:
        """Compute specialized reward for each layer."""
        reward = base_reward * layer_weight
        
        # Bonus for contributing to final action
        if layer_action == coordinated_action:
            reward += 0.01
        
        # Layer-specific shaping
        if layer_name == "momentum":
            # Momentum bonus for trend-following
            if aux.get("trend", 0) > 0 and layer_action == Action.BUY:
                reward += 0.05
            elif aux.get("trend", 0) < 0 and layer_action == Action.SELL:
                reward += 0.05
        elif layer_name == "mean_reversion":
            # Mean-reversion bonus for contrarian trades
            if aux.get("price_deviation", 0) < -0.02 and layer_action == Action.BUY:
                reward += 0.04
            elif aux.get("price_deviation", 0) > 0.02 and layer_action == Action.SELL:
                reward += 0.04
        elif layer_name == "regime_aware":
            # Regime-aware bonus for appropriate actions
            regime = aux.get("regime", "chop")
            if regime == "bull" and layer_action == Action.BUY:
                reward += 0.06
            elif regime == "bear" and layer_action == Action.SELL:
                reward += 0.06
        
        return reward
    
    def _compute_td_target(
        self, 
        state: Dict[str, float], 
        action: Action, 
        reward: float, 
        next_state: Dict[str, float],
        layer_name: str
    ) -> float:
        """Compute TD target for Q-update."""
        layer = self._layers[layer_name]
        best_next_action = layer.choose_action(next_state, exploration=0.0)
        next_q = layer.fractal_q_table.q_value(next_state, best_next_action)
        return reward + self._config.discount * next_q
    
    def _comm_state_key(self, state_dict: Dict[str, float]) -> Tuple[int, ...]:
        """Convert state dict to communication state key."""
        # Quantize for communication Q-table
        return (
            int(state_dict.get("regime", 0.0) * 4) % 4,
            int(state_dict.get("trend", 0.0) * 4) % 4,
        )
    
    def _comm_features(self, state_dict: Dict[str, float], aux: Dict[str, Any]) -> Dict[str, float]:
        """Extract communication features."""
        return {
            "reward": aux.get("reward", 0.0),
            "regime": state_dict.get("regime", 0.0),
            "volatility": aux.get("volatility", 0.0),
        }
    
    def _rollout(self) -> Any:
        """Generate trace batch from current policy."""
        from .trainer import TraceBatch
        
        trace = TraceBatch()
        obs = self._env.reset()
        self._emotion.reset()
        
        for _ in range(self._config.episode_length):
            state_dict = self._obs_to_state_dict(obs)
            action, layer_weights = self._coordinator.select_action(state_dict)
            comm_action = self._communication.best_action(self._comm_state_key(state_dict))
            
            # Get layer votes
            layer_votes = {}
            for layer_name, layer in self._layers.items():
                layer_vote = layer.choose_action(state_dict, exploration=0.0)
                layer_votes[layer_name] = layer_vote
            
            # Determine momentum/meanrev/regime votes
            momentum_vote = layer_votes.get("momentum", Action.HOLD)
            meanrev_vote = layer_votes.get("mean_reversion", Action.HOLD)
            regime_vote = layer_votes.get("regime_aware", Action.HOLD)
            
            # Emotive bits (simplified - use mood from state if available)
            mood_bucket = state_dict.get("mood", 0) if isinstance(state_dict.get("mood"), int) else 0
            emote_bits = self._emotion.render(mood_bucket, comm_action)
            
            trace.append({
                "q_buy": 1 if action == Action.BUY else 0,
                "q_sell": 1 if action == Action.SELL else 0,
                "risk_budget_ok": 1,
                "q_emote_positive": emote_bits.get("positive", 0),
                "q_emote_alert": emote_bits.get("alert", 0),
                "q_emote_persistence": emote_bits.get("persistence", 0),
                "q_regime": int(state_dict.get("regime", 0.0) * 16) % 32,
                "price_up": 1 if state_dict.get("trend", 0.0) > 0 else 0,
                "price_down": 1 if state_dict.get("trend", 0.0) < 0 else 0,
                "weight_momentum": 1 if layer_weights.get("momentum", 0.0) > 0.3 else 0,
                "weight_contra": 1 if layer_weights.get("mean_reversion", 0.0) > 0.3 else 0,
                "weight_trend": 1 if layer_weights.get("regime_aware", 0.0) > 0.3 else 0,
            })
            
            obs, _, _ = self._step_env(action)
            
            obs, _, _ = self._step_env(action)
        
        return trace
    
    def stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "mean_reward": np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            "episode_rewards": self._episode_rewards,
            "comm_action_counts": self._comm_action_counts,
        }

