#!/usr/bin/env python3
"""
Scaled Q-Learning System
- 100s to 1000s of states
- Multiple training regimes
- Performance optimization
- Hyperparameter tuning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import IntEnum
import time
import json
import hashlib
from pathlib import Path


# ============================================================================
# COMPACT STATE ENCODING
# ============================================================================

class CompactStateEncoder:
    """
    Efficient state encoding with configurable granularity
    
    Features encoded:
    1. Price momentum (short-term)
    2. Price momentum (long-term)
    3. Volatility level
    4. Position status
    5. PnL status
    6. Streak status
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Configurable bins per feature
        self.momentum_short_bins = config.get("momentum_short", 5)
        self.momentum_long_bins = config.get("momentum_long", 5)
        self.volatility_bins = config.get("volatility", 4)
        self.position_bins = 3  # idle, profit, loss
        self.pnl_bins = config.get("pnl", 5)
        self.streak_bins = config.get("streak", 5)
        
        self.n_states = (self.momentum_short_bins * 
                        self.momentum_long_bins * 
                        self.volatility_bins * 
                        self.position_bins * 
                        self.pnl_bins * 
                        self.streak_bins)
        
        # Feature bounds
        self.bounds = {
            "momentum_short": (-0.5, 0.5),
            "momentum_long": (-0.3, 0.3),
            "volatility": (0, 0.3),
            "pnl": (-0.3, 0.3),
            "streak": (-5, 5)
        }
    
    def _discretize(self, value: float, min_v: float, max_v: float, n_bins: int) -> int:
        """Discretize value into bin"""
        normalized = (value - min_v) / (max_v - min_v + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        return min(int(normalized * n_bins), n_bins - 1)
    
    def encode(self, features: Dict) -> int:
        """Encode feature dict to state index"""
        m_short = self._discretize(features.get("momentum_short", 0),
                                   *self.bounds["momentum_short"], 
                                   self.momentum_short_bins)
        
        m_long = self._discretize(features.get("momentum_long", 0),
                                  *self.bounds["momentum_long"],
                                  self.momentum_long_bins)
        
        vol = self._discretize(features.get("volatility", 0),
                              *self.bounds["volatility"],
                              self.volatility_bins)
        
        # Position: 0=idle, 1=profit, 2=loss
        pos = 0
        if features.get("position", False):
            pos = 1 if features.get("unrealized_pnl", 0) >= 0 else 2
        
        pnl = self._discretize(features.get("unrealized_pnl", 0),
                              *self.bounds["pnl"],
                              self.pnl_bins)
        
        streak = self._discretize(features.get("streak", 0),
                                 *self.bounds["streak"],
                                 self.streak_bins)
        
        # Combine
        idx = m_short
        idx = idx * self.momentum_long_bins + m_long
        idx = idx * self.volatility_bins + vol
        idx = idx * self.position_bins + pos
        idx = idx * self.pnl_bins + pnl
        idx = idx * self.streak_bins + streak
        
        return idx


# ============================================================================
# HIERARCHICAL Q-TABLE
# ============================================================================

class HierarchicalQTable:
    """
    Multi-level Q-table with:
    - Primary table (main decisions)
    - Secondary tables (specialized strategies)
    - Meta-controller (chooses which table to use)
    """
    
    def __init__(self, n_states: int, n_actions: int = 4,
                 n_levels: int = 3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_levels = n_levels
        
        # Primary Q-table (always used)
        self.primary = np.zeros((n_states, n_actions))
        
        # Specialized tables
        self.specialized = {
            "momentum": np.zeros((n_states, n_actions)),
            "mean_revert": np.zeros((n_states, n_actions)),
            "defensive": np.zeros((n_states, n_actions)),
        }
        
        # Meta-controller: which strategy to use per state
        self.meta = np.ones((n_states, len(self.specialized))) / len(self.specialized)
        
        # Visit counts for UCB
        self.visits = np.zeros((n_states, n_actions))
        self.total_visits = 0
        
        # Learning parameters
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 0.2
        
        # Initialize with priors
        self._init_priors()
    
    def _init_priors(self):
        """Initialize tables with domain knowledge"""
        # Momentum: favor buying on up, selling on down
        # Mean revert: opposite
        # Defensive: favor holding
        pass  # Let learning discover
    
    def get_q_values(self, state: int) -> np.ndarray:
        """Get combined Q-values using meta-controller"""
        weights = self.meta[state]
        combined = self.primary[state].copy()
        
        for i, (name, table) in enumerate(self.specialized.items()):
            combined += weights[i] * table[state]
        
        return combined / (1 + weights.sum())
    
    def select_action(self, state: int, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select action with UCB exploration"""
        q_values = self.get_q_values(state)
        
        if explore:
            # UCB exploration bonus
            visits = self.visits[state] + 1
            ucb = np.sqrt(2 * np.log(self.total_visits + 1) / visits)
            
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = np.argmax(q_values + 0.5 * ucb)
        else:
            action = np.argmax(q_values)
        
        return action, q_values
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, context: Dict = None):
        """Update all relevant tables"""
        next_q = self.get_q_values(next_state)
        target = reward + self.gamma * np.max(next_q)
        
        # Update primary
        error = target - self.primary[state, action]
        self.primary[state, action] += self.lr * error
        
        # Update specialized based on context
        if context:
            if context.get("trend_up", False):
                self.specialized["momentum"][state, action] += self.lr * error
            if context.get("volatility_high", False):
                self.specialized["defensive"][state, action] += self.lr * error * 0.5
            if context.get("mean_reverting", False):
                self.specialized["mean_revert"][state, action] += self.lr * error
        
        # Update visit count
        self.visits[state, action] += 1
        self.total_visits += 1
        
        # Update meta-controller based on which strategy worked
        if reward > 0:
            # Boost weights for strategies that align with current context
            if context:
                if context.get("trend_up"):
                    self.meta[state, 0] *= 1.1  # momentum
                if context.get("mean_reverting"):
                    self.meta[state, 1] *= 1.1  # mean_revert
            # Normalize
            self.meta[state] /= self.meta[state].sum() + 1e-8


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================

@dataclass
class MarketSimulator:
    """Simulates different market conditions"""
    
    regime: str = "mixed"
    volatility: float = 0.1
    trend_strength: float = 0.0
    
    # Internal state
    price: float = 100.0
    history: List[float] = field(default_factory=list)
    
    def step(self) -> Tuple[bool, bool, Dict]:
        """Generate next price movement"""
        # Base random walk
        noise = np.random.randn() * self.volatility
        
        # Trend component
        trend = 0
        if self.regime == "bull":
            trend = 0.02
        elif self.regime == "bear":
            trend = -0.02
        elif self.regime == "volatile":
            trend = 0.05 * np.sin(len(self.history) * 0.1)
        
        # Mean reversion
        if len(self.history) > 20:
            ma = np.mean(self.history[-20:])
            reversion = 0.01 * (ma - self.price) / self.price
        else:
            reversion = 0
        
        # Apply
        change = trend + noise + reversion
        self.price *= (1 + change)
        self.history.append(self.price)
        
        price_up = change > 0
        price_down = change < 0
        
        # Context for learning
        context = {
            "trend_up": trend > 0,
            "trend_down": trend < 0,
            "volatility_high": abs(noise) > self.volatility,
            "mean_reverting": abs(reversion) > 0.005
        }
        
        return price_up, price_down, context
    
    def get_features(self, position: bool, entry_price: float, 
                    streak: int) -> Dict:
        """Extract features for state encoding"""
        h = self.history
        
        # Momentum
        if len(h) >= 5:
            m_short = (h[-1] - h[-5]) / h[-5] if h[-5] > 0 else 0
        else:
            m_short = 0
        
        if len(h) >= 20:
            m_long = (h[-1] - h[-20]) / h[-20] if h[-20] > 0 else 0
        else:
            m_long = 0
        
        # Volatility
        if len(h) >= 10:
            returns = np.diff(h[-10:]) / np.array(h[-10:-1])
            vol = np.std(returns)
        else:
            vol = self.volatility
        
        # PnL
        if position and entry_price > 0:
            pnl = (h[-1] - entry_price) / entry_price
        else:
            pnl = 0
        
        return {
            "momentum_short": m_short,
            "momentum_long": m_long,
            "volatility": vol,
            "position": position,
            "unrealized_pnl": pnl,
            "streak": streak
        }


# ============================================================================
# TRAINING SYSTEM
# ============================================================================

class TrainingSystem:
    """Complete training pipeline with hyperparameter optimization"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # State encoder
        self.encoder = CompactStateEncoder(self.config.get("encoder", {}))
        print(f"State space: {self.encoder.n_states:,} states")
        
        # Q-table
        self.q_table = HierarchicalQTable(
            self.encoder.n_states,
            n_actions=4
        )
        
        # Training stats
        self.episode_rewards: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_wins: List[int] = []
        self.best_reward = float('-inf')
        
        # Current episode state
        self.position = False
        self.entry_price = 0.0
        self.streak = 0
        self.episode_reward = 0.0
        self.trades = 0
        self.wins = 0
    
    def _default_config(self) -> Dict:
        return {
            "encoder": {
                "momentum_short": 7,
                "momentum_long": 5,
                "volatility": 5,
                "pnl": 7,
                "streak": 5
            },
            "training": {
                "episodes": 100,
                "steps_per_episode": 200,
                "warmup_episodes": 10
            },
            "hyperparams": {
                "lr": 0.15,
                "gamma": 0.95,
                "epsilon_start": 0.3,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.995
            }
        }
    
    def train_episode(self, market: MarketSimulator, verbose: bool = False) -> float:
        """Train one episode"""
        self.position = False
        self.entry_price = 0.0
        self.episode_reward = 0.0
        self.trades = 0
        self.wins = 0
        
        steps = self.config["training"]["steps_per_episode"]
        
        for step in range(steps):
            # Get market data
            price_up, price_down, context = market.step()
            
            # Get features and state
            features = market.get_features(self.position, self.entry_price, self.streak)
            state = self.encoder.encode(features)
            
            # Select action
            action, q_values = self.q_table.select_action(state)
            
            # Validate
            if action == 1 and self.position:  # BUY while holding
                action = 0  # HOLD
            elif action == 2 and not self.position:  # SELL while idle
                action = 0  # HOLD
            
            # Execute
            reward = 0.0
            if action == 1:  # BUY
                self.position = True
                self.entry_price = market.price
                reward = -0.01  # Transaction cost
            elif action == 2:  # SELL
                pnl = (market.price - self.entry_price) / self.entry_price
                reward = 1.0 + pnl * 5  # Base + PnL multiplier
                self.position = False
                self.trades += 1
                if pnl > 0:
                    self.wins += 1
                    self.streak = max(1, self.streak + 1)
                else:
                    self.streak = min(-1, self.streak - 1)
            else:  # HOLD or WAIT
                reward = 0.005 if self.position else 0.001
            
            # Get next state
            next_features = market.get_features(self.position, self.entry_price, self.streak)
            next_state = self.encoder.encode(next_features)
            
            # Update Q-table
            self.q_table.update(state, action, reward, next_state, context)
            
            self.episode_reward += reward
            
            if verbose and step % 50 == 0:
                print(f"  Step {step}: action={action}, reward={reward:.3f}, "
                      f"total={self.episode_reward:.2f}")
        
        self.episode_rewards.append(self.episode_reward)
        self.episode_trades.append(self.trades)
        self.episode_wins.append(self.wins)
        
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        
        return self.episode_reward
    
    def train(self, n_episodes: int = None, verbose: bool = True) -> Dict:
        """Full training loop"""
        n_episodes = n_episodes or self.config["training"]["episodes"]
        
        print(f"\n{'='*60}")
        print(f"🎓 Training {n_episodes} episodes")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Different market regimes for diverse training
        regimes = ["mixed", "bull", "bear", "volatile"]
        
        for ep in range(n_episodes):
            # Rotate through regimes
            regime = regimes[ep % len(regimes)]
            market = MarketSimulator(regime=regime, volatility=0.05)
            
            # Decay epsilon
            hp = self.config["hyperparams"]
            self.q_table.epsilon = max(
                hp["epsilon_end"],
                hp["epsilon_start"] * (hp["epsilon_decay"] ** ep)
            )
            
            reward = self.train_episode(market, verbose=False)
            
            if verbose and (ep + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_trades = np.mean(self.episode_trades[-10:])
                win_rate = sum(self.episode_wins[-10:]) / max(1, sum(self.episode_trades[-10:]))
                coverage = self.q_table.total_visits / self.encoder.n_states * 100
                
                print(f"Ep {ep+1:4d} | Reward: {avg_reward:7.2f} | "
                      f"Trades: {avg_trades:5.1f} | Win: {win_rate*100:5.1f}% | "
                      f"ε: {self.q_table.epsilon:.3f} | Cov: {coverage:.1f}%")
        
        elapsed = time.time() - start_time
        
        # Final stats
        stats = self.get_stats()
        stats["training_time"] = elapsed
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete in {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"Best Reward: {self.best_reward:.2f}")
        print(f"Final Avg Reward: {stats['avg_reward']:.2f}")
        print(f"State Coverage: {stats['coverage']:.2f}%")
        
        return stats
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "episodes": len(self.episode_rewards),
            "total_reward": sum(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            "best_reward": self.best_reward,
            "total_trades": sum(self.episode_trades),
            "total_wins": sum(self.episode_wins),
            "win_rate": sum(self.episode_wins) / max(1, sum(self.episode_trades)),
            "coverage": self.q_table.total_visits / self.encoder.n_states * 100,
            "total_updates": self.q_table.total_visits
        }
    
    def _artifact_dir(self, base_path: Path) -> Path:
        """Resolve the artifact directory used for safe persistence.
        
        Preconditions:
            - base_path is a Path
        
        Postconditions:
            - Returned path has no file suffixes (prevents misleading extensions)
            - Returned path is suitable to be created as a directory
        """
        name = base_path.name
        for suffix in base_path.suffixes:
            if not suffix:
                continue
            name = name[: -len(suffix)]
        if not name:
            name = base_path.name
        return base_path.with_name(name)

    def save(self, path: str) -> None:
        """Save model using safe serialization (JSON + NPZ).
        
        Preconditions:
            - path is a valid writable path
            - Model state is consistent
        
        Postconditions:
            - Creates <path>/metadata.json and <path>/arrays.npz
            - SHA-256 digest stored in metadata for integrity
        """
        base_path = Path(path)
        artifact_dir = self._artifact_dir(base_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        meta_path = artifact_dir / "metadata.json"
        arrays_path = artifact_dir / "arrays.npz"
        
        # Save arrays separately (data-only, no code execution)
        np.savez_compressed(
            arrays_path,
            q_primary=self.q_table.primary,
            q_specialized=self.q_table.specialized,
            meta=self.q_table.meta,
            visits=self.q_table.visits,
        )
        
        # Compute integrity digest
        with open(arrays_path, "rb") as f:
            arrays_digest = hashlib.sha256(f.read()).hexdigest()
        
        # Save metadata as JSON (safe, no code execution)
        metadata = {
            "version": 2,
            "format": "json+npz",
            "config": self.config,
            "stats": self.get_stats(),
            "arrays_sha256": arrays_digest,
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"Saved to {meta_path} and {arrays_path}")
    
    def load(self, path: str, *, verify_integrity: bool = True) -> None:
        """Load model using safe deserialization.
        
        Preconditions:
            - path exists either as:
              - <path>/metadata.json and <path>/arrays.npz (preferred)
              - legacy: <path>.meta.json and <path>.arrays.npz
        
        Postconditions:
            - Model state restored from files
            - If verify_integrity=True, SHA-256 digest verified
        
        Args:
            path: Base path (without extension)
            verify_integrity: Whether to verify SHA-256 digest
            
        Raises:
            ValueError: If integrity check fails
            FileNotFoundError: If required files missing
        """
        base_path = Path(path)
        if base_path.is_dir():
            meta_path = base_path / "metadata.json"
            arrays_path = base_path / "arrays.npz"
        else:
            artifact_dir = self._artifact_dir(base_path)
            if artifact_dir.is_dir():
                meta_path = artifact_dir / "metadata.json"
                arrays_path = artifact_dir / "arrays.npz"
            else:
                meta_path = base_path.with_suffix(".meta.json")
                arrays_path = base_path.with_suffix(".arrays.npz")
        
        # Load metadata (JSON is safe)
        metadata = json.loads(meta_path.read_text())
        
        if metadata.get("version", 1) < 2:
            raise ValueError(
                f"Legacy pickle format detected at {path}. "
                "Please migrate to safe format using the migration script."
            )
        
        # Verify integrity before loading arrays
        if verify_integrity:
            with open(arrays_path, "rb") as f:
                actual_digest = hashlib.sha256(f.read()).hexdigest()
            expected_digest = metadata.get("arrays_sha256", "")
            if actual_digest != expected_digest:
                raise ValueError(
                    f"Integrity check failed for {arrays_path}. "
                    f"Expected {expected_digest[:16]}..., got {actual_digest[:16]}..."
                )
        
        # Load arrays (NumPy NPZ is data-only, no code execution)
        with np.load(arrays_path, allow_pickle=False) as data:
            self.q_table.primary = data["q_primary"]
            self.q_table.specialized = data["q_specialized"]
            self.q_table.meta = data["meta"]
            self.q_table.visits = data["visits"]
        
        print(f"Loaded from {meta_path} and {arrays_path}")


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

class HyperparameterTuner:
    """Grid search and random search for hyperparameters"""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def grid_search(self, param_grid: Dict, n_episodes: int = 50) -> Dict:
        """Grid search over hyperparameters"""
        print(f"\n{'='*60}")
        print(f"🔧 Hyperparameter Grid Search")
        print(f"{'='*60}")
        
        # Generate all combinations
        from itertools import product
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        best_params = None
        best_reward = float('-inf')
        
        for combo in product(*values):
            params = dict(zip(keys, combo))
            
            # Create config
            config = {
                "encoder": {
                    "momentum_short": params.get("momentum_bins", 5),
                    "momentum_long": params.get("momentum_bins", 5),
                    "volatility": params.get("volatility_bins", 4),
                    "pnl": params.get("pnl_bins", 5),
                    "streak": 5
                },
                "training": {
                    "episodes": n_episodes,
                    "steps_per_episode": 100,
                    "warmup_episodes": 5
                },
                "hyperparams": {
                    "lr": params.get("lr", 0.1),
                    "gamma": params.get("gamma", 0.95),
                    "epsilon_start": params.get("epsilon", 0.2),
                    "epsilon_end": 0.05,
                    "epsilon_decay": 0.99
                }
            }
            
            # Train
            trainer = TrainingSystem(config)
            stats = trainer.train(verbose=False)
            
            result = {
                "params": params,
                "reward": stats["avg_reward"],
                "win_rate": stats["win_rate"],
                "coverage": stats["coverage"]
            }
            self.results.append(result)
            
            print(f"  {params} -> reward={stats['avg_reward']:.2f}, "
                  f"win={stats['win_rate']*100:.1f}%")
            
            if stats["avg_reward"] > best_reward:
                best_reward = stats["avg_reward"]
                best_params = params
        
        print(f"\n🏆 Best params: {best_params}")
        print(f"   Best reward: {best_reward:.2f}")
        
        return {"best_params": best_params, "best_reward": best_reward}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 Scaled Q-Learning System")
    print("=" * 70)
    
    # Phase 1: Small scale training
    print("\n📊 Phase 1: Small Scale (1,575 states)")
    config_small = {
        "encoder": {
            "momentum_short": 5,
            "momentum_long": 3,
            "volatility": 3,
            "pnl": 5,
            "streak": 7
        },
        "training": {
            "episodes": 100,
            "steps_per_episode": 200
        },
        "hyperparams": {
            "lr": 0.2,
            "gamma": 0.95,
            "epsilon_start": 0.3,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99
        }
    }
    
    trainer_small = TrainingSystem(config_small)
    trainer_small.train(verbose=True)
    
    # Phase 2: Medium scale
    print("\n📊 Phase 2: Medium Scale (11,025 states)")
    config_med = {
        "encoder": {
            "momentum_short": 7,
            "momentum_long": 5,
            "volatility": 5,
            "pnl": 7,
            "streak": 9
        },
        "training": {
            "episodes": 200,
            "steps_per_episode": 200
        },
        "hyperparams": {
            "lr": 0.15,
            "gamma": 0.95,
            "epsilon_start": 0.25,
            "epsilon_end": 0.03,
            "epsilon_decay": 0.995
        }
    }
    
    trainer_med = TrainingSystem(config_med)
    trainer_med.train(verbose=True)
    
    # Phase 3: Large scale
    print("\n📊 Phase 3: Large Scale (33,075 states)")
    config_large = {
        "encoder": {
            "momentum_short": 9,
            "momentum_long": 7,
            "volatility": 5,
            "pnl": 9,
            "streak": 11
        },
        "training": {
            "episodes": 500,
            "steps_per_episode": 300
        },
        "hyperparams": {
            "lr": 0.1,
            "gamma": 0.97,
            "epsilon_start": 0.2,
            "epsilon_end": 0.02,
            "epsilon_decay": 0.998
        }
    }
    
    trainer_large = TrainingSystem(config_large)
    stats = trainer_large.train(verbose=True)
    
    # Save best model
    trainer_large.save("/tmp/q_model_large.pkl")
    
    # Quick hyperparameter search
    print("\n🔧 Quick Hyperparameter Search")
    tuner = HyperparameterTuner()
    tuner.grid_search({
        "lr": [0.1, 0.2],
        "gamma": [0.9, 0.95],
        "epsilon": [0.2, 0.3],
        "momentum_bins": [5, 7]
    }, n_episodes=30)
    
    print("\n✅ All phases complete!")


if __name__ == "__main__":
    main()

