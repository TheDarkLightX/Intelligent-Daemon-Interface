#!/usr/bin/env python3
"""
Full Training Pipeline with Performance Iteration
- Combines emotional intelligence + scaled Q-tables
- Automated training loops
- Performance tracking and improvement
- Emoji-rich output
"""

import os
import json
import sys
import time
import csv
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase8_emoji"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase9_scaled"))


# ============================================================================
# SEEDING / DETERMINISM
# ============================================================================

def set_seed_from_env(default: int = 42) -> int:
    """Set numpy/Python RNG seed from SEED env var for reproducibility."""
    seed_str = os.getenv("SEED")
    try:
        seed = int(seed_str) if seed_str is not None else default
    except ValueError:
        seed = default
    np.random.seed(seed)
    return seed


def get_env_int(name: str, default: int) -> int:
    """Read an int env var with a default."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# ============================================================================
# EMOJI DEFINITIONS
# ============================================================================

class Emoji:
    # Performance
    ROCKET = "🚀"
    FIRE = "🔥"
    TROPHY = "🏆"
    MEDAL_GOLD = "🥇"
    MEDAL_SILVER = "🥈"
    MEDAL_BRONZE = "🥉"
    CHART_UP = "📈"
    CHART_DOWN = "📉"
    MONEY = "💰"
    DIAMOND = "💎"
    
    # Emotions
    HAPPY = "😊"
    EXCITED = "🤩"
    NEUTRAL = "😐"
    WORRIED = "😟"
    SCARED = "😰"
    CONFIDENT = "😎"
    THINKING = "🤔"
    
    # Actions
    BUY = "🟢"
    SELL = "🔴"
    HOLD = "⏸️"
    BURN = "🔥"
    
    # Status
    CHECK = "✅"
    CROSS = "❌"
    STAR = "⭐"
    LIGHTNING = "⚡"
    BRAIN = "🧠"
    GEAR = "⚙️"
    TARGET = "🎯"
    PROGRESS = "📊"


# ============================================================================
# MEGA Q-TABLE (1000+ states)
# ============================================================================

class MegaQTable:
    """
    Large-scale Q-table with:
    - 1000+ discrete states
    - Function approximation for unseen states
    - Experience replay
    - Prioritized updates
    """
    
    def __init__(self, n_states: int, n_actions: int = 4):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Main Q-table (sparse)
        self.q: Dict[int, np.ndarray] = {}
        self.visits: Dict[int, np.ndarray] = {}
        
        # Experience replay buffer
        self.replay_buffer: List[Tuple] = []
        self.buffer_size = 10000
        
        # Prioritized replay
        self.priorities: List[float] = []
        
        # Learning params
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        
        # Stats
        self.total_updates = 0
        self.unique_states = 0
        # Metrics logging
        self.episode_rewards: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_wins: List[int] = []

    # --------------------------- Persistence --------------------------------
    def save(self, path: Path) -> None:
        """Save Q-table using safe serialization (JSON + NPZ).
        
        Preconditions:
            - path is a valid writable path
            - Model state is consistent
        
        Postconditions:
            - Creates {path}.meta.json and {path}.arrays.npz
            - SHA-256 digest stored in metadata for integrity
        """
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        arrays_path = path.with_suffix(".arrays.npz")
        
        # Convert sparse dicts to arrays for NPZ storage
        # Store state indices and corresponding arrays
        q_states = list(self.q.keys())
        q_values = np.array([self.q[s] for s in q_states]) if q_states else np.array([])
        visit_values = np.array([self.visits[s] for s in q_states]) if q_states else np.array([])
        
        # Save arrays (data-only, no code execution)
        np.savez_compressed(
            arrays_path,
            q_states=np.array(q_states, dtype=np.int64),
            q_values=q_values,
            visit_values=visit_values,
            episode_rewards=np.array(self.episode_rewards),
            episode_trades=np.array(self.episode_trades),
            episode_wins=np.array(self.episode_wins),
            priorities=np.array(self.priorities),
        )
        
        # Compute integrity digest
        with open(arrays_path, "rb") as f:
            arrays_digest = hashlib.sha256(f.read()).hexdigest()
        
        # Save metadata as JSON (safe, no code execution)
        # Note: replay_buffer contains tuples which we serialize as lists
        replay_serializable = [list(t) for t in self.replay_buffer]
        
        metadata = {
            "version": 2,
            "format": "json+npz",
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "unique_states": self.unique_states,
            "replay_buffer": replay_serializable,
            "arrays_sha256": arrays_digest,
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"Saved to {meta_path} and {arrays_path}")

    @classmethod
    def load(cls, path: Path, *, verify_integrity: bool = True) -> "MegaQTable":
        """Load Q-table using safe deserialization.
        
        Preconditions:
            - path exists with .meta.json and .arrays.npz files
        
        Postconditions:
            - Model state restored from files
            - If verify_integrity=True, SHA-256 digest verified
        
        Args:
            path: Base path (without extension)
            verify_integrity: Whether to verify SHA-256 digest
            
        Raises:
            ValueError: If integrity check fails or legacy format detected
            FileNotFoundError: If required files missing
        """
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        arrays_path = path.with_suffix(".arrays.npz")
        
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
        
        # Create instance
        obj = cls(metadata["n_states"], metadata["n_actions"])
        obj.lr = metadata.get("lr", obj.lr)
        obj.gamma = metadata.get("gamma", obj.gamma)
        obj.epsilon = metadata.get("epsilon", obj.epsilon)
        obj.total_updates = metadata.get("total_updates", 0)
        obj.unique_states = metadata.get("unique_states", 0)
        
        # Restore replay buffer (convert lists back to tuples)
        obj.replay_buffer = [tuple(t) for t in metadata.get("replay_buffer", [])]
        
        # Load arrays (NumPy NPZ is data-only, no code execution)
        with np.load(arrays_path, allow_pickle=False) as data:
            q_states = data["q_states"]
            q_values = data["q_values"]
            visit_values = data["visit_values"]
            
            # Reconstruct sparse dicts
            obj.q = {int(s): q_values[i] for i, s in enumerate(q_states)}
            obj.visits = {int(s): visit_values[i] for i, s in enumerate(q_states)}
            
            obj.episode_rewards = data["episode_rewards"].tolist()
            obj.episode_trades = data["episode_trades"].tolist()
            obj.episode_wins = data["episode_wins"].tolist()
            obj.priorities = data["priorities"].tolist()
        
        print(f"Loaded from {meta_path} and {arrays_path}")
        return obj
    
    def get_q(self, state: int) -> np.ndarray:
        """Get Q-values with lazy initialization"""
        if state not in self.q:
            # Initialize with small random values
            self.q[state] = np.random.randn(self.n_actions) * 0.01
            self.visits[state] = np.zeros(self.n_actions)
            self.unique_states += 1
        return self.q[state]
    
    def select_action(self, state: int, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select action with exploration"""
        q_values = self.get_q(state)
        
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # UCB exploration bonus
            visits = self.visits.get(state, np.ones(self.n_actions))
            ucb = np.sqrt(2 * np.log(self.total_updates + 1) / (visits + 1))
            action = np.argmax(q_values + 0.1 * ucb)
        
        return action, q_values
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool = False):
        """Update Q-value and add to replay buffer"""
        # Add to replay buffer
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # Compute TD error for priority
        q = self.get_q(state)
        next_q = self.get_q(next_state)
        target = reward if done else reward + self.gamma * np.max(next_q)
        td_error = abs(target - q[action])
        self.priorities.append(td_error + 0.01)
        if len(self.priorities) > self.buffer_size:
            self.priorities.pop(0)
        
        # Update
        q[action] += self.lr * (target - q[action])
        self.visits[state][action] += 1
        self.total_updates += 1
        # Track simple stats per step (aggregated later)
        self._last_reward = reward
        self._last_action = action
    
    def replay(self, batch_size: int = 32, alpha: float = 0.6, beta: float = 0.4):
        """Experience replay with prioritized sampling"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Prioritized sampling
        priorities = np.array(self.priorities, dtype=float)
        if priorities.sum() <= 0:
            priorities = np.ones_like(priorities)
        probs = (priorities ** alpha) / np.sum(priorities ** alpha)
        indices = np.random.choice(len(self.replay_buffer), batch_size, 
                                  p=probs, replace=False)
        
        for idx in indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            q = self.get_q(state)
            next_q = self.get_q(next_state)
            target = reward if done else reward + self.gamma * np.max(next_q)
            # Importance sampling weight (stabilize)
            weight = (1 / (len(self.replay_buffer) * probs[idx])) ** beta
            weight /= np.max((1 / (len(self.replay_buffer) * probs)) ** beta)
            q[action] += self.lr * 0.5 * weight * (target - q[action])  # Smaller LR for replay
    
    def get_stats(self) -> Dict:
        return {
            "n_states": self.n_states,
            "unique_states": self.unique_states,
            "coverage": self.unique_states / self.n_states * 100,
            "total_updates": self.total_updates,
            "replay_buffer_size": len(self.replay_buffer),
            "episodes": len(self.episode_rewards),
            "total_reward": float(np.sum(self.episode_rewards)) if self.episode_rewards else 0.0,
            "trades": int(np.sum(self.episode_trades)) if self.episode_trades else 0,
            "wins": int(np.sum(self.episode_wins)) if self.episode_wins else 0,
            "win_rate": (np.sum(self.episode_wins) / max(1, np.sum(self.episode_trades)))
                        if self.episode_trades else 0.0,
        }


# ============================================================================
# FEATURE EXTRACTOR (1000+ dimensions)
# ============================================================================

class RichFeatureExtractor:
    """
    Extracts rich features for large state space
    
    Features:
    - Price momentum (multiple timeframes)
    - Volatility (multiple timeframes)
    - Position info (duration, pnl buckets)
    - Streak info
    - Market regime indicators
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Feature dimensions
        self.momentum_bins = config.get("momentum", 9)  # -4 to +4
        self.momentum_tf = config.get("momentum_timeframes", 3)  # short, med, long
        self.volatility_bins = config.get("volatility", 5)
        self.position_bins = config.get("position", 4)  # idle, profit_small, profit_big, loss
        self.duration_bins = config.get("duration", 5)
        self.streak_bins = config.get("streak", 7)  # -3 to +3
        self.regime_bins = config.get("regime", 3)  # bull, neutral, bear
        
        # Calculate state space size
        self.n_states = (self.momentum_bins ** self.momentum_tf * 
                        self.volatility_bins * 
                        self.position_bins * 
                        self.duration_bins * 
                        self.streak_bins * 
                        self.regime_bins)
    
    def extract(self, market_history: List[float], position: bool,
                entry_price: float, hold_duration: int, streak: int) -> int:
        """Extract features and encode to state index"""
        
        # Momentum at different timeframes
        momentums = []
        for tf in [5, 15, 30]:
            if len(market_history) >= tf:
                m = (market_history[-1] - market_history[-tf]) / market_history[-tf]
                m_bin = self._bin(m, -0.1, 0.1, self.momentum_bins)
            else:
                m_bin = self.momentum_bins // 2
            momentums.append(m_bin)
        
        # Volatility
        if len(market_history) >= 20:
            returns = np.diff(market_history[-20:]) / np.array(market_history[-20:-1])
            vol = np.std(returns)
        else:
            vol = 0.02
        vol_bin = self._bin(vol, 0, 0.1, self.volatility_bins)
        
        # Position state
        if not position:
            pos_bin = 0
        else:
            pnl = (market_history[-1] - entry_price) / entry_price if entry_price > 0 else 0
            if pnl > 0.05:
                pos_bin = 2  # Big profit
            elif pnl > 0:
                pos_bin = 1  # Small profit
            else:
                pos_bin = 3  # Loss
        
        # Duration
        dur_bin = min(hold_duration // 10, self.duration_bins - 1)
        
        # Streak
        streak_bin = self._bin(streak, -3, 3, self.streak_bins)
        
        # Regime
        if len(market_history) >= 50:
            long_trend = (market_history[-1] - market_history[-50]) / market_history[-50]
            if long_trend > 0.02:
                regime_bin = 0  # Bull
            elif long_trend < -0.02:
                regime_bin = 2  # Bear
            else:
                regime_bin = 1  # Neutral
        else:
            regime_bin = 1
        
        # Combine into single index
        idx = momentums[0]
        idx = idx * self.momentum_bins + momentums[1]
        idx = idx * self.momentum_bins + momentums[2]
        idx = idx * self.volatility_bins + vol_bin
        idx = idx * self.position_bins + pos_bin
        idx = idx * self.duration_bins + dur_bin
        idx = idx * self.streak_bins + streak_bin
        idx = idx * self.regime_bins + regime_bin
        
        return min(idx, self.n_states - 1)
    
    def _bin(self, value: float, min_v: float, max_v: float, n_bins: int) -> int:
        normalized = (value - min_v) / (max_v - min_v + 1e-8)
        normalized = np.clip(normalized, 0, 0.999)
        return int(normalized * n_bins)


# ============================================================================
# EMOTIONAL LAYER
# ============================================================================

@dataclass
class EmotionState:
    """Emotional state affecting decisions"""
    fear: float = 0.3
    greed: float = 0.3
    confidence: float = 0.5
    patience: float = 0.5
    
    streak: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    def update_from_trade(self, pnl: float):
        """Update emotions after trade"""
        if pnl > 0:
            self.streak = max(1, self.streak + 1)
            self.confidence = min(1.0, self.confidence + 0.1)
            self.greed = min(1.0, self.greed + 0.05)
            self.fear = max(0.0, self.fear - 0.1)
        else:
            self.streak = min(-1, self.streak - 1)
            self.confidence = max(0.0, self.confidence - 0.15)
            self.fear = min(1.0, self.fear + 0.15)
            self.greed = max(0.0, self.greed - 0.1)
        
        # Patience affected by streak
        if self.streak < -2:
            self.patience = max(0.1, self.patience - 0.1)
        elif self.streak > 2:
            self.patience = min(1.0, self.patience + 0.05)
    
    def get_emoji(self) -> str:
        """Get emoji representing current emotion"""
        score = (self.confidence * 0.4 + (1 - self.fear) * 0.3 + 
                self.greed * 0.15 + self.patience * 0.15)
        
        if score > 0.8:
            return Emoji.EXCITED
        elif score > 0.6:
            return Emoji.HAPPY
        elif score > 0.4:
            return Emoji.NEUTRAL
        elif score > 0.25:
            return Emoji.WORRIED
        else:
            return Emoji.SCARED
    
    def get_risk_multiplier(self) -> float:
        """Get risk adjustment based on emotions"""
        # More fear = less risk, more greed = more risk
        return 0.5 + 0.5 * (self.greed - self.fear + 1) / 2


# ============================================================================
# MEGA AGENT
# ============================================================================

class MegaQAgent:
    """
    Full-featured Q-agent with:
    - 1000+ state Q-table
    - Emotional intelligence
    - Emoji communication
    - Performance tracking
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Feature extractor
        self.features = RichFeatureExtractor(config.get("features", {}))
        print(f"{Emoji.BRAIN} State space: {self.features.n_states:,} states")
        
        # Q-table
        self.q_table = MegaQTable(self.features.n_states)
        
        # Emotional state
        self.emotions = EmotionState()
        
        # Trading state
        self.position = False
        self.entry_price = 0.0
        self.hold_duration = 0
        self.streak = 0
        
        # Market history
        self.price_history: List[float] = [100.0]
        
        # Performance tracking
        self.total_reward = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.episode_rewards: List[float] = []
        self.equity_curve: List[float] = [1.0]
        
        # Communication
        self.messages: List[str] = []
    
    def step(self, price_up: bool, price_down: bool) -> Tuple[int, float, str]:
        """Execute one step"""
        # Update price
        change = 0.01 if price_up else -0.01 if price_down else 0
        new_price = self.price_history[-1] * (1 + change)
        self.price_history.append(new_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Get state
        state = self.features.extract(
            self.price_history,
            self.position,
            self.entry_price,
            self.hold_duration,
            self.streak
        )
        
        # Select action (emotion affects epsilon)
        emotion_epsilon = self.q_table.epsilon * (1 + self.emotions.fear * 0.5)
        self.q_table.epsilon = min(0.5, emotion_epsilon)
        action, q_values = self.q_table.select_action(state)
        
        # Validate action
        if action == 1 and self.position:  # BUY while holding
            action = 0
        elif action == 2 and not self.position:  # SELL while idle
            action = 0
        
        # Execute
        reward = 0.0
        pnl = 0.0
        
        if action == 1:  # BUY
            self.position = True
            self.entry_price = new_price
            self.hold_duration = 0
            reward = -0.01 * (1 + self.emotions.greed * 0.5)  # Greed increases cost
        
        elif action == 2:  # SELL
            pnl = (new_price - self.entry_price) / self.entry_price
            reward = 1.0 + pnl * 10 * self.emotions.get_risk_multiplier()
            
            self.position = False
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                self.streak = max(1, self.streak + 1)
            else:
                self.streak = min(-1, self.streak - 1)
            
            self.emotions.update_from_trade(pnl)
        
        else:  # HOLD/WAIT
            if self.position:
                self.hold_duration += 1
                # Patience reward
                reward = 0.001 * self.emotions.patience
            else:
                reward = 0.0005
        
        # Update equity
        if self.position:
            unrealized = (new_price - self.entry_price) / self.entry_price
            self.equity_curve.append(self.equity_curve[-1] * (1 + unrealized * 0.01))
        else:
            self.equity_curve.append(self.equity_curve[-1])
        
        # Get next state
        next_state = self.features.extract(
            self.price_history,
            self.position,
            self.entry_price,
            self.hold_duration,
            self.streak
        )
        
        # Update Q-table
        self.q_table.update(state, action, reward, next_state)
        
        # Replay
        if self.q_table.total_updates % 100 == 0:
            self.q_table.replay(32)
        
        self.total_reward += reward
        
        # Generate message
        message = self._generate_message(action, reward, pnl, q_values)
        self.messages.append(message)
        
        return action, reward, message
    
    def _generate_message(self, action: int, reward: float, pnl: float, 
                         q_values: np.ndarray) -> str:
        """Generate emoji communication"""
        parts = []
        
        # Emotion
        parts.append(self.emotions.get_emoji())
        
        # Action
        action_emoji = [Emoji.HOLD, Emoji.BUY, Emoji.SELL, Emoji.HOLD][action]
        parts.append(action_emoji)
        
        # Confidence bar
        conf = self.emotions.confidence
        bar = "█" * int(conf * 5) + "░" * (5 - int(conf * 5))
        parts.append(f"[{bar}]")
        
        # Special indicators
        if self.streak > 2:
            parts.append(Emoji.FIRE)
        if self.streak < -2:
            parts.append("💀")
        if pnl > 0.05:
            parts.append(Emoji.MONEY)
        if self.emotions.greed > 0.7:
            parts.append(Emoji.DIAMOND)
        if self.emotions.fear > 0.7:
            parts.append("🚨")
        
        # Reward indicator
        if reward > 0.5:
            parts.append(Emoji.CHART_UP)
        elif reward < -0.5:
            parts.append(Emoji.CHART_DOWN)
        
        return " ".join(parts)
    
    def run_episode(self, market_data: List[Tuple[bool, bool]], 
                   verbose: bool = False) -> float:
        """Run one episode"""
        episode_reward = 0.0
        
        for step, (price_up, price_down) in enumerate(market_data):
            action, reward, message = self.step(price_up, price_down)
            episode_reward += reward
            
            if verbose and step % 20 == 0:
                signal = Emoji.CHART_UP if price_up else Emoji.CHART_DOWN
                print(f"Step {step:3d}: {signal} {message} R:{reward:+.3f}")
        
        self.episode_rewards.append(episode_reward)
        return episode_reward
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats"""
        q_stats = self.q_table.get_stats()
        
        return {
            "total_reward": self.total_reward,
            "trades": self.total_trades,
            "wins": self.winning_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "episodes": len(self.episode_rewards),
            "avg_episode_reward": np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            "best_episode": max(self.episode_rewards) if self.episode_rewards else 0,
            "final_equity": self.equity_curve[-1],
            "max_drawdown": min(np.array(self.equity_curve) / np.maximum.accumulate(self.equity_curve)) - 1,
            "emotion": self.emotions.get_emoji(),
            "confidence": self.emotions.confidence,
            "fear": self.emotions.fear,
            "greed": self.emotions.greed,
            "streak": self.streak,
            **q_stats
        }
    
    def print_status(self):
        """Print rich status"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"{Emoji.BRAIN} Agent Status {stats['emotion']}")
        print(f"{'='*60}")
        
        print(f"{Emoji.MONEY} Total Reward: {stats['total_reward']:.2f}")
        print(f"{Emoji.TARGET} Trades: {stats['trades']} (Win: {stats['win_rate']*100:.1f}%)")
        print(f"{Emoji.CHART_UP} Final Equity: {stats['final_equity']:.4f}")
        print(f"{Emoji.CHART_DOWN} Max Drawdown: {stats['max_drawdown']*100:.1f}%")
        
        print(f"\n{Emoji.THINKING} Emotions:")
        print(f"  Fear:       {'█'*int(stats['fear']*10)}{'░'*(10-int(stats['fear']*10))} {stats['fear']:.2f}")
        print(f"  Greed:      {'█'*int(stats['greed']*10)}{'░'*(10-int(stats['greed']*10))} {stats['greed']:.2f}")
        print(f"  Confidence: {'█'*int(stats['confidence']*10)}{'░'*(10-int(stats['confidence']*10))} {stats['confidence']:.2f}")
        print(f"  Streak: {stats['streak']}")
        
        print(f"\n{Emoji.GEAR} Q-Table:")
        print(f"  States: {stats['unique_states']:,} / {stats['n_states']:,} ({stats['coverage']:.2f}%)")
        print(f"  Updates: {stats['total_updates']:,}")
        print(f"  Replay Buffer: {stats['replay_buffer_size']:,}")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def generate_market(n_steps: int, regime: str = "mixed") -> List[Tuple[bool, bool]]:
    """Generate market data"""
    data = []
    
    if regime == "bull":
        prob = 0.6
    elif regime == "bear":
        prob = 0.4
    elif regime == "volatile":
        prob = 0.5
    else:
        prob = 0.5
    
    for i in range(n_steps):
        # Add some regime switching
        if regime == "volatile":
            prob = 0.3 + 0.4 * np.sin(i * 0.1)
        
        up = np.random.random() < prob
        data.append((up, not up))
    
    return data


def run_training_pipeline():
    """Full training pipeline with iterations"""
    
    print(f"{'='*70}")
    print(f"{Emoji.ROCKET} MEGA Q-AGENT TRAINING PIPELINE {Emoji.ROCKET}")
    print(f"{'='*70}")
    
    # Configuration
    config = {
        "features": {
            "momentum": 7,
            "momentum_timeframes": 3,
            "volatility": 5,
            "position": 4,
            "duration": 5,
            "streak": 7,
            "regime": 3
        }
    }
    
    agent = MegaQAgent(config)
    
    # Training phases
    phases = [
        ("Warmup", 20, 100, 0.3),
        ("Learning", 50, 200, 0.2),
        ("Refinement", 100, 300, 0.1),
        ("Mastery", 200, 300, 0.05),
    ]
    
    results = []
    
    for phase_name, n_episodes, steps, epsilon in phases:
        print(f"\n{Emoji.STAR} Phase: {phase_name}")
        print(f"  Episodes: {n_episodes}, Steps: {steps}, ε: {epsilon}")
        print("-" * 50)
        
        agent.q_table.epsilon = epsilon
        phase_rewards = []
        
        regimes = ["mixed", "bull", "bear", "volatile"]
        
        for ep in range(n_episodes):
            regime = regimes[ep % len(regimes)]
            market = generate_market(steps, regime)
            reward = agent.run_episode(market, verbose=False)
            phase_rewards.append(reward)
            
            if (ep + 1) % max(1, n_episodes // 5) == 0:
                avg = np.mean(phase_rewards[-10:])
                stats = agent.get_stats()
                print(f"  Ep {ep+1:3d}: {agent.emotions.get_emoji()} "
                      f"Reward={avg:7.2f} Win={stats['win_rate']*100:5.1f}% "
                      f"Cov={stats['coverage']:.1f}%")
        
        phase_avg = np.mean(phase_rewards)
        results.append({
            "phase": phase_name,
            "avg_reward": phase_avg,
            "best": max(phase_rewards),
            "win_rate": agent.get_stats()["win_rate"]
        })
        
        print(f"  {Emoji.CHECK} Phase complete: avg={phase_avg:.2f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{Emoji.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    agent.q_table.epsilon = 0.0  # No exploration
    
    for regime in ["bull", "bear", "volatile", "mixed"]:
        market = generate_market(500, regime)
        reward = agent.run_episode(market, verbose=False)
        stats = agent.get_stats()
        print(f"  {regime:10}: Reward={reward:8.2f} Win={stats['win_rate']*100:5.1f}%")
    
    agent.print_status()
    
    # Show recent messages
    print(f"\n{Emoji.LIGHTNING} Recent Communications:")
    for msg in agent.messages[-10:]:
        print(f"  {msg}")
    
    # Phase comparison
    print(f"\n{Emoji.PROGRESS} Training Progress:")
    for r in results:
        bar = "█" * int(r["avg_reward"] / 5) if r["avg_reward"] > 0 else ""
        print(f"  {r['phase']:12}: {bar} {r['avg_reward']:.2f}")
    
    return agent, results


if __name__ == "__main__":
    agent, results = run_training_pipeline()

