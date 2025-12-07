#!/usr/bin/env python3
"""
Emotional Intelligence Q-Agent with Emoji Communication

This agent has emotional states that affect decision-making and
communicates via rich emoji expressions.
"""

import numpy as np
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time


# ============================================================================
# EMOJI VOCABULARY
# ============================================================================

class EmojiVocab:
    """Rich emoji vocabulary for agent communication"""
    
    # Market signals
    MARKET_UP = "📈"
    MARKET_DOWN = "📉"
    MARKET_FLAT = "➡️"
    MARKET_VOLATILE = "🎢"
    MARKET_CALM = "😴"
    
    # Actions
    ACTION_BUY = "🟢"
    ACTION_SELL = "🔴"
    ACTION_HOLD = "⏸️"
    ACTION_WAIT = "⏳"
    ACTION_BURN = "🔥"
    
    # Emotions
    EMOTION_HAPPY = "😊"
    EMOTION_EXCITED = "🤩"
    EMOTION_EUPHORIC = "🚀"
    EMOTION_NEUTRAL = "😐"
    EMOTION_WORRIED = "😟"
    EMOTION_SCARED = "😰"
    EMOTION_PANICKED = "😱"
    EMOTION_ANGRY = "😠"
    EMOTION_SAD = "😢"
    EMOTION_RELIEVED = "😌"
    EMOTION_CONFIDENT = "😎"
    EMOTION_CAUTIOUS = "🤔"
    EMOTION_GREEDY = "🤑"
    EMOTION_FEARFUL = "😨"
    
    # States
    STATE_IDLE = "💤"
    STATE_HOLDING = "🏠"
    STATE_PROFIT = "💰"
    STATE_LOSS = "📉"
    STATE_WINNING = "🏆"
    STATE_LOSING = "💔"
    
    # Intensity
    INTENSITY_LOW = "·"
    INTENSITY_MED = "○"
    INTENSITY_HIGH = "●"
    INTENSITY_MAX = "★"
    
    # Communication
    COMM_BULLISH = "🐂"
    COMM_BEARISH = "🐻"
    COMM_THINKING = "🧠"
    COMM_ALERT = "🚨"
    COMM_SUCCESS = "✅"
    COMM_FAILURE = "❌"
    COMM_MONEY = "💵"
    COMM_FIRE = "🔥"
    COMM_DIAMOND = "💎"
    COMM_HANDS = "🙌"
    COMM_ROCKET = "🚀"
    COMM_MOON = "🌙"
    COMM_SKULL = "💀"
    COMM_HEART = "❤️"


class Emotion(IntEnum):
    """Emotional states"""
    EUPHORIC = 0
    EXCITED = 1
    HAPPY = 2
    CONFIDENT = 3
    NEUTRAL = 4
    CAUTIOUS = 5
    WORRIED = 6
    SCARED = 7
    PANICKED = 8


class MarketRegime(IntEnum):
    """Market regime classification"""
    STRONG_BULL = 0
    BULL = 1
    WEAK_BULL = 2
    NEUTRAL = 3
    WEAK_BEAR = 4
    BEAR = 5
    STRONG_BEAR = 6


class Volatility(IntEnum):
    """Volatility levels"""
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4


class Momentum(IntEnum):
    """Momentum indicators"""
    STRONG_DOWN = 0
    DOWN = 1
    WEAK_DOWN = 2
    FLAT = 3
    WEAK_UP = 4
    UP = 5
    STRONG_UP = 6


class Action(IntEnum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    WAIT = 3


# ============================================================================
# EMOTIONAL STATE
# ============================================================================

@dataclass
class EmotionalState:
    """Agent's emotional state"""
    fear: float = 0.0          # 0-1, affects risk aversion
    greed: float = 0.0         # 0-1, affects risk seeking
    confidence: float = 0.5    # 0-1, affects position sizing
    patience: float = 0.5      # 0-1, affects hold duration
    excitement: float = 0.0    # 0-1, affects volatility seeking
    
    # Recent history affects emotions
    recent_wins: int = 0
    recent_losses: int = 0
    streak: int = 0  # positive = wins, negative = losses
    max_drawdown: float = 0.0
    peak_profit: float = 0.0
    
    def get_emotion(self) -> Emotion:
        """Compute dominant emotion"""
        score = self.confidence * 0.3 + (1 - self.fear) * 0.3 + self.greed * 0.2 + self.excitement * 0.2
        
        if score > 0.9:
            return Emotion.EUPHORIC
        elif score > 0.8:
            return Emotion.EXCITED
        elif score > 0.7:
            return Emotion.HAPPY
        elif score > 0.6:
            return Emotion.CONFIDENT
        elif score > 0.4:
            return Emotion.NEUTRAL
        elif score > 0.3:
            return Emotion.CAUTIOUS
        elif score > 0.2:
            return Emotion.WORRIED
        elif score > 0.1:
            return Emotion.SCARED
        else:
            return Emotion.PANICKED
    
    def get_emoji(self) -> str:
        """Get emoji for current emotion"""
        emotion = self.get_emotion()
        return {
            Emotion.EUPHORIC: EmojiVocab.EMOTION_EUPHORIC + EmojiVocab.COMM_ROCKET,
            Emotion.EXCITED: EmojiVocab.EMOTION_EXCITED,
            Emotion.HAPPY: EmojiVocab.EMOTION_HAPPY,
            Emotion.CONFIDENT: EmojiVocab.EMOTION_CONFIDENT,
            Emotion.NEUTRAL: EmojiVocab.EMOTION_NEUTRAL,
            Emotion.CAUTIOUS: EmojiVocab.EMOTION_CAUTIOUS,
            Emotion.WORRIED: EmojiVocab.EMOTION_WORRIED,
            Emotion.SCARED: EmojiVocab.EMOTION_SCARED,
            Emotion.PANICKED: EmojiVocab.EMOTION_PANICKED + EmojiVocab.COMM_SKULL,
        }[emotion]
    
    def update_after_trade(self, profit: float):
        """Update emotional state after a trade"""
        if profit > 0:
            self.recent_wins += 1
            self.streak = max(1, self.streak + 1)
            self.confidence = min(1.0, self.confidence + 0.1)
            self.greed = min(1.0, self.greed + 0.05)
            self.fear = max(0.0, self.fear - 0.1)
            self.peak_profit = max(self.peak_profit, profit)
        else:
            self.recent_losses += 1
            self.streak = min(-1, self.streak - 1)
            self.confidence = max(0.0, self.confidence - 0.15)
            self.fear = min(1.0, self.fear + 0.1)
            self.greed = max(0.0, self.greed - 0.05)
            self.max_drawdown = min(self.max_drawdown, profit)
        
        # Excitement from volatility
        self.excitement = min(1.0, abs(profit) * 0.5)
        
        # Patience decays with losses
        if self.streak < -2:
            self.patience = max(0.1, self.patience - 0.1)
        elif self.streak > 2:
            self.patience = min(1.0, self.patience + 0.05)
    
    def update_from_market(self, volatility: float, trend_strength: float):
        """Update emotions based on market conditions"""
        # High volatility increases fear and excitement
        self.fear = min(1.0, self.fear + volatility * 0.1)
        self.excitement = min(1.0, self.excitement + volatility * 0.2)
        
        # Strong trends affect greed/fear
        if trend_strength > 0.5:
            self.greed = min(1.0, self.greed + 0.1)
        elif trend_strength < -0.5:
            self.fear = min(1.0, self.fear + 0.1)
        
        # Decay towards neutral
        self.fear *= 0.95
        self.greed *= 0.95
        self.excitement *= 0.9


# ============================================================================
# LARGE-SCALE STATE ENCODING
# ============================================================================

@dataclass
class MarketFeatures:
    """Rich market feature set for state encoding"""
    # Price action (last N periods)
    price_changes: List[float] = field(default_factory=list)
    
    # Derived features
    momentum_short: float = 0.0   # 5-period
    momentum_long: float = 0.0    # 20-period
    volatility: float = 0.0
    trend_strength: float = 0.0
    
    # Position info
    position: bool = False
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_duration: int = 0
    
    # Historical performance
    total_trades: int = 0
    win_rate: float = 0.5
    avg_profit: float = 0.0
    
    def compute_derived(self):
        """Compute derived features from price changes"""
        if len(self.price_changes) >= 5:
            self.momentum_short = sum(self.price_changes[-5:]) / 5
        if len(self.price_changes) >= 20:
            self.momentum_long = sum(self.price_changes[-20:]) / 20
        if len(self.price_changes) >= 10:
            self.volatility = np.std(self.price_changes[-10:])
        if len(self.price_changes) >= 20:
            ups = sum(1 for p in self.price_changes[-20:] if p > 0)
            self.trend_strength = (ups - 10) / 10  # -1 to 1


class StateEncoder:
    """Encodes market features into discrete state indices"""
    
    def __init__(self, 
                 n_momentum_bins: int = 7,
                 n_volatility_bins: int = 5,
                 n_trend_bins: int = 7,
                 n_position_states: int = 4,
                 n_emotion_states: int = 9,
                 n_pnl_bins: int = 5):
        
        self.n_momentum = n_momentum_bins
        self.n_volatility = n_volatility_bins
        self.n_trend = n_trend_bins
        self.n_position = n_position_states
        self.n_emotion = n_emotion_states
        self.n_pnl = n_pnl_bins
        
        # Total state space size
        self.n_states = (n_momentum_bins * n_volatility_bins * n_trend_bins * 
                        n_position_states * n_emotion_states * n_pnl_bins)
        
        print(f"State space size: {self.n_states:,} states")
    
    def encode(self, features: MarketFeatures, emotions: EmotionalState) -> int:
        """Encode features to state index"""
        # Discretize momentum
        momentum_bin = self._bin_value(features.momentum_short, -1, 1, self.n_momentum)
        
        # Discretize volatility  
        vol_bin = self._bin_value(features.volatility, 0, 0.5, self.n_volatility)
        
        # Discretize trend
        trend_bin = self._bin_value(features.trend_strength, -1, 1, self.n_trend)
        
        # Position state: (idle, holding_profit, holding_loss, holding_flat)
        if not features.position:
            pos_bin = 0
        elif features.unrealized_pnl > 0.01:
            pos_bin = 1
        elif features.unrealized_pnl < -0.01:
            pos_bin = 2
        else:
            pos_bin = 3
        
        # Emotion state
        emotion_bin = emotions.get_emotion().value
        
        # PnL bin
        pnl_bin = self._bin_value(features.unrealized_pnl, -0.5, 0.5, self.n_pnl)
        
        # Combine into single index
        idx = momentum_bin
        idx = idx * self.n_volatility + vol_bin
        idx = idx * self.n_trend + trend_bin
        idx = idx * self.n_position + pos_bin
        idx = idx * self.n_emotion + emotion_bin
        idx = idx * self.n_pnl + pnl_bin
        
        return min(idx, self.n_states - 1)
    
    def _bin_value(self, value: float, min_val: float, max_val: float, n_bins: int) -> int:
        """Discretize continuous value into bins"""
        normalized = (value - min_val) / (max_val - min_val + 1e-6)
        normalized = max(0, min(1, normalized))
        return int(normalized * (n_bins - 1))
    
    def decode(self, idx: int) -> Dict:
        """Decode state index back to features (for debugging)"""
        pnl_bin = idx % self.n_pnl
        idx //= self.n_pnl
        emotion_bin = idx % self.n_emotion
        idx //= self.n_emotion
        pos_bin = idx % self.n_position
        idx //= self.n_position
        trend_bin = idx % self.n_trend
        idx //= self.n_trend
        vol_bin = idx % self.n_volatility
        idx //= self.n_volatility
        momentum_bin = idx
        
        return {
            "momentum": momentum_bin,
            "volatility": vol_bin,
            "trend": trend_bin,
            "position": pos_bin,
            "emotion": Emotion(emotion_bin).name,
            "pnl": pnl_bin
        }


# ============================================================================
# SCALED Q-TABLE
# ============================================================================

class ScaledQTable:
    """Large-scale Q-table with efficient storage"""
    
    def __init__(self, n_states: int, n_actions: int = 4,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 epsilon: float = 0.1):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
        # Sparse storage - only store visited states
        self.q: Dict[int, np.ndarray] = {}
        self.visit_counts: Dict[int, np.ndarray] = {}
        
        # Default Q-values for unvisited states
        self.default_q = np.zeros(n_actions)
        
        # Statistics
        self.total_updates = 0
        self.unique_states = 0
    
    def get_q(self, state: int) -> np.ndarray:
        """Get Q-values for state (lazy initialization)"""
        if state not in self.q:
            self.q[state] = self.default_q.copy()
            self.visit_counts[state] = np.zeros(self.n_actions)
            self.unique_states += 1
        return self.q[state]
    
    def select_action(self, state: int, explore: bool = True) -> Tuple[Action, np.ndarray]:
        """Epsilon-greedy action selection with UCB exploration bonus"""
        q_values = self.get_q(state)
        
        if explore and np.random.random() < self.epsilon:
            action = Action(np.random.randint(self.n_actions))
        else:
            # UCB bonus for exploration
            visits = self.visit_counts.get(state, np.ones(self.n_actions))
            ucb_bonus = np.sqrt(2 * np.log(self.total_updates + 1) / (visits + 1))
            adjusted_q = q_values + 0.1 * ucb_bonus
            action = Action(np.argmax(adjusted_q))
        
        return action, q_values
    
    def update(self, state: int, action: Action, reward: float, 
               next_state: int, done: bool = False):
        """Q-learning update with visit counting"""
        q_values = self.get_q(state)
        next_q = self.get_q(next_state)
        
        a = action.value
        target = reward if done else reward + self.gamma * np.max(next_q)
        
        # Adaptive learning rate based on visit count
        visits = self.visit_counts[state][a]
        adaptive_lr = self.lr / (1 + visits * 0.01)
        
        q_values[a] += adaptive_lr * (target - q_values[a])
        self.visit_counts[state][a] += 1
        self.total_updates += 1
    
    def get_stats(self) -> Dict:
        """Get Q-table statistics"""
        return {
            "total_states": self.n_states,
            "visited_states": self.unique_states,
            "coverage": self.unique_states / self.n_states * 100,
            "total_updates": self.total_updates,
            "avg_updates_per_state": self.total_updates / max(1, self.unique_states)
        }


# ============================================================================
# EMOTIONAL Q-AGENT
# ============================================================================

class EmotionalQAgent:
    """Q-learning agent with emotional intelligence and emoji communication"""
    
    def __init__(self, scale: str = "medium"):
        """
        Initialize agent with configurable scale
        scale: "small" (~1K states), "medium" (~10K), "large" (~100K), "xlarge" (~1M)
        """
        scales = {
            "small": (5, 3, 5, 4, 5, 3),    # ~1,350 states
            "medium": (7, 5, 7, 4, 9, 5),   # ~44,100 states
            "large": (11, 7, 11, 4, 9, 7),  # ~266,420 states
            "xlarge": (15, 9, 15, 4, 9, 9)  # ~1,312,200 states
        }
        
        params = scales.get(scale, scales["medium"])
        self.encoder = StateEncoder(*params)
        
        # Multiple Q-tables for different strategies
        self.q_tables = {
            "main": ScaledQTable(self.encoder.n_states, epsilon=0.15),
            "conservative": ScaledQTable(self.encoder.n_states, epsilon=0.05),
            "aggressive": ScaledQTable(self.encoder.n_states, epsilon=0.25),
        }
        
        # Emotional state
        self.emotions = EmotionalState()
        
        # Market features
        self.features = MarketFeatures()
        
        # Strategy weights (emotional state affects these)
        self.strategy_weights = {"main": 0.5, "conservative": 0.3, "aggressive": 0.2}
        
        # Communication history
        self.messages: List[str] = []
        
        # Performance tracking
        self.total_reward = 0.0
        self.trades = 0
        self.wins = 0
        self.episode_rewards: List[float] = []
    
    def update_strategy_weights(self):
        """Adjust strategy weights based on emotional state"""
        fear = self.emotions.fear
        greed = self.emotions.greed
        confidence = self.emotions.confidence
        
        # More fear -> more conservative
        # More greed -> more aggressive
        # More confidence -> more main
        
        self.strategy_weights["conservative"] = 0.2 + fear * 0.4
        self.strategy_weights["aggressive"] = 0.1 + greed * 0.3
        self.strategy_weights["main"] = 1.0 - self.strategy_weights["conservative"] - self.strategy_weights["aggressive"]
    
    def select_action(self, state: int) -> Tuple[Action, str]:
        """Select action using weighted ensemble and generate emoji message"""
        # Get actions from all strategies
        actions = {}
        q_values = {}
        
        for name, q_table in self.q_tables.items():
            action, qv = q_table.select_action(state)
            actions[name] = action
            q_values[name] = qv
        
        # Weighted vote
        action_scores = np.zeros(4)
        for name, action in actions.items():
            weight = self.strategy_weights[name]
            action_scores[action.value] += weight
        
        final_action = Action(np.argmax(action_scores))
        
        # Generate emoji message
        message = self._generate_message(final_action, q_values)
        self.messages.append(message)
        
        return final_action, message
    
    def _generate_message(self, action: Action, q_values: Dict) -> str:
        """Generate emoji communication message"""
        parts = []
        
        # Emotion
        parts.append(self.emotions.get_emoji())
        
        # Market assessment
        if self.features.momentum_short > 0.3:
            parts.append(EmojiVocab.COMM_BULLISH)
        elif self.features.momentum_short < -0.3:
            parts.append(EmojiVocab.COMM_BEARISH)
        else:
            parts.append(EmojiVocab.COMM_THINKING)
        
        # Action
        action_emoji = {
            Action.HOLD: EmojiVocab.ACTION_HOLD,
            Action.BUY: EmojiVocab.ACTION_BUY,
            Action.SELL: EmojiVocab.ACTION_SELL,
            Action.WAIT: EmojiVocab.ACTION_WAIT
        }
        parts.append(action_emoji[action])
        
        # Confidence indicator
        conf = self.emotions.confidence
        if conf > 0.8:
            parts.append(EmojiVocab.INTENSITY_MAX)
        elif conf > 0.6:
            parts.append(EmojiVocab.INTENSITY_HIGH)
        elif conf > 0.4:
            parts.append(EmojiVocab.INTENSITY_MED)
        else:
            parts.append(EmojiVocab.INTENSITY_LOW)
        
        # Special states
        if self.emotions.streak > 3:
            parts.append(EmojiVocab.COMM_FIRE)
        if self.emotions.streak < -3:
            parts.append(EmojiVocab.COMM_SKULL)
        if self.features.position and self.features.unrealized_pnl > 0.1:
            parts.append(EmojiVocab.COMM_MONEY)
        if self.emotions.fear > 0.7:
            parts.append(EmojiVocab.COMM_ALERT)
        if self.emotions.greed > 0.7:
            parts.append(EmojiVocab.COMM_DIAMOND)
        
        return " ".join(parts)
    
    def step(self, price_up: bool, price_down: bool) -> Tuple[Action, float, str]:
        """Process one step"""
        # Update market features
        change = 0.1 if price_up else -0.1 if price_down else 0
        self.features.price_changes.append(change)
        if len(self.features.price_changes) > 50:
            self.features.price_changes.pop(0)
        self.features.compute_derived()
        
        # Update emotions from market
        self.emotions.update_from_market(self.features.volatility, self.features.trend_strength)
        
        # Update strategy weights
        self.update_strategy_weights()
        
        # Get state
        state = self.encoder.encode(self.features, self.emotions)
        
        # Select action
        action, message = self.select_action(state)
        
        # Validate action
        if action == Action.BUY and self.features.position:
            action = Action.HOLD
        elif action == Action.SELL and not self.features.position:
            action = Action.HOLD
        
        # Execute and compute reward
        reward = 0.01
        if action == Action.BUY:
            self.features.position = True
            self.features.entry_price = 1.0  # Normalized
            self.features.hold_duration = 0
            reward = -0.05  # Small cost to enter
        elif action == Action.SELL:
            # Compute PnL
            pnl = self.features.unrealized_pnl
            reward = 1.0 + pnl * 2  # Base + PnL bonus
            self.features.position = False
            self.features.total_trades += 1
            self.trades += 1
            
            if pnl > 0:
                self.wins += 1
                self.emotions.update_after_trade(pnl)
            else:
                self.emotions.update_after_trade(pnl)
        else:
            # Holding cost/benefit
            if self.features.position:
                self.features.hold_duration += 1
                self.features.unrealized_pnl += change
                # Patience reward
                reward = 0.01 * self.emotions.patience
        
        # Update Q-tables
        next_state = self.encoder.encode(self.features, self.emotions)
        for q_table in self.q_tables.values():
            q_table.update(state, action, reward, next_state)
        
        self.total_reward += reward
        
        return action, reward, message
    
    def run_episode(self, market_data: List[Tuple[bool, bool]], verbose: bool = False) -> float:
        """Run one episode"""
        episode_reward = 0.0
        
        for step, (price_up, price_down) in enumerate(market_data):
            action, reward, message = self.step(price_up, price_down)
            episode_reward += reward
            
            if verbose:
                signal = "📈" if price_up else "📉"
                print(f"Step {step:3d}: {signal} {message} R:{reward:+.2f}")
        
        self.episode_rewards.append(episode_reward)
        return episode_reward
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        q_stats = {name: q.get_stats() for name, q in self.q_tables.items()}
        
        return {
            "total_reward": self.total_reward,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.trades),
            "episodes": len(self.episode_rewards),
            "avg_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "emotion": self.emotions.get_emotion().name,
            "fear": self.emotions.fear,
            "greed": self.emotions.greed,
            "confidence": self.emotions.confidence,
            "q_tables": q_stats
        }
    
    def print_status(self):
        """Print current status with emojis"""
        print("\n" + "=" * 60)
        print(f"🤖 Agent Status {self.emotions.get_emoji()}")
        print("=" * 60)
        
        stats = self.get_stats()
        
        print(f"💰 Total Reward: {stats['total_reward']:.2f}")
        print(f"📊 Trades: {stats['trades']} (Win Rate: {stats['win_rate']*100:.1f}%)")
        print(f"🧠 Emotion: {stats['emotion']}")
        print(f"   Fear: {'█' * int(stats['fear']*10)}{'·' * (10-int(stats['fear']*10))} {stats['fear']:.2f}")
        print(f"   Greed: {'█' * int(stats['greed']*10)}{'·' * (10-int(stats['greed']*10))} {stats['greed']:.2f}")
        print(f"   Confidence: {'█' * int(stats['confidence']*10)}{'·' * (10-int(stats['confidence']*10))} {stats['confidence']:.2f}")
        
        print(f"\n📈 Q-Table Coverage:")
        for name, qs in stats['q_tables'].items():
            print(f"   {name}: {qs['visited_states']:,}/{qs['total_states']:,} ({qs['coverage']:.2f}%)")


# ============================================================================
# TESTING
# ============================================================================

def generate_market(n_steps: int, regime: str = "mixed") -> List[Tuple[bool, bool]]:
    """Generate market data with different regimes"""
    data = []
    
    if regime == "bull":
        prob_up = 0.7
    elif regime == "bear":
        prob_up = 0.3
    elif regime == "volatile":
        prob_up = 0.5 + 0.3 * np.sin(np.arange(n_steps) * 0.5)
    else:  # mixed
        prob_up = 0.5
    
    for i in range(n_steps):
        p = prob_up if isinstance(prob_up, float) else prob_up[i]
        up = np.random.random() < p
        data.append((up, not up))
    
    return data


def test_emotional_agent():
    """Test the emotional agent"""
    print("=" * 70)
    print("🧠 Emotional Q-Agent Test")
    print("=" * 70)
    
    # Test different scales
    for scale in ["small", "medium"]:
        print(f"\n--- Scale: {scale} ---")
        agent = EmotionalQAgent(scale=scale)
        
        # Training phase
        print("Training...")
        for ep in range(20):
            market = generate_market(100, "mixed")
            reward = agent.run_episode(market, verbose=False)
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}: reward={reward:.2f}, "
                      f"emotion={agent.emotions.get_emoji()}")
        
        agent.print_status()
    
    # Detailed run with largest scale
    print("\n" + "=" * 70)
    print("🚀 Detailed Run (Large Scale)")
    print("=" * 70)
    
    agent = EmotionalQAgent(scale="large")
    
    # Train
    print("\nTraining on 50 episodes...")
    for ep in range(50):
        market = generate_market(200, "mixed")
        agent.run_episode(market, verbose=False)
        if (ep + 1) % 10 == 0:
            stats = agent.get_stats()
            print(f"  Ep {ep+1}: reward={stats['avg_episode_reward']:.2f}, "
                  f"coverage={stats['q_tables']['main']['coverage']:.2f}%, "
                  f"emotion={agent.emotions.get_emoji()}")
    
    # Test with different regimes
    print("\n--- Testing Different Market Regimes ---")
    for regime in ["bull", "bear", "volatile"]:
        market = generate_market(100, regime)
        reward = agent.run_episode(market, verbose=False)
        print(f"  {regime:8}: reward={reward:.2f} {agent.emotions.get_emoji()}")
    
    # Show detailed episode
    print("\n--- Detailed Episode ---")
    market = generate_market(30, "mixed")
    agent.run_episode(market, verbose=True)
    
    agent.print_status()
    
    # Show recent messages
    print("\n--- Recent Messages ---")
    for msg in agent.messages[-10:]:
        print(f"  {msg}")


if __name__ == "__main__":
    test_emotional_agent()

