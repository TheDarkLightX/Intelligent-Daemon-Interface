#!/usr/bin/env python3
"""
Advanced Q-Agent v3
- Ensemble Q-learning with weighted voting
- Automatic market regime detection
- Dynamic position sizing
- Performance attribution
- Rich emoji communication protocol
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import IntEnum
from collections import deque
import time


# ============================================================================
# EMOJI PROTOCOL v2
# ============================================================================

class Mood(IntEnum):
    EUPHORIC = 0
    EXCITED = 1
    CONFIDENT = 2
    CALM = 3
    CAUTIOUS = 4
    NERVOUS = 5
    FEARFUL = 6
    PANICKED = 7


class MarketRegime(IntEnum):
    STRONG_BULL = 0
    BULL = 1
    NEUTRAL = 2
    BEAR = 3
    STRONG_BEAR = 4
    HIGH_VOL = 5
    LOW_VOL = 6


class E:
    """Comprehensive emoji vocabulary"""
    # Moods
    MOODS = {
        Mood.EUPHORIC: "🚀",
        Mood.EXCITED: "🤩",
        Mood.CONFIDENT: "😎",
        Mood.CALM: "😊",
        Mood.CAUTIOUS: "🤔",
        Mood.NERVOUS: "😟",
        Mood.FEARFUL: "😰",
        Mood.PANICKED: "😱",
    }
    
    # Market regimes
    REGIMES = {
        MarketRegime.STRONG_BULL: "🐂🔥",
        MarketRegime.BULL: "🐂",
        MarketRegime.NEUTRAL: "➡️",
        MarketRegime.BEAR: "🐻",
        MarketRegime.STRONG_BEAR: "🐻💀",
        MarketRegime.HIGH_VOL: "🎢",
        MarketRegime.LOW_VOL: "😴",
    }
    
    # Actions
    BUY = "🟢"
    SELL = "🔴"
    HOLD = "⏸️"
    SCALE_IN = "🟢+"
    SCALE_OUT = "🔴-"
    
    # Status
    PROFIT = "💰"
    LOSS = "📉"
    BREAKEVEN = "➖"
    FIRE = "🔥"
    SKULL = "💀"
    DIAMOND = "💎"
    ROCKET = "🚀"
    ALERT = "🚨"
    STAR = "⭐"
    CHECK = "✅"
    CROSS = "❌"
    BRAIN = "🧠"
    CHART = "📊"
    TARGET = "🎯"
    TROPHY = "🏆"
    
    # Confidence bars
    @staticmethod
    def confidence_bar(conf: float, width: int = 5) -> str:
        filled = int(conf * width)
        return "█" * filled + "░" * (width - filled)


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """Detects market regime from price history"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.prices: deque = deque(maxlen=lookback)
        self.returns: deque = deque(maxlen=lookback)
        
    def update(self, price: float):
        """Update with new price"""
        if self.prices:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
        self.prices.append(price)
    
    def detect(self) -> Tuple[MarketRegime, float]:
        """Detect current regime with confidence"""
        if len(self.returns) < 10:
            return MarketRegime.NEUTRAL, 0.5
        
        returns = np.array(self.returns)
        
        # Trend metrics
        mean_return = np.mean(returns)
        momentum_short = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        momentum_long = np.mean(returns[-20:]) if len(returns) >= 20 else mean_return
        
        # Volatility
        volatility = np.std(returns)
        vol_percentile = np.sum(np.abs(returns) < volatility * 2) / len(returns)
        
        # Determine regime
        confidence = 0.5
        
        if volatility > 0.03:
            regime = MarketRegime.HIGH_VOL
            confidence = min(0.9, volatility / 0.05)
        elif volatility < 0.01:
            regime = MarketRegime.LOW_VOL
            confidence = min(0.9, 0.02 / (volatility + 0.001))
        elif momentum_short > 0.02 and momentum_long > 0.01:
            regime = MarketRegime.STRONG_BULL
            confidence = min(0.9, momentum_short / 0.03)
        elif momentum_short > 0.005:
            regime = MarketRegime.BULL
            confidence = 0.6 + momentum_short * 10
        elif momentum_short < -0.02 and momentum_long < -0.01:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(0.9, abs(momentum_short) / 0.03)
        elif momentum_short < -0.005:
            regime = MarketRegime.BEAR
            confidence = 0.6 + abs(momentum_short) * 10
        else:
            regime = MarketRegime.NEUTRAL
            confidence = 0.5 + (1 - abs(momentum_short) * 20)
        
        return regime, min(0.95, confidence)
    
    def get_features(self) -> Dict[str, float]:
        """Get regime features for state encoding"""
        if len(self.returns) < 5:
            return {
                "momentum_short": 0.0,
                "momentum_long": 0.0,
                "volatility": 0.02,
                "trend_strength": 0.0
            }
        
        returns = np.array(self.returns)
        
        return {
            "momentum_short": np.mean(returns[-5:]),
            "momentum_long": np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns),
            "volatility": np.std(returns),
            "trend_strength": np.mean(returns[-10:]) / (np.std(returns[-10:]) + 0.001) if len(returns) >= 10 else 0
        }


# ============================================================================
# ENSEMBLE Q-TABLE
# ============================================================================

class EnsembleQTable:
    """
    Ensemble of Q-tables with different configurations
    - Different learning rates
    - Different discount factors
    - Different exploration strategies
    Combines predictions through weighted voting
    """
    
    def __init__(self, n_states: int, n_actions: int = 4, n_members: int = 5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_members = n_members
        
        # Create ensemble members with different hyperparameters
        self.members = []
        lr_range = [0.05, 0.1, 0.15, 0.2, 0.25]
        gamma_range = [0.9, 0.92, 0.95, 0.97, 0.99]
        
        for i in range(n_members):
            member = {
                "q": np.zeros((n_states, n_actions)),
                "visits": np.zeros((n_states, n_actions)),
                "lr": lr_range[i % len(lr_range)],
                "gamma": gamma_range[i % len(gamma_range)],
                "performance": 1.0,  # Track member performance
                "recent_rewards": deque(maxlen=100)
            }
            self.members.append(member)
        
        self.epsilon = 0.15
        self.total_updates = 0
    
    def get_weights(self) -> np.ndarray:
        """Get voting weights based on recent performance"""
        performances = np.array([m["performance"] for m in self.members])
        # Softmax with temperature
        temp = 0.5
        exp_perf = np.exp(performances / temp)
        return exp_perf / exp_perf.sum()
    
    def select_action(self, state: int, explore: bool = True) -> Tuple[int, np.ndarray, Dict]:
        """Select action using weighted ensemble voting"""
        weights = self.get_weights()
        
        # Get Q-values from each member
        q_values_all = np.array([m["q"][state] for m in self.members])
        
        # Weighted average
        combined_q = np.average(q_values_all, axis=0, weights=weights)
        
        # Exploration
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Add UCB bonus
            visits = sum(m["visits"][state] for m in self.members)
            ucb = np.sqrt(2 * np.log(self.total_updates + 1) / (visits + 1))
            action = int(np.argmax(combined_q + 0.2 * ucb))
        
        # Member agreement (for confidence)
        member_actions = [int(np.argmax(q)) for q in q_values_all]
        agreement = member_actions.count(action) / len(member_actions)
        
        info = {
            "weights": weights,
            "agreement": agreement,
            "q_values_all": q_values_all,
            "member_actions": member_actions
        }
        
        return action, combined_q, info
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update all ensemble members"""
        for member in self.members:
            # Standard Q-learning update
            next_q = np.max(member["q"][next_state])
            target = reward + member["gamma"] * next_q
            
            member["q"][state, action] += member["lr"] * (target - member["q"][state, action])
            member["visits"][state, action] += 1
            
            # Track performance
            member["recent_rewards"].append(reward)
            if len(member["recent_rewards"]) >= 10:
                member["performance"] = np.mean(list(member["recent_rewards"])[-50:])
        
        self.total_updates += 1
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics"""
        return {
            "n_members": self.n_members,
            "weights": self.get_weights().tolist(),
            "performances": [m["performance"] for m in self.members],
            "total_updates": self.total_updates,
            "visited_states": sum(np.sum(m["visits"] > 0) for m in self.members) / self.n_members
        }


# ============================================================================
# POSITION SIZER
# ============================================================================

class PositionSizer:
    """Dynamic position sizing based on multiple factors"""
    
    def __init__(self, base_size: float = 1.0, max_size: float = 3.0):
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = 0.25
        
        # Kelly criterion factors
        self.win_rate = 0.5
        self.avg_win = 0.02
        self.avg_loss = 0.02
        
        # Track history
        self.trade_results: List[float] = []
    
    def update_stats(self, pnl: float):
        """Update win rate and average win/loss"""
        self.trade_results.append(pnl)
        
        if len(self.trade_results) >= 10:
            recent = self.trade_results[-100:]
            wins = [r for r in recent if r > 0]
            losses = [r for r in recent if r < 0]
            
            self.win_rate = len(wins) / len(recent) if recent else 0.5
            self.avg_win = np.mean(wins) if wins else 0.02
            self.avg_loss = abs(np.mean(losses)) if losses else 0.02
    
    def kelly_fraction(self) -> float:
        """Calculate Kelly criterion fraction"""
        if self.avg_loss == 0:
            return 0.5
        
        # f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
        b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1
        p = self.win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b if b > 0 else 0
        
        # Half-Kelly for safety
        return max(0, min(1, kelly * 0.5))
    
    def calculate_size(self, confidence: float, regime_confidence: float,
                      ensemble_agreement: float, volatility: float) -> float:
        """Calculate position size"""
        # Base Kelly
        kelly = self.kelly_fraction()
        
        # Confidence adjustment
        conf_factor = 0.5 + confidence * 0.5
        
        # Regime adjustment
        regime_factor = 0.7 + regime_confidence * 0.3
        
        # Ensemble agreement adjustment
        agreement_factor = 0.6 + ensemble_agreement * 0.4
        
        # Volatility adjustment (reduce size in high vol)
        vol_factor = max(0.5, 1.5 - volatility * 20)
        
        # Combine
        size = self.base_size * kelly * conf_factor * regime_factor * agreement_factor * vol_factor
        
        return max(self.min_size, min(self.max_size, size))


# ============================================================================
# PERFORMANCE ATTRIBUTION
# ============================================================================

class PerformanceAttributor:
    """Analyze what contributes to performance"""
    
    def __init__(self):
        self.trades: List[Dict] = []
        
    def record_trade(self, entry_regime: MarketRegime, exit_regime: MarketRegime,
                    entry_confidence: float, pnl: float, hold_duration: int,
                    position_size: float, ensemble_agreement: float):
        """Record trade for attribution"""
        self.trades.append({
            "entry_regime": entry_regime,
            "exit_regime": exit_regime,
            "entry_confidence": entry_confidence,
            "pnl": pnl,
            "hold_duration": hold_duration,
            "position_size": position_size,
            "agreement": ensemble_agreement
        })
    
    def analyze(self) -> Dict:
        """Analyze performance attribution"""
        if len(self.trades) < 10:
            return {"status": "insufficient_data"}
        
        trades = self.trades
        
        # By regime
        regime_perf = {}
        for regime in MarketRegime:
            regime_trades = [t for t in trades if t["entry_regime"] == regime]
            if regime_trades:
                regime_perf[regime.name] = {
                    "count": len(regime_trades),
                    "avg_pnl": np.mean([t["pnl"] for t in regime_trades]),
                    "win_rate": np.mean([1 if t["pnl"] > 0 else 0 for t in regime_trades])
                }
        
        # By confidence
        high_conf = [t for t in trades if t["entry_confidence"] > 0.7]
        low_conf = [t for t in trades if t["entry_confidence"] < 0.4]
        
        conf_analysis = {
            "high_confidence": {
                "count": len(high_conf),
                "avg_pnl": np.mean([t["pnl"] for t in high_conf]) if high_conf else 0,
                "win_rate": np.mean([1 if t["pnl"] > 0 else 0 for t in high_conf]) if high_conf else 0
            },
            "low_confidence": {
                "count": len(low_conf),
                "avg_pnl": np.mean([t["pnl"] for t in low_conf]) if low_conf else 0,
                "win_rate": np.mean([1 if t["pnl"] > 0 else 0 for t in low_conf]) if low_conf else 0
            }
        }
        
        # By ensemble agreement
        high_agree = [t for t in trades if t["agreement"] > 0.8]
        low_agree = [t for t in trades if t["agreement"] < 0.5]
        
        agreement_analysis = {
            "high_agreement": {
                "count": len(high_agree),
                "avg_pnl": np.mean([t["pnl"] for t in high_agree]) if high_agree else 0,
                "win_rate": np.mean([1 if t["pnl"] > 0 else 0 for t in high_agree]) if high_agree else 0
            },
            "low_agreement": {
                "count": len(low_agree),
                "avg_pnl": np.mean([t["pnl"] for t in low_agree]) if low_agree else 0,
                "win_rate": np.mean([1 if t["pnl"] > 0 else 0 for t in low_agree]) if low_agree else 0
            }
        }
        
        return {
            "total_trades": len(trades),
            "by_regime": regime_perf,
            "by_confidence": conf_analysis,
            "by_agreement": agreement_analysis
        }
    
    def print_report(self):
        """Print attribution report"""
        analysis = self.analyze()
        
        if analysis.get("status") == "insufficient_data":
            print("Insufficient data for attribution analysis")
            return
        
        print(f"\n{E.CHART} Performance Attribution Report")
        print("=" * 60)
        
        print(f"\n{E.TARGET} By Market Regime:")
        for regime, stats in analysis["by_regime"].items():
            pnl_emoji = E.PROFIT if stats["avg_pnl"] > 0 else E.LOSS
            print(f"  {regime:15} | {stats['count']:4} trades | "
                  f"{pnl_emoji} {stats['avg_pnl']*100:+6.2f}% | "
                  f"Win: {stats['win_rate']*100:5.1f}%")
        
        print(f"\n{E.BRAIN} By Confidence:")
        for level, stats in analysis["by_confidence"].items():
            if stats["count"] > 0:
                pnl_emoji = E.PROFIT if stats["avg_pnl"] > 0 else E.LOSS
                print(f"  {level:15} | {stats['count']:4} trades | "
                      f"{pnl_emoji} {stats['avg_pnl']*100:+6.2f}% | "
                      f"Win: {stats['win_rate']*100:5.1f}%")
        
        print(f"\n{E.STAR} By Ensemble Agreement:")
        for level, stats in analysis["by_agreement"].items():
            if stats["count"] > 0:
                pnl_emoji = E.PROFIT if stats["avg_pnl"] > 0 else E.LOSS
                print(f"  {level:15} | {stats['count']:4} trades | "
                      f"{pnl_emoji} {stats['avg_pnl']*100:+6.2f}% | "
                      f"Win: {stats['win_rate']*100:5.1f}%")


# ============================================================================
# ADVANCED Q-AGENT
# ============================================================================

class AdvancedQAgent:
    """Full advanced Q-agent with all features"""
    
    def __init__(self, n_states: int = 500):
        print(f"{E.BRAIN} Initializing Advanced Q-Agent v3")
        
        # Components
        self.regime_detector = RegimeDetector()
        self.ensemble = EnsembleQTable(n_states, n_actions=4, n_members=5)
        self.position_sizer = PositionSizer()
        self.attributor = PerformanceAttributor()
        
        # State
        self.position = False
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_regime = MarketRegime.NEUTRAL
        self.entry_confidence = 0.5
        self.entry_agreement = 0.5
        self.hold_duration = 0
        self.streak = 0
        
        # Stats
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.episode_pnls: List[float] = []
        self.messages: List[str] = []
        
        # Mood system
        self.mood = Mood.CALM
        self.confidence = 0.5
        
        print(f"  {E.CHECK} Ensemble: {self.ensemble.n_members} members")
        print(f"  {E.CHECK} State space: {n_states} states")
    
    def _encode_state(self, features: Dict, regime: MarketRegime) -> int:
        """Encode features to state index"""
        # Momentum bins (0-9)
        m_short = min(9, max(0, int((features["momentum_short"] + 0.05) / 0.01)))
        
        # Volatility bins (0-4)
        vol = min(4, int(features["volatility"] / 0.01))
        
        # Regime bins (0-6)
        regime_bin = regime.value
        
        # Position bins (0-2)
        if not self.position:
            pos_bin = 0
        elif self.entry_price > 0:
            pnl = (self.regime_detector.prices[-1] - self.entry_price) / self.entry_price if self.regime_detector.prices else 0
            pos_bin = 1 if pnl > 0 else 2
        else:
            pos_bin = 1
        
        # Combine (10 * 5 * 7 * 3 = 1050, clipped to n_states)
        idx = m_short * 105 + vol * 21 + regime_bin * 3 + pos_bin
        return min(idx, self.ensemble.n_states - 1)
    
    def _update_mood(self):
        """Update agent mood based on performance"""
        if len(self.episode_pnls) < 5:
            self.mood = Mood.CALM
            return
        
        recent = self.episode_pnls[-10:]
        avg_pnl = np.mean(recent)
        trend = recent[-1] - recent[0] if len(recent) > 1 else 0
        
        win_rate = self.wins / max(1, self.trades)
        
        # Determine mood
        if win_rate > 0.7 and avg_pnl > 0.01:
            self.mood = Mood.EUPHORIC
        elif win_rate > 0.6 and avg_pnl > 0:
            self.mood = Mood.EXCITED
        elif win_rate > 0.5 and avg_pnl > 0:
            self.mood = Mood.CONFIDENT
        elif win_rate > 0.45:
            self.mood = Mood.CALM
        elif win_rate > 0.4:
            self.mood = Mood.CAUTIOUS
        elif win_rate > 0.35:
            self.mood = Mood.NERVOUS
        elif win_rate > 0.3:
            self.mood = Mood.FEARFUL
        else:
            self.mood = Mood.PANICKED
    
    def _generate_message(self, action: int, reward: float, regime: MarketRegime,
                         agreement: float, size: float) -> str:
        """Generate rich emoji message"""
        parts = []
        
        # Mood
        parts.append(E.MOODS[self.mood])
        
        # Regime
        parts.append(E.REGIMES[regime])
        
        # Action with size indicator
        if action == 1:
            if size > 1.5:
                parts.append(E.SCALE_IN)
            else:
                parts.append(E.BUY)
        elif action == 2:
            parts.append(E.SELL)
        else:
            parts.append(E.HOLD)
        
        # Confidence bar
        parts.append(f"[{E.confidence_bar(self.confidence)}]")
        
        # Agreement indicator
        if agreement > 0.8:
            parts.append(E.CHECK)
        elif agreement < 0.4:
            parts.append(E.ALERT)
        
        # Special indicators
        if self.streak > 3:
            parts.append(E.FIRE)
        elif self.streak < -3:
            parts.append(E.SKULL)
        
        if self.position:
            price = self.regime_detector.prices[-1] if self.regime_detector.prices else 0
            if price > 0 and self.entry_price > 0:
                unrealized = (price - self.entry_price) / self.entry_price
                if unrealized > 0.03:
                    parts.append(E.PROFIT)
                elif unrealized < -0.03:
                    parts.append(E.LOSS)
        
        return " ".join(parts)
    
    def step(self, price: float) -> Tuple[int, float, str]:
        """Execute one step"""
        # Update regime detector
        self.regime_detector.update(price)
        
        # Get regime and features
        regime, regime_conf = self.regime_detector.detect()
        features = self.regime_detector.get_features()
        
        # Encode state
        state = self._encode_state(features, regime)
        
        # Get action from ensemble
        action, q_values, info = self.ensemble.select_action(state)
        agreement = info["agreement"]
        
        # Update confidence
        self.confidence = 0.3 + agreement * 0.4 + regime_conf * 0.3
        
        # Calculate position size
        size = self.position_sizer.calculate_size(
            self.confidence, regime_conf, agreement, features["volatility"]
        )
        
        # Validate action
        executed = True
        if action == 1 and self.position:
            action = 0
            executed = False
        elif action == 2 and not self.position:
            action = 0
            executed = False
        
        # Execute
        reward = 0.0
        
        if action == 1:  # BUY
            self.position = True
            self.position_size = size
            self.entry_price = price
            self.entry_regime = regime
            self.entry_confidence = self.confidence
            self.entry_agreement = agreement
            self.hold_duration = 0
            reward = -0.005 * size  # Scaled transaction cost
            
        elif action == 2:  # SELL
            if self.entry_price > 0:
                pnl = (price - self.entry_price) / self.entry_price * self.position_size
                reward = 0.3 + pnl * 15  # Scaled by position size effect
                
                # Record for attribution
                self.attributor.record_trade(
                    self.entry_regime, regime,
                    self.entry_confidence, pnl,
                    self.hold_duration, self.position_size,
                    self.entry_agreement
                )
                
                # Update stats
                self.position_sizer.update_stats(pnl)
                self.total_pnl += pnl
                self.trades += 1
                
                if pnl > 0:
                    self.wins += 1
                    self.streak = max(1, self.streak + 1)
                else:
                    self.streak = min(-1, self.streak - 1)
            
            self.position = False
            self.position_size = 0.0
            
        else:  # HOLD
            if self.position:
                self.hold_duration += 1
                # Holding reward based on unrealized PnL
                if self.entry_price > 0:
                    unrealized = (price - self.entry_price) / self.entry_price
                    reward = 0.002 * np.sign(unrealized) * (1 + abs(unrealized) * 5)
            else:
                reward = -0.001  # Opportunity cost
        
        # Update ensemble
        next_state = self._encode_state(
            self.regime_detector.get_features(),
            self.regime_detector.detect()[0]
        )
        self.ensemble.update(state, action, reward, next_state)
        
        # Update mood
        self._update_mood()
        
        # Generate message
        msg = self._generate_message(action, reward, regime, agreement, size)
        self.messages.append(msg)
        
        return action, reward, msg
    
    def run_episode(self, prices: List[float], verbose: bool = False) -> float:
        """Run one episode"""
        ep_reward = 0.0
        
        for i, price in enumerate(prices):
            action, reward, msg = self.step(price)
            ep_reward += reward
            
            if verbose and i % 50 == 0:
                print(f"  Step {i:3d}: {msg} R:{reward:+.3f}")
        
        self.episode_pnls.append(ep_reward)
        return ep_reward
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        ensemble_stats = self.ensemble.get_stats()
        
        return {
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.trades),
            "avg_episode_pnl": np.mean(self.episode_pnls[-20:]) if self.episode_pnls else 0,
            "mood": self.mood.name,
            "confidence": self.confidence,
            "streak": self.streak,
            "ensemble": ensemble_stats,
            "kelly_fraction": self.position_sizer.kelly_fraction()
        }
    
    def print_status(self):
        """Print comprehensive status"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"{E.BRAIN} Advanced Q-Agent v3 Status {E.MOODS[self.mood]}")
        print(f"{'='*70}")
        
        print(f"\n{E.TARGET} Performance:")
        pnl_emoji = E.PROFIT if stats["total_pnl"] > 0 else E.LOSS
        print(f"  Total PnL: {pnl_emoji} {stats['total_pnl']*100:+.2f}%")
        print(f"  Trades: {stats['trades']} (Win Rate: {stats['win_rate']*100:.1f}%)")
        print(f"  Avg Episode: {stats['avg_episode_pnl']:.3f}")
        print(f"  Streak: {stats['streak']}")
        
        print(f"\n{E.STAR} Ensemble:")
        print(f"  Members: {stats['ensemble']['n_members']}")
        print(f"  Weights: {[f'{w:.2f}' for w in stats['ensemble']['weights']]}")
        
        print(f"\n{E.CHART} Position Sizing:")
        print(f"  Kelly Fraction: {stats['kelly_fraction']:.2f}")
        print(f"  Confidence: [{E.confidence_bar(stats['confidence'])}] {stats['confidence']:.2f}")
        
        # Attribution report
        self.attributor.print_report()


# ============================================================================
# TRAINING
# ============================================================================

def generate_prices(n: int, regime: str = "mixed", start: float = 100.0) -> List[float]:
    """Generate price series"""
    prices = [start]
    
    for i in range(n - 1):
        if regime == "bull":
            drift = 0.0003
            vol = 0.01
        elif regime == "bear":
            drift = -0.0003
            vol = 0.01
        elif regime == "volatile":
            drift = 0.0001 * np.sin(i * 0.1)
            vol = 0.02
        else:  # mixed
            drift = 0.0001 * np.sin(i * 0.05)
            vol = 0.012
        
        change = drift + np.random.randn() * vol
        prices.append(prices[-1] * (1 + change))
    
    return prices


def run_advanced_training():
    """Run advanced training pipeline"""
    print(f"{'='*70}")
    print(f"{E.ROCKET} ADVANCED Q-AGENT v3 TRAINING {E.ROCKET}")
    print(f"{'='*70}")
    
    agent = AdvancedQAgent(n_states=500)
    
    # Training phases
    phases = [
        ("Exploration", 30, 0.25),
        ("Learning", 50, 0.15),
        ("Optimization", 80, 0.08),
        ("Mastery", 100, 0.03),
    ]
    
    for phase_name, episodes, epsilon in phases:
        print(f"\n{E.STAR} Phase: {phase_name}")
        print("-" * 50)
        
        agent.ensemble.epsilon = epsilon
        phase_rewards = []
        
        regimes = ["mixed", "bull", "bear", "volatile"]
        
        for ep in range(episodes):
            regime = regimes[ep % len(regimes)]
            prices = generate_prices(300, regime)
            reward = agent.run_episode(prices, verbose=False)
            phase_rewards.append(reward)
            
            if (ep + 1) % max(1, episodes // 4) == 0:
                stats = agent.get_stats()
                print(f"  Ep {ep+1:3d}: {E.MOODS[agent.mood]} "
                      f"R={np.mean(phase_rewards[-10:]):7.3f} "
                      f"Win={stats['win_rate']*100:5.1f}% "
                      f"Kelly={stats['kelly_fraction']:.2f}")
        
        print(f"  {E.CHECK} Phase avg: {np.mean(phase_rewards):.3f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{E.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    agent.ensemble.epsilon = 0.0
    
    for regime in ["bull", "bear", "volatile", "mixed"]:
        prices = generate_prices(500, regime)
        reward = agent.run_episode(prices, verbose=False)
        stats = agent.get_stats()
        regime_emoji = {"bull": "🐂", "bear": "🐻", "volatile": "🎢", "mixed": "➡️"}[regime]
        print(f"  {regime_emoji} {regime:10}: R={reward:8.3f} Win={stats['win_rate']*100:5.1f}%")
    
    agent.print_status()
    
    # Show recent messages
    print(f"\n{E.FIRE} Recent Communications:")
    for msg in agent.messages[-10:]:
        print(f"  {msg}")
    
    return agent


if __name__ == "__main__":
    agent = run_advanced_training()

