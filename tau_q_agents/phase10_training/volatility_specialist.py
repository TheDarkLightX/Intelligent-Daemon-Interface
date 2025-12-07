#!/usr/bin/env python3
"""
Volatility Specialist Agent
- Adaptive strategies for different volatility regimes
- Momentum reversal detection
- Breakout/breakdown trading
- Volatility clustering awareness
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
from enum import IntEnum


# ============================================================================
# EMOJI PROTOCOL
# ============================================================================

class E:
    # Core
    UP = "📈"; DOWN = "📉"; PROFIT = "💰"; LOSS = "📉"
    BUY = "🟢"; SELL = "🔴"; HOLD = "⏸️"
    
    # Volatility specific
    CALM = "😌"; ALERT = "⚡"; STORM = "🌪️"; ROCKET = "🚀"
    BREAKOUT = "💥"; REVERSAL = "🔄"; SQUEEZE = "🤏"; EXPANSION = "💨"
    
    # Status
    FIRE = "🔥"; SKULL = "💀"; STAR = "⭐"; CHECK = "✅"; BRAIN = "🧠"
    TROPHY = "🏆"; TARGET = "🎯"; CHART = "📊"
    
    @staticmethod
    def vol_meter(vol: float) -> str:
        """Volatility meter visualization"""
        if vol < 0.005:
            return "▁▁▁▁"  # Very low
        elif vol < 0.01:
            return "▂▂▁▁"  # Low
        elif vol < 0.015:
            return "▄▄▂▂"  # Normal
        elif vol < 0.02:
            return "▆▆▄▄"  # High
        else:
            return "█████"  # Very high


# ============================================================================
# VOLATILITY ANALYZER
# ============================================================================

class VolatilityAnalyzer:
    """Deep volatility analysis"""
    
    def __init__(self, lookback: int = 100):
        self.prices: deque = deque(maxlen=lookback)
        self.returns: deque = deque(maxlen=lookback)
        self.vol_history: deque = deque(maxlen=50)
        
    def update(self, price: float):
        if self.prices:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
        self.prices.append(price)
        
        # Update volatility estimate
        if len(self.returns) >= 10:
            vol = np.std(list(self.returns)[-20:])
            self.vol_history.append(vol)
    
    def get_vol_features(self) -> Dict:
        """Get comprehensive volatility features"""
        if len(self.returns) < 10:
            return {
                "current_vol": 0.015,
                "vol_trend": 0,
                "vol_percentile": 50,
                "is_clustering": False,
                "expansion_signal": False,
                "contraction_signal": False,
                "breakout_potential": 0.5
            }
        
        returns = np.array(self.returns)
        
        # Current volatility
        current_vol = np.std(returns[-20:])
        
        # Volatility trend
        if len(self.vol_history) >= 10:
            vol_arr = np.array(self.vol_history)
            vol_trend = (vol_arr[-5:].mean() - vol_arr[-10:-5].mean()) / (vol_arr.mean() + 0.001)
        else:
            vol_trend = 0
        
        # Volatility percentile (relative to history)
        if len(self.vol_history) >= 20:
            vol_arr = np.array(self.vol_history)
            vol_percentile = np.percentile(vol_arr, [np.sum(vol_arr < current_vol) / len(vol_arr) * 100])[0]
        else:
            vol_percentile = 50
        
        # Volatility clustering detection (GARCH-like)
        if len(returns) >= 10:
            squared_returns = returns[-10:] ** 2
            is_clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] > 0.3
        else:
            is_clustering = False
        
        # Expansion/contraction signals
        expansion_signal = vol_trend > 0.2 and current_vol > 0.015
        contraction_signal = vol_trend < -0.2 and current_vol < 0.01
        
        # Breakout potential (high vol after low vol period)
        breakout_potential = 0.5
        if len(self.vol_history) >= 10:
            recent_vol_range = max(self.vol_history) - min(self.vol_history)
            if recent_vol_range > 0.01:
                breakout_potential = 0.8
            elif recent_vol_range < 0.005:
                breakout_potential = 0.3
        
        return {
            "current_vol": current_vol,
            "vol_trend": vol_trend,
            "vol_percentile": vol_percentile,
            "is_clustering": is_clustering,
            "expansion_signal": expansion_signal,
            "contraction_signal": contraction_signal,
            "breakout_potential": breakout_potential
        }
    
    def detect_patterns(self) -> Dict:
        """Detect volatility patterns"""
        if len(self.returns) < 20:
            return {"pattern": "unknown", "confidence": 0}
        
        returns = np.array(self.returns)
        
        # Check for squeeze (low vol building up)
        recent_vol = np.std(returns[-10:])
        older_vol = np.std(returns[-20:-10])
        
        if recent_vol < older_vol * 0.7:
            return {"pattern": "squeeze", "confidence": 0.7}
        
        # Check for expansion
        if recent_vol > older_vol * 1.3:
            return {"pattern": "expansion", "confidence": 0.7}
        
        # Check for mean reversion setup
        last_move = abs(returns[-1])
        avg_move = np.mean(np.abs(returns[-20:]))
        
        if last_move > avg_move * 2:
            return {"pattern": "reversal_setup", "confidence": 0.6}
        
        # Check for trend continuation
        momentum = np.sum(returns[-5:])
        if abs(momentum) > 0.02:
            return {"pattern": "trend", "confidence": 0.6}
        
        return {"pattern": "ranging", "confidence": 0.5}


# ============================================================================
# ADAPTIVE STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """Selects optimal strategy based on volatility regime"""
    
    def __init__(self):
        self.strategies = {
            "momentum": self._momentum_signal,
            "reversal": self._reversal_signal,
            "breakout": self._breakout_signal,
            "squeeze_play": self._squeeze_signal,
        }
        
        self.strategy_performance: Dict[str, List[float]] = {
            k: [] for k in self.strategies
        }
    
    def _momentum_signal(self, returns: np.ndarray, vol_features: Dict) -> Tuple[int, float]:
        """Momentum following strategy"""
        if len(returns) < 5:
            return 0, 0.5
        
        momentum = np.sum(returns[-5:])
        
        if momentum > 0.015:
            return 1, min(0.9, 0.5 + momentum * 10)  # BUY
        elif momentum < -0.015:
            return 2, min(0.9, 0.5 + abs(momentum) * 10)  # SELL
        return 0, 0.4  # HOLD
    
    def _reversal_signal(self, returns: np.ndarray, vol_features: Dict) -> Tuple[int, float]:
        """Mean reversion strategy"""
        if len(returns) < 3:
            return 0, 0.5
        
        # Look for oversold/overbought
        recent = np.sum(returns[-3:])
        
        if recent < -0.02:  # Oversold
            return 1, min(0.85, 0.5 + abs(recent) * 10)  # BUY
        elif recent > 0.02:  # Overbought
            return 2, min(0.85, 0.5 + recent * 10)  # SELL
        return 0, 0.4
    
    def _breakout_signal(self, returns: np.ndarray, vol_features: Dict) -> Tuple[int, float]:
        """Breakout trading strategy"""
        if len(returns) < 10:
            return 0, 0.5
        
        # Check for high vol after low vol
        if vol_features["expansion_signal"]:
            recent_direction = np.sum(returns[-3:])
            if recent_direction > 0:
                return 1, 0.75  # BUY on upside breakout
            else:
                return 2, 0.75  # SELL on downside breakout
        
        return 0, 0.4
    
    def _squeeze_signal(self, returns: np.ndarray, vol_features: Dict) -> Tuple[int, float]:
        """Volatility squeeze strategy"""
        if vol_features["contraction_signal"]:
            # Wait for direction
            recent = np.sum(returns[-3:]) if len(returns) >= 3 else 0
            if recent > 0.005:
                return 1, 0.7
            elif recent < -0.005:
                return 2, 0.7
        return 0, 0.5
    
    def select_strategy(self, vol_features: Dict, pattern: str) -> str:
        """Select best strategy for current conditions"""
        current_vol = vol_features["current_vol"]
        
        if pattern == "squeeze":
            return "squeeze_play"
        elif pattern == "expansion" or pattern == "reversal_setup":
            if current_vol > 0.015:
                return "reversal"
            else:
                return "breakout"
        elif pattern == "trend":
            return "momentum"
        else:
            # Use historical performance
            perfs = {k: np.mean(v[-20:]) if v else 0 
                    for k, v in self.strategy_performance.items()}
            return max(perfs, key=perfs.get) if perfs else "momentum"
    
    def get_signal(self, strategy: str, returns: np.ndarray, 
                   vol_features: Dict) -> Tuple[int, float]:
        """Get trading signal from selected strategy"""
        return self.strategies[strategy](returns, vol_features)
    
    def update_performance(self, strategy: str, pnl: float):
        """Update strategy performance tracking"""
        self.strategy_performance[strategy].append(pnl)
        # Keep last 100
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy].pop(0)


# ============================================================================
# VOLATILITY SPECIALIST AGENT
# ============================================================================

class VolatilitySpecialist:
    """Q-agent specialized for volatile markets"""
    
    def __init__(self, n_states: int = 200):
        print(f"{E.BRAIN} Initializing Volatility Specialist")
        
        self.vol_analyzer = VolatilityAnalyzer()
        self.strategy_selector = StrategySelector()
        
        # Q-tables for each strategy
        self.q_tables = {
            "momentum": np.zeros((n_states, 4)),
            "reversal": np.zeros((n_states, 4)),
            "breakout": np.zeros((n_states, 4)),
            "squeeze_play": np.zeros((n_states, 4)),
        }
        self.visits = {k: np.zeros((n_states, 4)) for k in self.q_tables}
        
        # Hyperparameters
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 0.15
        self.n_states = n_states
        
        # State
        self.position = False
        self.entry_price = 0.0
        self.entry_strategy = "momentum"
        self.hold_duration = 0
        
        # Stats
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.episode_pnls: List[float] = []
        self.strategy_trades: Dict[str, int] = {k: 0 for k in self.q_tables}
        self.strategy_wins: Dict[str, int] = {k: 0 for k in self.q_tables}
        
        self.messages: List[str] = []
        
        print(f"  {E.CHECK} Strategies: {list(self.q_tables.keys())}")
    
    def _encode_state(self, vol_features: Dict, pattern: str) -> int:
        """Encode volatility state"""
        # Vol level (0-4)
        vol = vol_features["current_vol"]
        vol_bin = min(4, int(vol / 0.005))
        
        # Vol trend (0-2)
        trend = vol_features["vol_trend"]
        if trend > 0.1:
            trend_bin = 2
        elif trend < -0.1:
            trend_bin = 0
        else:
            trend_bin = 1
        
        # Pattern (0-4)
        pattern_map = {"squeeze": 0, "expansion": 1, "reversal_setup": 2, "trend": 3, "ranging": 4}
        pattern_bin = pattern_map.get(pattern, 4)
        
        # Position (0-1)
        pos_bin = 1 if self.position else 0
        
        # Breakout potential (0-2)
        bp = vol_features["breakout_potential"]
        bp_bin = 0 if bp < 0.4 else (1 if bp < 0.7 else 2)
        
        # 5 * 3 * 5 * 2 * 3 = 450, clip to n_states
        idx = vol_bin * 90 + trend_bin * 30 + pattern_bin * 6 + pos_bin * 3 + bp_bin
        return min(idx, self.n_states - 1)
    
    def step(self, price: float) -> Tuple[int, float, str]:
        """Execute one step"""
        # Update analyzer
        self.vol_analyzer.update(price)
        
        # Get volatility features and pattern
        vol_features = self.vol_analyzer.get_vol_features()
        pattern_info = self.vol_analyzer.detect_patterns()
        pattern = pattern_info["pattern"]
        
        # Select strategy
        strategy = self.strategy_selector.select_strategy(vol_features, pattern)
        
        # Get state
        state = self._encode_state(vol_features, pattern)
        
        # Get Q-values for selected strategy
        q_values = self.q_tables[strategy][state]
        
        # Select action (epsilon-greedy)
        if np.random.random() < self.epsilon:
            action = np.random.randint(4)
        else:
            # Combine Q-value with strategy signal
            strat_action, strat_conf = self.strategy_selector.get_signal(
                strategy, np.array(self.vol_analyzer.returns), vol_features
            )
            
            # Weighted combination
            adjusted_q = q_values.copy()
            adjusted_q[strat_action] += strat_conf
            action = int(np.argmax(adjusted_q))
        
        # Validate
        if action == 1 and self.position:
            action = 0
        elif action == 2 and not self.position:
            action = 0
        
        # Execute
        reward = 0.0
        
        if action == 1:  # BUY
            self.position = True
            self.entry_price = price
            self.entry_strategy = strategy
            self.hold_duration = 0
            
            # Small reward for good timing
            if vol_features["contraction_signal"]:
                reward = 0.02  # Buying in low vol
            else:
                reward = -0.01
                
        elif action == 2:  # SELL
            if self.entry_price > 0:
                pnl = (price - self.entry_price) / self.entry_price
                
                # Update strategy performance
                self.strategy_selector.update_performance(self.entry_strategy, pnl)
                
                # Reward shaping for volatile markets
                if pnl > 0:
                    reward = 0.5 + pnl * 15
                    # Bonus for quick wins in high vol
                    if self.hold_duration < 5 and vol_features["current_vol"] > 0.015:
                        reward *= 1.3
                else:
                    reward = -0.3 + pnl * 10
                    # Smaller penalty for quick stop in high vol
                    if self.hold_duration < 3 and vol_features["current_vol"] > 0.015:
                        reward *= 0.7
                
                self.total_pnl += pnl
                self.trades += 1
                self.strategy_trades[self.entry_strategy] += 1
                
                if pnl > 0:
                    self.wins += 1
                    self.strategy_wins[self.entry_strategy] += 1
            
            self.position = False
            
        else:  # HOLD
            if self.position:
                self.hold_duration += 1
                
                # Different hold rewards based on vol regime
                if self.entry_price > 0:
                    unrealized = (price - self.entry_price) / self.entry_price
                    
                    if vol_features["current_vol"] > 0.015:
                        # High vol: tighter trailing
                        if unrealized > 0.02:
                            reward = 0.01
                        elif unrealized < -0.01:
                            reward = -0.02  # Encourage exits
                    else:
                        # Low vol: can hold longer
                        reward = 0.002 * np.sign(unrealized)
            else:
                # Waiting reward
                if vol_features["contraction_signal"]:
                    reward = 0.005  # Patience in squeeze
                else:
                    reward = -0.001
        
        # Update Q-table
        next_features = self.vol_analyzer.get_vol_features()
        next_pattern = self.vol_analyzer.detect_patterns()["pattern"]
        next_state = self._encode_state(next_features, next_pattern)
        
        # Q-learning update
        next_q = np.max(self.q_tables[strategy][next_state])
        target = reward + self.gamma * next_q
        self.q_tables[strategy][state, action] += self.lr * (target - self.q_tables[strategy][state, action])
        self.visits[strategy][state, action] += 1
        
        # Generate message
        msg = self._generate_message(strategy, action, vol_features, pattern, reward)
        self.messages.append(msg)
        
        return action, reward, msg
    
    def _generate_message(self, strategy: str, action: int, 
                         vol_features: Dict, pattern: str, reward: float) -> str:
        """Generate volatility-focused message"""
        parts = []
        
        # Volatility meter
        vol = vol_features["current_vol"]
        parts.append(E.vol_meter(vol))
        
        # Vol status
        if vol > 0.02:
            parts.append(E.STORM)
        elif vol > 0.015:
            parts.append(E.ALERT)
        else:
            parts.append(E.CALM)
        
        # Pattern
        pattern_emoji = {
            "squeeze": E.SQUEEZE,
            "expansion": E.EXPANSION,
            "reversal_setup": E.REVERSAL,
            "trend": E.ROCKET,
            "ranging": "➡️"
        }
        parts.append(pattern_emoji.get(pattern, "❓"))
        
        # Strategy (first 3 chars)
        parts.append(f"[{strategy[:3]}]")
        
        # Action
        action_emoji = [E.HOLD, E.BUY, E.SELL, E.HOLD]
        parts.append(action_emoji[action])
        
        # Reward
        if reward > 0.3:
            parts.append(E.PROFIT)
        elif reward < -0.1:
            parts.append(E.LOSS)
        
        return " ".join(parts)
    
    def run_episode(self, prices: List[float], verbose: bool = False) -> float:
        """Run one episode"""
        ep_pnl = 0.0
        
        for i, price in enumerate(prices):
            action, reward, msg = self.step(price)
            ep_pnl += reward
            
            if verbose and i % 50 == 0:
                print(f"  Step {i:3d}: {msg}")
        
        self.episode_pnls.append(ep_pnl)
        return ep_pnl
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.trades),
            "strategy_performance": {
                k: {
                    "trades": self.strategy_trades[k],
                    "wins": self.strategy_wins[k],
                    "win_rate": self.strategy_wins[k] / max(1, self.strategy_trades[k])
                }
                for k in self.q_tables
            }
        }
    
    def print_status(self):
        """Print status"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"{E.BRAIN} Volatility Specialist Status")
        print(f"{'='*70}")
        
        pnl_emoji = E.PROFIT if stats["total_pnl"] > 0 else E.LOSS
        print(f"\n{E.TARGET} Performance:")
        print(f"  Total PnL: {pnl_emoji} {stats['total_pnl']*100:+.2f}%")
        print(f"  Trades: {stats['trades']} (Win Rate: {stats['win_rate']*100:.1f}%)")
        
        print(f"\n{E.CHART} Strategy Breakdown:")
        for strategy, perf in stats["strategy_performance"].items():
            if perf["trades"] > 0:
                emoji = E.CHECK if perf["win_rate"] > 0.5 else "❌"
                print(f"  {emoji} {strategy:12}: {perf['trades']:4} trades, "
                      f"Win: {perf['win_rate']*100:5.1f}%")


# ============================================================================
# TRAINING
# ============================================================================

def generate_volatile_prices(n: int, regime: str = "high_vol") -> List[float]:
    """Generate price series with volatility focus"""
    prices = [100.0]
    
    for i in range(n - 1):
        if regime == "high_vol":
            vol = 0.02 + 0.01 * np.sin(i * 0.1)
            drift = 0.0001 * np.sin(i * 0.05)
        elif regime == "squeeze":
            # Start high vol, compress, then expand
            phase = i / n
            if phase < 0.3:
                vol = 0.02
            elif phase < 0.6:
                vol = 0.005  # Squeeze
            else:
                vol = 0.025  # Expansion
            drift = 0.0002 * np.sin(i * 0.03)
        elif regime == "trending_vol":
            vol = 0.015
            drift = 0.0003  # Uptrend
        elif regime == "reversal":
            # Big moves that reverse
            vol = 0.018
            if i % 20 < 10:
                drift = 0.003
            else:
                drift = -0.003
        else:
            vol = 0.012
            drift = 0
        
        change = drift + np.random.randn() * vol
        prices.append(prices[-1] * (1 + change))
    
    return prices


def run_volatility_training():
    """Run volatility specialist training"""
    print(f"{'='*70}")
    print(f"{E.STORM} VOLATILITY SPECIALIST TRAINING {E.STORM}")
    print(f"{'='*70}")
    
    agent = VolatilitySpecialist(n_states=200)
    
    # Training phases
    regimes = ["high_vol", "squeeze", "trending_vol", "reversal", "mixed"]
    
    for phase in range(4):
        print(f"\n{E.STAR} Phase {phase + 1}")
        print("-" * 50)
        
        agent.epsilon = max(0.05, 0.2 - phase * 0.05)
        phase_rewards = []
        
        for ep in range(50):
            regime = regimes[ep % len(regimes)]
            prices = generate_volatile_prices(300, regime)
            reward = agent.run_episode(prices, verbose=False)
            phase_rewards.append(reward)
            
            if (ep + 1) % 12 == 0:
                stats = agent.get_stats()
                print(f"  Ep {ep+1:3d}: R={np.mean(phase_rewards[-10:]):7.3f} "
                      f"Win={stats['win_rate']*100:5.1f}%")
        
        print(f"  {E.CHECK} Phase avg: {np.mean(phase_rewards):.3f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{E.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    agent.epsilon = 0.0
    
    for regime in regimes:
        prices = generate_volatile_prices(400, regime)
        reward = agent.run_episode(prices, verbose=False)
        emoji = {"high_vol": E.STORM, "squeeze": E.SQUEEZE, 
                "trending_vol": E.ROCKET, "reversal": E.REVERSAL,
                "mixed": "➡️"}.get(regime, "❓")
        print(f"  {emoji} {regime:15}: R={reward:8.3f}")
    
    agent.print_status()
    
    # Show messages
    print(f"\n{E.FIRE} Recent Communications:")
    for msg in agent.messages[-10:]:
        print(f"  {msg}")
    
    return agent


if __name__ == "__main__":
    agent = run_volatility_training()

