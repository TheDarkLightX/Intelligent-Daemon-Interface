#!/usr/bin/env python3
"""
MASTER Q-AGENT v4
================
Combines ALL best features into one unified agent:
- Ensemble Q-learning (5 members with weighted voting)
- Market regime detection (bull/bear/high-vol/low-vol)
- Volatility-adaptive strategies (momentum/reversal/breakout/squeeze)
- Dynamic position sizing (Kelly + confidence)
- Multi-agent consensus
- Emotional intelligence
- Rich emoji communication
- Performance attribution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from collections import deque
from enum import IntEnum
import time


# ============================================================================
# COMPREHENSIVE EMOJI SYSTEM
# ============================================================================

class Mood(IntEnum):
    EUPHORIC = 0; EXCITED = 1; CONFIDENT = 2; CALM = 3
    CAUTIOUS = 4; NERVOUS = 5; FEARFUL = 6; PANICKED = 7


class Regime(IntEnum):
    STRONG_BULL = 0; BULL = 1; NEUTRAL = 2; BEAR = 3; STRONG_BEAR = 4
    HIGH_VOL = 5; LOW_VOL = 6; SQUEEZE = 7; BREAKOUT = 8


class E:
    """Master emoji vocabulary"""
    # Moods
    MOODS = {
        Mood.EUPHORIC: "🚀", Mood.EXCITED: "🤩", Mood.CONFIDENT: "😎",
        Mood.CALM: "😊", Mood.CAUTIOUS: "🤔", Mood.NERVOUS: "😟",
        Mood.FEARFUL: "😰", Mood.PANICKED: "😱"
    }
    
    # Regimes
    REGIMES = {
        Regime.STRONG_BULL: "🐂🔥", Regime.BULL: "🐂", Regime.NEUTRAL: "➡️",
        Regime.BEAR: "🐻", Regime.STRONG_BEAR: "🐻💀",
        Regime.HIGH_VOL: "🌪️", Regime.LOW_VOL: "😴",
        Regime.SQUEEZE: "🤏", Regime.BREAKOUT: "💥"
    }
    
    # Core actions
    BUY = "🟢"; SELL = "🔴"; HOLD = "⏸️"; SCALE = "📊"
    
    # Status
    PROFIT = "💰"; LOSS = "📉"; FIRE = "🔥"; SKULL = "💀"
    STAR = "⭐"; CHECK = "✅"; CROSS = "❌"; ALERT = "🚨"
    BRAIN = "🧠"; TROPHY = "🏆"; TARGET = "🎯"; ROCKET = "🚀"
    DIAMOND = "💎"; CHART = "📊"
    
    @staticmethod
    def bar(value: float, width: int = 5) -> str:
        filled = int(min(1, max(0, value)) * width)
        return "█" * filled + "░" * (width - filled)
    
    @staticmethod
    def sparkline(values: List[float]) -> str:
        if len(values) < 3:
            return "▁▁▁"
        blocks = "▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        return "".join(blocks[min(7, int((v - mn) / rng * 8))] for v in values[-5:])


# ============================================================================
# UNIFIED ANALYZER
# ============================================================================

class UnifiedAnalyzer:
    """Combined market analysis"""
    
    def __init__(self, lookback: int = 100):
        self.prices: deque = deque(maxlen=lookback)
        self.returns: deque = deque(maxlen=lookback)
        self.vol_history: deque = deque(maxlen=50)
        
    def update(self, price: float):
        if self.prices:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
        self.prices.append(price)
        
        if len(self.returns) >= 10:
            vol = np.std(list(self.returns)[-20:])
            self.vol_history.append(vol)
    
    def get_features(self) -> Dict:
        """Get all features"""
        if len(self.returns) < 5:
            return self._default_features()
        
        returns = np.array(self.returns)
        
        # Momentum
        mom_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        mom_20 = np.mean(returns[-20:]) if len(returns) >= 20 else mom_5
        
        # Volatility
        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.015
        vol_trend = 0
        if len(self.vol_history) >= 10:
            vol_arr = np.array(self.vol_history)
            vol_trend = (vol_arr[-5:].mean() - vol_arr[-10:-5].mean()) / (vol_arr.mean() + 0.001)
        
        # Trend strength
        trend_strength = mom_5 / (vol + 0.001)
        
        # Pattern detection
        pattern = self._detect_pattern(returns, vol)
        
        return {
            "price": self.prices[-1] if self.prices else 100,
            "momentum_short": mom_5,
            "momentum_long": mom_20,
            "volatility": vol,
            "vol_trend": vol_trend,
            "trend_strength": trend_strength,
            "pattern": pattern,
            "returns_history": list(returns[-10:])
        }
    
    def _default_features(self) -> Dict:
        return {
            "price": 100, "momentum_short": 0, "momentum_long": 0,
            "volatility": 0.015, "vol_trend": 0, "trend_strength": 0,
            "pattern": "unknown", "returns_history": []
        }
    
    def _detect_pattern(self, returns: np.ndarray, vol: float) -> str:
        if len(returns) < 10:
            return "unknown"
        
        recent_vol = np.std(returns[-5:])
        older_vol = np.std(returns[-15:-5]) if len(returns) >= 15 else vol
        
        if recent_vol < older_vol * 0.6:
            return "squeeze"
        elif recent_vol > older_vol * 1.4:
            return "expansion"
        
        mom = np.sum(returns[-5:])
        if mom > 0.02:
            return "uptrend"
        elif mom < -0.02:
            return "downtrend"
        
        return "ranging"
    
    def detect_regime(self) -> Tuple[Regime, float]:
        """Detect market regime"""
        features = self.get_features()
        
        vol = features["volatility"]
        mom = features["momentum_short"]
        pattern = features["pattern"]
        
        confidence = 0.6
        
        if pattern == "squeeze":
            return Regime.SQUEEZE, 0.7
        elif pattern == "expansion":
            return Regime.BREAKOUT, 0.7
        
        if vol > 0.025:
            return Regime.HIGH_VOL, min(0.9, vol / 0.03)
        elif vol < 0.008:
            return Regime.LOW_VOL, min(0.9, 0.015 / (vol + 0.001))
        
        if mom > 0.03:
            return Regime.STRONG_BULL, min(0.85, 0.5 + mom * 10)
        elif mom > 0.01:
            return Regime.BULL, 0.65
        elif mom < -0.03:
            return Regime.STRONG_BEAR, min(0.85, 0.5 + abs(mom) * 10)
        elif mom < -0.01:
            return Regime.BEAR, 0.65
        
        return Regime.NEUTRAL, 0.5


# ============================================================================
# ENSEMBLE Q-TABLE
# ============================================================================

class MasterEnsemble:
    """Ensemble of specialized Q-tables"""
    
    def __init__(self, n_states: int, n_actions: int = 4):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Specialized Q-tables
        self.tables = {
            "momentum": np.zeros((n_states, n_actions)),
            "reversal": np.zeros((n_states, n_actions)),
            "breakout": np.zeros((n_states, n_actions)),
            "conservative": np.zeros((n_states, n_actions)),
            "aggressive": np.zeros((n_states, n_actions)),
        }
        
        # Performance tracking
        self.performance = {k: deque(maxlen=100) for k in self.tables}
        self.visits = {k: np.zeros((n_states, n_actions)) for k in self.tables}
        
        # Hyperparameters per table
        self.params = {
            "momentum": {"lr": 0.15, "gamma": 0.95},
            "reversal": {"lr": 0.1, "gamma": 0.9},
            "breakout": {"lr": 0.2, "gamma": 0.92},
            "conservative": {"lr": 0.08, "gamma": 0.98},
            "aggressive": {"lr": 0.25, "gamma": 0.9},
        }
        
        self.epsilon = 0.15
        self.total_updates = 0
    
    def select_table(self, regime: Regime, pattern: str) -> str:
        """Select best table for current conditions"""
        if regime in [Regime.STRONG_BULL, Regime.BULL]:
            return "momentum"
        elif regime in [Regime.STRONG_BEAR, Regime.BEAR]:
            return "reversal"
        elif regime in [Regime.BREAKOUT, Regime.HIGH_VOL]:
            return "breakout"
        elif regime == Regime.LOW_VOL:
            return "conservative"
        elif regime == Regime.SQUEEZE:
            return "aggressive"
        else:
            # Use best performing
            avg_perfs = {k: np.mean(list(v)) if v else 0 
                        for k, v in self.performance.items()}
            return max(avg_perfs, key=avg_perfs.get)
    
    def get_weights(self) -> Dict[str, float]:
        """Get voting weights"""
        perfs = {k: np.mean(list(v)) + 0.1 if v else 0.1 
                for k, v in self.performance.items()}
        total = sum(max(0.01, p) for p in perfs.values())
        return {k: max(0.01, p) / total for k, p in perfs.items()}
    
    def select_action(self, state: int, regime: Regime, pattern: str,
                     explore: bool = True) -> Tuple[int, str, Dict]:
        """Select action using ensemble"""
        # Primary table
        primary = self.select_table(regime, pattern)
        primary_q = self.tables[primary][state]
        
        # Weighted ensemble
        weights = self.get_weights()
        combined_q = np.zeros(self.n_actions)
        for name, table in self.tables.items():
            combined_q += weights[name] * table[state]
        
        # Exploration
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Use combined Q with primary bonus
            final_q = 0.6 * primary_q + 0.4 * combined_q
            action = int(np.argmax(final_q))
        
        # Agreement
        all_actions = [int(np.argmax(t[state])) for t in self.tables.values()]
        agreement = all_actions.count(action) / len(all_actions)
        
        info = {
            "primary_table": primary,
            "weights": weights,
            "agreement": agreement,
            "all_q": {k: t[state].tolist() for k, t in self.tables.items()}
        }
        
        return action, primary, info
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, table_name: str):
        """Update specific table"""
        params = self.params[table_name]
        table = self.tables[table_name]
        
        target = reward + params["gamma"] * np.max(table[next_state])
        table[state, action] += params["lr"] * (target - table[state, action])
        
        self.visits[table_name][state, action] += 1
        self.performance[table_name].append(reward)
        self.total_updates += 1


# ============================================================================
# POSITION MANAGER
# ============================================================================

class PositionManager:
    """Manages position sizing and risk"""
    
    def __init__(self):
        self.trades: List[float] = []
        self.base_size = 1.0
        
    def update(self, pnl: float):
        self.trades.append(pnl)
    
    def kelly(self) -> float:
        if len(self.trades) < 10:
            return 0.5
        
        wins = [t for t in self.trades[-50:] if t > 0]
        losses = [t for t in self.trades[-50:] if t < 0]
        
        if not wins or not losses:
            return 0.5
        
        p = len(wins) / (len(wins) + len(losses))
        b = np.mean(wins) / abs(np.mean(losses)) if losses else 1
        
        kelly = (b * p - (1 - p)) / b if b > 0 else 0
        return max(0.1, min(1.0, kelly * 0.5))
    
    def size(self, confidence: float, agreement: float, vol: float) -> float:
        k = self.kelly()
        conf_adj = 0.5 + confidence * 0.5
        agree_adj = 0.6 + agreement * 0.4
        vol_adj = max(0.5, 1.5 - vol * 15)
        
        return max(0.25, min(2.0, self.base_size * k * conf_adj * agree_adj * vol_adj))


# ============================================================================
# MASTER AGENT
# ============================================================================

class MasterAgent:
    """The ultimate Q-agent combining all features"""
    
    def __init__(self, n_states: int = 500):
        print(f"{E.BRAIN} Initializing MASTER AGENT v4")
        
        self.analyzer = UnifiedAnalyzer()
        self.ensemble = MasterEnsemble(n_states)
        self.position_mgr = PositionManager()
        
        # State
        self.position = False
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_regime = Regime.NEUTRAL
        self.entry_table = "momentum"
        self.hold_duration = 0
        self.streak = 0
        
        # Stats
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.mood = Mood.CALM
        self.confidence = 0.5
        
        # History
        self.episode_pnls: List[float] = []
        self.pnl_history: deque = deque(maxlen=100)
        self.messages: List[str] = []
        self.trade_log: List[Dict] = []
        
        print(f"  {E.CHECK} Ensemble: {len(self.ensemble.tables)} specialized tables")
        print(f"  {E.CHECK} State space: {n_states}")
    
    def _encode_state(self, features: Dict, regime: Regime) -> int:
        """Encode comprehensive state"""
        # Momentum (0-9)
        m = min(9, max(0, int((features["momentum_short"] + 0.05) / 0.01)))
        
        # Volatility (0-4)
        v = min(4, int(features["volatility"] / 0.006))
        
        # Regime (0-8)
        r = regime.value
        
        # Position state (0-2)
        if not self.position:
            p = 0
        else:
            unr = (features["price"] - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
            p = 1 if unr > 0 else 2
        
        # Streak (0-2)
        s = 0 if self.streak < -1 else (1 if self.streak < 2 else 2)
        
        # 10 * 5 * 9 * 3 * 3 = 4050, clip
        idx = m * 405 + v * 81 + r * 9 + p * 3 + s
        return min(idx, self.ensemble.n_states - 1)
    
    def _update_mood(self):
        """Update emotional state"""
        if len(self.pnl_history) < 5:
            self.mood = Mood.CALM
            return
        
        recent = list(self.pnl_history)[-20:]
        avg = np.mean(recent)
        win_rate = self.wins / max(1, self.trades)
        
        if win_rate > 0.65 and avg > 0.02:
            self.mood = Mood.EUPHORIC
        elif win_rate > 0.58 and avg > 0:
            self.mood = Mood.EXCITED
        elif win_rate > 0.52:
            self.mood = Mood.CONFIDENT
        elif win_rate > 0.48:
            self.mood = Mood.CALM
        elif win_rate > 0.42:
            self.mood = Mood.CAUTIOUS
        elif win_rate > 0.38:
            self.mood = Mood.NERVOUS
        elif win_rate > 0.35:
            self.mood = Mood.FEARFUL
        else:
            self.mood = Mood.PANICKED
    
    def step(self, price: float) -> Tuple[int, float, str]:
        """Execute one step"""
        # Update analyzer
        self.analyzer.update(price)
        
        # Get analysis
        features = self.analyzer.get_features()
        regime, regime_conf = self.analyzer.detect_regime()
        pattern = features["pattern"]
        
        # Encode state
        state = self._encode_state(features, regime)
        
        # Get action from ensemble
        action, table_used, info = self.ensemble.select_action(
            state, regime, pattern
        )
        agreement = info["agreement"]
        
        # Update confidence
        self.confidence = 0.3 + regime_conf * 0.3 + agreement * 0.4
        
        # Position sizing
        size = self.position_mgr.size(
            self.confidence, agreement, features["volatility"]
        )
        
        # Validate action
        if action == 1 and self.position:
            action = 0
        elif action == 2 and not self.position:
            action = 0
        
        # Execute
        reward = 0.0
        
        if action == 1:  # BUY
            self.position = True
            self.position_size = size
            self.entry_price = price
            self.entry_regime = regime
            self.entry_table = table_used
            self.hold_duration = 0
            
            reward = -0.005 * size
            if features["volatility"] < 0.01:
                reward += 0.01
                
        elif action == 2:  # SELL
            if self.entry_price > 0:
                pnl = (price - self.entry_price) / self.entry_price * self.position_size
                
                # Shaped reward
                if pnl > 0:
                    reward = 0.5 + pnl * 15
                    if self.hold_duration < 10:
                        reward *= 1.2
                else:
                    reward = -0.2 + pnl * 10
                    if self.hold_duration < 5:
                        reward *= 0.8
                
                # Update trackers
                self.position_mgr.update(pnl)
                self.total_pnl += pnl
                self.pnl_history.append(pnl)
                self.trades += 1
                
                if pnl > 0:
                    self.wins += 1
                    self.streak = max(1, self.streak + 1)
                else:
                    self.streak = min(-1, self.streak - 1)
                
                # Log trade
                self.trade_log.append({
                    "entry_regime": self.entry_regime.name,
                    "exit_regime": regime.name,
                    "table": self.entry_table,
                    "pnl": pnl,
                    "hold": self.hold_duration
                })
            
            self.position = False
            self.position_size = 0
            
        else:  # HOLD
            if self.position:
                self.hold_duration += 1
                if self.entry_price > 0:
                    unrealized = (price - self.entry_price) / self.entry_price
                    reward = 0.003 * np.sign(unrealized) * (1 + abs(unrealized) * 5)
            else:
                reward = -0.001
        
        # Update ensemble
        next_features = self.analyzer.get_features()
        next_regime, _ = self.analyzer.detect_regime()
        next_state = self._encode_state(next_features, next_regime)
        
        self.ensemble.update(state, action, reward, next_state, table_used)
        
        # Update mood
        self._update_mood()
        
        # Generate message
        msg = self._message(action, regime, table_used, info, features)
        self.messages.append(msg)
        
        return action, reward, msg
    
    def _message(self, action: int, regime: Regime, table: str, 
                info: Dict, features: Dict) -> str:
        """Generate rich message"""
        parts = []
        
        # Mood
        parts.append(E.MOODS[self.mood])
        
        # Regime
        parts.append(E.REGIMES[regime])
        
        # Table indicator
        table_icons = {
            "momentum": "⚡", "reversal": "🔄", "breakout": "💥",
            "conservative": "🛡️", "aggressive": "⚔️"
        }
        parts.append(table_icons.get(table, "?"))
        
        # Action
        parts.append([E.HOLD, E.BUY, E.SELL, E.HOLD][action])
        
        # Confidence bar
        parts.append(f"[{E.bar(self.confidence)}]")
        
        # Agreement indicator
        if info["agreement"] > 0.8:
            parts.append(E.CHECK)
        elif info["agreement"] < 0.4:
            parts.append(E.ALERT)
        
        # Streak
        if self.streak > 2:
            parts.append(E.FIRE)
        elif self.streak < -2:
            parts.append(E.SKULL)
        
        # Position status
        if self.position and self.entry_price > 0:
            unr = (features["price"] - self.entry_price) / self.entry_price
            if unr > 0.02:
                parts.append(E.PROFIT)
            elif unr < -0.02:
                parts.append(E.LOSS)
        
        return " ".join(parts)
    
    def run_episode(self, prices: List[float], verbose: bool = False) -> float:
        """Run one episode"""
        ep_pnl = 0.0
        
        for i, price in enumerate(prices):
            action, reward, msg = self.step(price)
            ep_pnl += reward
            
            if verbose and i % 60 == 0:
                print(f"  {i:3d}: {msg} R:{reward:+.3f}")
        
        self.episode_pnls.append(ep_pnl)
        return ep_pnl
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats"""
        return {
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.trades),
            "mood": self.mood.name,
            "confidence": self.confidence,
            "streak": self.streak,
            "kelly": self.position_mgr.kelly(),
            "ensemble_weights": self.ensemble.get_weights(),
            "episode_avg": np.mean(self.episode_pnls[-20:]) if self.episode_pnls else 0
        }
    
    def print_status(self):
        """Print full status"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"{E.TROPHY} MASTER AGENT v4 STATUS {E.MOODS[self.mood]}")
        print(f"{'='*70}")
        
        pnl_emoji = E.PROFIT if stats["total_pnl"] > 0 else E.LOSS
        print(f"\n{E.TARGET} Performance:")
        print(f"  PnL: {pnl_emoji} {stats['total_pnl']*100:+.2f}%")
        print(f"  Trades: {stats['trades']} | Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Streak: {stats['streak']} | Kelly: {stats['kelly']:.2f}")
        
        print(f"\n{E.BRAIN} Ensemble Weights:")
        for name, weight in stats["ensemble_weights"].items():
            bar = E.bar(weight * 5)
            print(f"  {name:12}: [{bar}] {weight:.2f}")
        
        print(f"\n{E.CHART} Recent PnL Sparkline: {E.sparkline(list(self.pnl_history)[-20:])}")
        
        # Trade attribution
        if self.trade_log:
            print(f"\n{E.STAR} Trade Attribution (last 50):")
            recent = self.trade_log[-50:]
            
            by_table = {}
            for t in recent:
                tbl = t["table"]
                if tbl not in by_table:
                    by_table[tbl] = {"count": 0, "wins": 0, "pnl": 0}
                by_table[tbl]["count"] += 1
                by_table[tbl]["pnl"] += t["pnl"]
                if t["pnl"] > 0:
                    by_table[tbl]["wins"] += 1
            
            for tbl, data in sorted(by_table.items(), key=lambda x: -x[1]["pnl"]):
                wr = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
                emoji = E.CHECK if wr > 50 else E.CROSS
                print(f"  {emoji} {tbl:12}: {data['count']:3} trades | "
                      f"Win: {wr:5.1f}% | PnL: {data['pnl']*100:+6.2f}%")


# ============================================================================
# TRAINING
# ============================================================================

def gen_prices(n: int, regime: str = "mixed") -> List[float]:
    """Generate price series"""
    prices = [100.0]
    
    for i in range(n - 1):
        if regime == "bull":
            drift, vol = 0.0004, 0.012
        elif regime == "bear":
            drift, vol = -0.0004, 0.012
        elif regime == "high_vol":
            drift, vol = 0.0001, 0.025
        elif regime == "low_vol":
            drift, vol = 0.0001, 0.006
        elif regime == "squeeze":
            phase = i / n
            vol = 0.02 if phase < 0.3 else (0.005 if phase < 0.6 else 0.025)
            drift = 0.0002 * (1 if phase > 0.6 else 0)
        else:  # mixed
            drift = 0.0001 * np.sin(i * 0.03)
            vol = 0.012 + 0.005 * np.sin(i * 0.05)
        
        prices.append(prices[-1] * (1 + drift + np.random.randn() * vol))
    
    return prices


def run_master_training():
    """Run master agent training"""
    print(f"{'='*70}")
    print(f"{E.ROCKET} MASTER AGENT v4 TRAINING {E.ROCKET}")
    print(f"{'='*70}")
    
    agent = MasterAgent(n_states=500)
    
    phases = [
        ("Exploration", 40, 0.25),
        ("Learning", 80, 0.15),
        ("Optimization", 120, 0.08),
        ("Mastery", 160, 0.03),
    ]
    
    regimes = ["mixed", "bull", "bear", "high_vol", "low_vol", "squeeze"]
    
    for phase_name, episodes, epsilon in phases:
        print(f"\n{E.STAR} Phase: {phase_name}")
        print("-" * 50)
        
        agent.ensemble.epsilon = epsilon
        phase_rewards = []
        
        for ep in range(episodes):
            regime = regimes[ep % len(regimes)]
            prices = gen_prices(350, regime)
            reward = agent.run_episode(prices)
            phase_rewards.append(reward)
            
            if (ep + 1) % max(1, episodes // 4) == 0:
                stats = agent.get_stats()
                print(f"  Ep {ep+1:3d}: {E.MOODS[agent.mood]} "
                      f"R={np.mean(phase_rewards[-10:]):7.3f} "
                      f"Win={stats['win_rate']*100:5.1f}% "
                      f"Kelly={stats['kelly']:.2f}")
        
        print(f"  {E.CHECK} Phase: {np.mean(phase_rewards):.3f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{E.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    agent.ensemble.epsilon = 0.0
    
    for regime in regimes:
        prices = gen_prices(500, regime)
        reward = agent.run_episode(prices)
        emoji = {"bull": "🐂", "bear": "🐻", "high_vol": "🌪️", 
                "low_vol": "😴", "squeeze": "🤏", "mixed": "➡️"}.get(regime, "?")
        stats = agent.get_stats()
        print(f"  {emoji} {regime:10}: R={reward:8.3f} Win={stats['win_rate']*100:.1f}%")
    
    agent.print_status()
    
    print(f"\n{E.FIRE} Recent Messages:")
    for msg in agent.messages[-12:]:
        print(f"  {msg}")
    
    return agent


if __name__ == "__main__":
    agent = run_master_training()

