#!/usr/bin/env python3
"""
Multi-Agent Q-Learning System
- Multiple agents with different strategies
- Communication via emoji signals
- Consensus-based decision making
- Competitive and cooperative modes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
import time


# ============================================================================
# EMOJI COMMUNICATION PROTOCOL
# ============================================================================

class Signal(IntEnum):
    """Inter-agent communication signals"""
    NEUTRAL = 0
    BULLISH = 1
    BEARISH = 2
    BUY_NOW = 3
    SELL_NOW = 4
    HOLD_STRONG = 5
    DANGER = 6
    OPPORTUNITY = 7


class E:
    """Emoji shortcuts"""
    UP = "📈"; DOWN = "📉"; MONEY = "💰"; FIRE = "🔥"
    ROCKET = "🚀"; TROPHY = "🏆"; BRAIN = "🧠"; TARGET = "🎯"
    CHECK = "✅"; STAR = "⭐"; DIAMOND = "💎"; SKULL = "💀"
    HAPPY = "😊"; EXCITED = "🤩"; NEUTRAL = "😐"; WORRIED = "😟"
    BUY = "🟢"; SELL = "🔴"; HOLD = "⏸️"; BURN = "🔥"
    BULL = "🐂"; BEAR = "🐻"; ALERT = "🚨"; THINK = "🤔"
    
    SIGNALS = {
        Signal.NEUTRAL: "😐",
        Signal.BULLISH: "🐂",
        Signal.BEARISH: "🐻",
        Signal.BUY_NOW: "🟢",
        Signal.SELL_NOW: "🔴",
        Signal.HOLD_STRONG: "💎",
        Signal.DANGER: "🚨",
        Signal.OPPORTUNITY: "⭐",
    }


# ============================================================================
# SIMPLE Q-TABLE
# ============================================================================

class SimpleQTable:
    """Lightweight Q-table for individual agents"""
    
    def __init__(self, n_states: int, n_actions: int = 4):
        self.q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions))
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 0.1
    
    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.q[state]))
    
    def update(self, s: int, a: int, r: float, ns: int):
        target = r + self.gamma * np.max(self.q[ns])
        self.q[s, a] += self.lr * (target - self.q[s, a])
        self.visits[s, a] += 1


# ============================================================================
# INDIVIDUAL AGENT
# ============================================================================

@dataclass
class AgentState:
    """State for individual agent"""
    position: bool = False
    entry_price: float = 0.0
    streak: int = 0
    total_reward: float = 0.0
    trades: int = 0
    wins: int = 0


class TradingAgent:
    """Individual trading agent with personality"""
    
    def __init__(self, name: str, strategy: str, n_states: int = 100):
        self.name = name
        self.strategy = strategy  # "momentum", "contrarian", "conservative", "aggressive"
        self.q_table = SimpleQTable(n_states)
        self.state = AgentState()
        
        # Personality parameters
        if strategy == "momentum":
            self.risk_tolerance = 0.7
            self.trend_sensitivity = 0.8
        elif strategy == "contrarian":
            self.risk_tolerance = 0.5
            self.trend_sensitivity = -0.6
        elif strategy == "conservative":
            self.risk_tolerance = 0.3
            self.trend_sensitivity = 0.4
        else:  # aggressive
            self.risk_tolerance = 0.9
            self.trend_sensitivity = 0.5
        
        # Communication
        self.last_signal = Signal.NEUTRAL
        self.confidence = 0.5
    
    def encode_state(self, momentum: float, volatility: float, 
                    peer_signals: List[Signal]) -> int:
        """Encode market + peer signals to state"""
        # Momentum bin (0-9)
        m_bin = min(9, max(0, int((momentum + 0.1) / 0.02)))
        
        # Volatility bin (0-4)
        v_bin = min(4, int(volatility / 0.02))
        
        # Position bin (0-1)
        p_bin = 1 if self.state.position else 0
        
        # Peer consensus (0-2)
        bullish = sum(1 for s in peer_signals if s in [Signal.BULLISH, Signal.BUY_NOW])
        bearish = sum(1 for s in peer_signals if s in [Signal.BEARISH, Signal.SELL_NOW])
        if bullish > bearish:
            peer_bin = 2
        elif bearish > bullish:
            peer_bin = 0
        else:
            peer_bin = 1
        
        # Combine (10 * 5 * 2 * 3 = 300 states max, but we limit to 100)
        idx = m_bin * 10 + v_bin * 2 + p_bin
        return min(idx, 99)
    
    def generate_signal(self, momentum: float, volatility: float) -> Signal:
        """Generate communication signal"""
        adjusted_momentum = momentum * self.trend_sensitivity
        
        # Confidence based on streak
        self.confidence = 0.5 + self.state.streak * 0.1
        self.confidence = max(0.1, min(0.9, self.confidence))
        
        if self.state.position:
            unrealized = 0.0  # Simplified
            if unrealized > 0.03:
                self.last_signal = Signal.HOLD_STRONG
            elif unrealized < -0.03:
                self.last_signal = Signal.DANGER
            elif adjusted_momentum > 0.02:
                self.last_signal = Signal.BULLISH
            else:
                self.last_signal = Signal.NEUTRAL
        else:
            if adjusted_momentum > 0.03:
                self.last_signal = Signal.BUY_NOW if self.confidence > 0.6 else Signal.BULLISH
            elif adjusted_momentum < -0.03:
                self.last_signal = Signal.BEARISH
            elif volatility < 0.02:
                self.last_signal = Signal.OPPORTUNITY
            else:
                self.last_signal = Signal.NEUTRAL
        
        return self.last_signal
    
    def decide(self, state: int, peer_signals: List[Signal]) -> int:
        """Make trading decision considering peers"""
        base_action = self.q_table.select_action(state)
        
        # Count peer opinions
        bullish = sum(1 for s in peer_signals if s in [Signal.BULLISH, Signal.BUY_NOW, Signal.OPPORTUNITY])
        bearish = sum(1 for s in peer_signals if s in [Signal.BEARISH, Signal.SELL_NOW, Signal.DANGER])
        
        # Adjust based on peer consensus (with personality weight)
        peer_weight = 0.3 * (1 - self.risk_tolerance)  # Conservative listens more to peers
        
        if bullish > bearish + 1 and not self.state.position:
            # Strong bullish consensus - consider buying
            if np.random.random() < peer_weight:
                return 1  # BUY
        elif bearish > bullish + 1 and self.state.position:
            # Strong bearish consensus - consider selling
            if np.random.random() < peer_weight:
                return 2  # SELL
        
        return base_action
    
    def step(self, action: int, price_change: float, price: float) -> float:
        """Execute action and return reward"""
        reward = 0.0
        
        # Validate
        if action == 1 and self.state.position:
            action = 0
        elif action == 2 and not self.state.position:
            action = 0
        
        if action == 1:  # BUY
            self.state.position = True
            self.state.entry_price = price
            reward = -0.01
        elif action == 2:  # SELL
            if self.state.entry_price > 0:
                pnl = (price - self.state.entry_price) / self.state.entry_price
                reward = 0.5 + pnl * 10
                self.state.trades += 1
                if pnl > 0:
                    self.state.wins += 1
                    self.state.streak = max(1, self.state.streak + 1)
                else:
                    self.state.streak = min(-1, self.state.streak - 1)
            self.state.position = False
        else:
            reward = 0.001 if self.state.position else -0.001
        
        self.state.total_reward += reward
        return reward
    
    def get_emoji_status(self) -> str:
        """Get emoji status string"""
        signal_emoji = E.SIGNALS[self.last_signal]
        
        if self.state.streak > 2:
            mood = E.FIRE
        elif self.state.streak < -2:
            mood = E.SKULL
        elif self.confidence > 0.6:
            mood = E.HAPPY
        else:
            mood = E.NEUTRAL
        
        pos = E.MONEY if self.state.position else "💤"
        
        return f"{self.name[:3]} {mood} {signal_emoji} {pos}"


# ============================================================================
# MULTI-AGENT SYSTEM
# ============================================================================

class MultiAgentSystem:
    """Coordinates multiple trading agents"""
    
    def __init__(self, n_agents: int = 4):
        strategies = ["momentum", "contrarian", "conservative", "aggressive"]
        self.agents = [
            TradingAgent(f"Agent_{i}", strategies[i % len(strategies)])
            for i in range(n_agents)
        ]
        
        self.prices: List[float] = [100.0]
        self.consensus_history: List[str] = []
        self.total_steps = 0
    
    def _get_momentum(self) -> float:
        if len(self.prices) < 5:
            return 0.0
        return (self.prices[-1] - self.prices[-5]) / self.prices[-5]
    
    def _get_volatility(self) -> float:
        if len(self.prices) < 10:
            return 0.02
        returns = np.diff(self.prices[-10:]) / np.array(self.prices[-10:-1])
        return np.std(returns)
    
    def step(self, price_up: bool) -> Dict:
        """Execute one step for all agents"""
        # Update price
        change = 0.01 if price_up else -0.01
        new_price = self.prices[-1] * (1 + change)
        self.prices.append(new_price)
        if len(self.prices) > 100:
            self.prices.pop(0)
        
        momentum = self._get_momentum()
        volatility = self._get_volatility()
        
        # Phase 1: Generate signals
        signals = []
        for agent in self.agents:
            signal = agent.generate_signal(momentum, volatility)
            signals.append(signal)
        
        # Phase 2: Each agent decides considering peer signals
        actions = []
        rewards = []
        
        for i, agent in enumerate(self.agents):
            peer_signals = signals[:i] + signals[i+1:]  # Exclude own signal
            
            state = agent.encode_state(momentum, volatility, peer_signals)
            action = agent.decide(state, peer_signals)
            reward = agent.step(action, change, new_price)
            
            # Update Q-table
            next_state = agent.encode_state(
                self._get_momentum(), self._get_volatility(), peer_signals
            )
            agent.q_table.update(state, action, reward, next_state)
            
            actions.append(action)
            rewards.append(reward)
        
        # Build consensus message
        signal_emojis = [E.SIGNALS[s] for s in signals]
        action_emojis = [E.BUY if a == 1 else E.SELL if a == 2 else E.HOLD for a in actions]
        
        consensus = f"Signals: {' '.join(signal_emojis)} | Actions: {' '.join(action_emojis)}"
        self.consensus_history.append(consensus)
        
        self.total_steps += 1
        
        return {
            "signals": signals,
            "actions": actions,
            "rewards": rewards,
            "consensus": consensus
        }
    
    def run_episode(self, market_data: List[bool], verbose: bool = False) -> float:
        """Run one episode"""
        total_reward = 0.0
        
        for step, price_up in enumerate(market_data):
            result = self.step(price_up)
            total_reward += sum(result["rewards"])
            
            if verbose and step % 50 == 0:
                signal = E.UP if price_up else E.DOWN
                print(f"Step {step:3d}: {signal} {result['consensus']}")
        
        return total_reward
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "agents": [
                {
                    "name": a.name,
                    "strategy": a.strategy,
                    "reward": a.state.total_reward,
                    "trades": a.state.trades,
                    "wins": a.state.wins,
                    "win_rate": a.state.wins / max(1, a.state.trades),
                    "streak": a.state.streak
                }
                for a in self.agents
            ],
            "total_steps": self.total_steps
        }
    
    def print_status(self):
        """Print system status"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"{E.BRAIN} Multi-Agent System Status")
        print(f"{'='*70}")
        
        print(f"\n{E.TARGET} Agent Performance:")
        print(f"{'Name':<12} {'Strategy':<12} {'Reward':>10} {'Trades':>8} {'Win%':>8}")
        print("-" * 50)
        
        for a in stats["agents"]:
            print(f"{a['name']:<12} {a['strategy']:<12} "
                  f"{a['reward']:>10.2f} {a['trades']:>8} "
                  f"{a['win_rate']*100:>7.1f}%")
        
        # Best agent
        best = max(stats["agents"], key=lambda x: x["reward"])
        print(f"\n{E.TROPHY} Best Agent: {best['name']} ({best['strategy']})")
        
        # Consensus summary
        print(f"\n{E.FIRE} Recent Consensus:")
        for msg in self.consensus_history[-5:]:
            print(f"  {msg}")


# ============================================================================
# TRAINING
# ============================================================================

def generate_market(n: int, regime: str = "mixed") -> List[bool]:
    """Generate market data"""
    data = []
    trend = 0.5
    
    for i in range(n):
        if regime == "bull":
            trend = 0.6
        elif regime == "bear":
            trend = 0.4
        elif regime == "volatile":
            trend = 0.5 + 0.2 * np.sin(i * 0.2)
        
        data.append(np.random.random() < trend)
    
    return data


def run_multi_agent_training():
    """Run multi-agent training"""
    print(f"{'='*70}")
    print(f"{E.ROCKET} MULTI-AGENT Q-LEARNING SYSTEM {E.ROCKET}")
    print(f"{'='*70}")
    
    system = MultiAgentSystem(n_agents=4)
    
    print(f"\n{E.BRAIN} Initialized {len(system.agents)} agents:")
    for agent in system.agents:
        print(f"  • {agent.name}: {agent.strategy} "
              f"(risk={agent.risk_tolerance:.1f}, trend={agent.trend_sensitivity:+.1f})")
    
    # Training phases
    phases = [
        ("Exploration", 30, 0.2),
        ("Learning", 60, 0.1),
        ("Coordination", 100, 0.05),
    ]
    
    for phase_name, episodes, epsilon in phases:
        print(f"\n{E.STAR} Phase: {phase_name}")
        print("-" * 50)
        
        # Set epsilon for all agents
        for agent in system.agents:
            agent.q_table.epsilon = epsilon
        
        phase_rewards = []
        regimes = ["mixed", "bull", "bear", "volatile"]
        
        for ep in range(episodes):
            regime = regimes[ep % len(regimes)]
            market = generate_market(200, regime)
            reward = system.run_episode(market, verbose=False)
            phase_rewards.append(reward)
            
            if (ep + 1) % max(1, episodes // 3) == 0:
                stats = system.get_stats()
                avg_win = np.mean([a["win_rate"] for a in stats["agents"]])
                print(f"  Ep {ep+1:3d}: Reward={np.mean(phase_rewards[-10:]):7.2f} "
                      f"Avg Win={avg_win*100:.1f}%")
        
        print(f"  {E.CHECK} Phase avg: {np.mean(phase_rewards):.2f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{E.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    for agent in system.agents:
        agent.q_table.epsilon = 0.0
    
    for regime in ["bull", "bear", "volatile", "mixed"]:
        market = generate_market(300, regime)
        reward = system.run_episode(market, verbose=False)
        print(f"  {regime:10}: Total Reward = {reward:8.2f}")
    
    system.print_status()
    
    # Detailed episode
    print(f"\n{E.FIRE} Detailed Episode:")
    market = generate_market(100, "mixed")
    system.run_episode(market, verbose=True)
    
    return system


if __name__ == "__main__":
    system = run_multi_agent_training()

