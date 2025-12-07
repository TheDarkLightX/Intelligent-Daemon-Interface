#!/usr/bin/env python3
"""
Live Trading Simulation Interface
- Real-time market simulation
- Interactive control
- Live emoji dashboard
- Performance tracking
"""

import numpy as np
import time
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum
import threading


# ============================================================================
# LIVE MARKET SIMULATOR
# ============================================================================

class MarketSimulator:
    """Generates realistic market data in real-time"""
    
    def __init__(self, initial_price: float = 100.0):
        self.price = initial_price
        self.history: List[float] = [initial_price]
        self.regime = "neutral"
        self.volatility = 0.01
        self.trend = 0.0
        
        # Regime parameters
        self.regimes = {
            "bull": {"drift": 0.0005, "vol": 0.012},
            "bear": {"drift": -0.0005, "vol": 0.012},
            "neutral": {"drift": 0.0, "vol": 0.01},
            "volatile": {"drift": 0.0, "vol": 0.025},
            "crash": {"drift": -0.002, "vol": 0.04},
            "rally": {"drift": 0.002, "vol": 0.02},
        }
        
        self.regime_counter = 0
        self.regime_duration = 50
    
    def set_regime(self, regime: str):
        """Manually set market regime"""
        if regime in self.regimes:
            self.regime = regime
            params = self.regimes[regime]
            self.trend = params["drift"]
            self.volatility = params["vol"]
    
    def tick(self) -> float:
        """Generate next price tick"""
        # Auto-switch regimes occasionally
        self.regime_counter += 1
        if self.regime_counter >= self.regime_duration:
            self.regime_counter = 0
            self.regime_duration = np.random.randint(30, 100)
            
            # Random regime switch
            if np.random.random() < 0.3:
                regimes = list(self.regimes.keys())
                self.set_regime(np.random.choice(regimes))
        
        # Generate price change
        change = self.trend + np.random.randn() * self.volatility
        self.price *= (1 + change)
        self.history.append(self.price)
        
        # Keep history bounded
        if len(self.history) > 1000:
            self.history.pop(0)
        
        return self.price
    
    def get_momentum(self, lookback: int = 10) -> float:
        """Get recent momentum"""
        if len(self.history) < lookback + 1:
            return 0.0
        return (self.history[-1] - self.history[-lookback]) / self.history[-lookback]
    
    def get_volatility(self, lookback: int = 20) -> float:
        """Get recent volatility"""
        if len(self.history) < lookback + 1:
            return self.volatility
        returns = np.diff(self.history[-lookback:]) / np.array(self.history[-lookback:-1])
        return np.std(returns)


# ============================================================================
# EMOJI DASHBOARD
# ============================================================================

class EmojiDashboard:
    """Rich emoji-based dashboard for live trading"""
    
    # Charts
    CHART_BLOCKS = " ▁▂▃▄▅▆▇█"
    
    def __init__(self, width: int = 50):
        self.width = width
    
    def price_chart(self, prices: List[float], width: int = 30) -> str:
        """Generate ASCII price chart"""
        if len(prices) < 2:
            return "No data"
        
        recent = prices[-width:]
        mn, mx = min(recent), max(recent)
        rng = mx - mn if mx != mn else 1
        
        chart = ""
        for p in recent:
            idx = int((p - mn) / rng * 8)
            chart += self.CHART_BLOCKS[min(8, max(0, idx))]
        
        return chart
    
    def format_pnl(self, pnl: float) -> str:
        """Format PnL with emoji"""
        if pnl > 0.05:
            return f"🚀 +{pnl*100:.2f}%"
        elif pnl > 0:
            return f"💰 +{pnl*100:.2f}%"
        elif pnl > -0.03:
            return f"📉 {pnl*100:.2f}%"
        else:
            return f"💀 {pnl*100:.2f}%"
    
    def regime_emoji(self, regime: str) -> str:
        """Get emoji for market regime"""
        emojis = {
            "bull": "🐂", "bear": "🐻", "neutral": "➡️",
            "volatile": "🌪️", "crash": "📉💀", "rally": "🚀"
        }
        return emojis.get(regime, "❓")
    
    def position_status(self, position: bool, entry: float, current: float) -> str:
        """Format position status"""
        if not position:
            return "💤 Flat"
        
        pnl = (current - entry) / entry if entry > 0 else 0
        if pnl > 0.02:
            return f"💰 Long +{pnl*100:.1f}%"
        elif pnl > 0:
            return f"📈 Long +{pnl*100:.1f}%"
        elif pnl > -0.02:
            return f"📉 Long {pnl*100:.1f}%"
        else:
            return f"💀 Long {pnl*100:.1f}%"
    
    def confidence_bar(self, conf: float, width: int = 10) -> str:
        """Confidence bar"""
        filled = int(conf * width)
        return "█" * filled + "░" * (width - filled)
    
    def render(self, market: MarketSimulator, agent_state: Dict) -> str:
        """Render full dashboard"""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("🤖 LIVE Q-AGENT DASHBOARD")
        lines.append("=" * 60)
        
        # Market section
        lines.append("")
        lines.append(f"📊 MARKET {self.regime_emoji(market.regime)}")
        lines.append(f"   Price: ${market.price:.2f}")
        lines.append(f"   Chart: {self.price_chart(market.history)}")
        lines.append(f"   Regime: {market.regime} | Vol: {market.get_volatility()*100:.2f}%")
        
        # Agent section
        lines.append("")
        lines.append("🧠 AGENT")
        lines.append(f"   {self.position_status(agent_state.get('position', False), agent_state.get('entry_price', 0), market.price)}")
        lines.append(f"   Total PnL: {self.format_pnl(agent_state.get('total_pnl', 0))}")
        lines.append(f"   Trades: {agent_state.get('trades', 0)} | Win: {agent_state.get('win_rate', 0)*100:.1f}%")
        lines.append(f"   Confidence: [{self.confidence_bar(agent_state.get('confidence', 0.5))}]")
        
        # Recent action
        lines.append("")
        lines.append(f"⚡ Last: {agent_state.get('last_message', 'Waiting...')}")
        
        # Controls hint
        lines.append("")
        lines.append("-" * 60)
        lines.append("Commands: [b]ull [e]ar [v]olatile [c]rash [r]ally [n]eutral [q]uit")
        
        return "\n".join(lines)


# ============================================================================
# LIVE TRADING SESSION
# ============================================================================

class LiveTradingSession:
    """Interactive live trading session"""
    
    def __init__(self, agent):
        self.agent = agent
        self.market = MarketSimulator()
        self.dashboard = EmojiDashboard()
        self.running = False
        self.tick_interval = 0.5  # seconds
        
        # State
        self.agent_state = {
            'position': False,
            'entry_price': 0.0,
            'total_pnl': 0.0,
            'trades': 0,
            'wins': 0,
            'win_rate': 0.5,
            'confidence': 0.5,
            'last_message': 'Starting...'
        }
    
    def run_tick(self):
        """Execute one market tick"""
        # Get new price
        price = self.market.tick()
        
        # Agent step
        action, reward, msg = self.agent.step(price)
        
        # Update state
        stats = self.agent.get_stats()
        self.agent_state.update({
            'position': self.agent.position,
            'entry_price': self.agent.entry_price,
            'total_pnl': stats.get('total_pnl', 0),
            'trades': stats.get('trades', 0),
            'wins': stats.get('wins', 0),
            'win_rate': stats.get('win_rate', 0.5),
            'confidence': stats.get('confidence', 0.5),
            'last_message': msg
        })
        
        return action, reward
    
    def display(self):
        """Display dashboard"""
        # Clear screen (cross-platform)
        print("\033[H\033[J", end="")
        print(self.dashboard.render(self.market, self.agent_state))
    
    def run(self, ticks: int = 100, interactive: bool = False):
        """Run trading session"""
        print("🚀 Starting Live Trading Session...")
        print(f"   Running for {ticks} ticks")
        print()
        
        self.running = True
        
        for i in range(ticks):
            if not self.running:
                break
            
            # Run tick
            action, reward = self.run_tick()
            
            # Display every 5 ticks or on action
            if i % 5 == 0 or action in [1, 2]:
                self.display()
            
            # Simulate real-time delay
            if interactive:
                time.sleep(self.tick_interval)
        
        # Final display
        self.display()
        print("\n✅ Session complete!")
        
        return self.agent_state


def run_live_demo():
    """Run live trading demo"""
    from master_agent import MasterAgent
    
    print("=" * 60)
    print("🤖 LIVE Q-AGENT DEMO")
    print("=" * 60)
    
    # Create and train agent
    print("\n📚 Pre-training agent...")
    agent = MasterAgent(n_states=300)
    agent.ensemble.epsilon = 0.0  # No exploration in live mode
    
    # Quick training
    from master_agent import gen_prices
    for _ in range(30):
        prices = gen_prices(200, "mixed")
        agent.run_episode(prices)
    
    stats = agent.get_stats()
    print(f"   Trained: {stats['trades']} trades, {stats['win_rate']*100:.1f}% win rate")
    
    # Run live session
    print("\n🔴 GOING LIVE...")
    time.sleep(1)
    
    session = LiveTradingSession(agent)
    final_state = session.run(ticks=200, interactive=False)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Total PnL: {final_state['total_pnl']*100:+.2f}%")
    print(f"   Trades: {final_state['trades']}")
    print(f"   Win Rate: {final_state['win_rate']*100:.1f}%")


if __name__ == "__main__":
    run_live_demo()

