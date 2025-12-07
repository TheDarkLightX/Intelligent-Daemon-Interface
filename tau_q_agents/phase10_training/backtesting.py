#!/usr/bin/env python3
"""
Backtesting Framework
- Historical data simulation
- Walk-forward optimization
- Performance metrics
- Statistical analysis
- Comparison benchmarks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    time_in_market: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_return': f"{self.total_return*100:.2f}%",
            'annual_return': f"{self.annual_return*100:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown*100:.2f}%",
            'win_rate': f"{self.win_rate*100:.1f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'avg_trade': f"{self.avg_trade*100:.3f}%",
            'avg_winner': f"{self.avg_winner*100:.3f}%",
            'avg_loser': f"{self.avg_loser*100:.3f}%",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'time_in_market': f"{self.time_in_market*100:.1f}%"
        }
    
    def print_report(self):
        """Print formatted report"""
        print("\n" + "=" * 50)
        print("📊 BACKTEST REPORT")
        print("=" * 50)
        
        # Returns
        ret_emoji = "🚀" if self.total_return > 0.1 else ("💰" if self.total_return > 0 else "📉")
        print(f"\n{ret_emoji} RETURNS")
        print(f"   Total Return: {self.total_return*100:+.2f}%")
        print(f"   Annual Return: {self.annual_return*100:+.2f}%")
        
        # Risk metrics
        risk_emoji = "✅" if self.sharpe_ratio > 1 else ("⚠️" if self.sharpe_ratio > 0 else "❌")
        print(f"\n{risk_emoji} RISK METRICS")
        print(f"   Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"   Max Drawdown: {self.max_drawdown*100:.2f}%")
        
        # Trading stats
        trade_emoji = "✅" if self.win_rate > 0.5 else "❌"
        print(f"\n{trade_emoji} TRADING STATS")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Win Rate: {self.win_rate*100:.1f}%")
        print(f"   Profit Factor: {self.profit_factor:.2f}")
        print(f"   Avg Winner: {self.avg_winner*100:+.3f}%")
        print(f"   Avg Loser: {self.avg_loser*100:+.3f}%")
        
        # Streaks
        print(f"\n📈 STREAKS")
        print(f"   Max Consecutive Wins: {self.max_consecutive_wins}")
        print(f"   Max Consecutive Losses: {self.max_consecutive_losses}")
        print(f"   Time in Market: {self.time_in_market*100:.1f}%")


def calculate_metrics(trades: List[float], prices: List[float], 
                     positions: List[bool], periods_per_year: int = 252) -> BacktestMetrics:
    """Calculate backtest metrics from trade history"""
    metrics = BacktestMetrics()
    
    if not trades:
        return metrics
    
    trades = np.array(trades)
    
    # Basic stats
    metrics.total_trades = len(trades)
    metrics.winning_trades = np.sum(trades > 0)
    metrics.losing_trades = np.sum(trades < 0)
    metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
    
    # Returns
    metrics.total_return = np.sum(trades)
    metrics.annual_return = metrics.total_return * (periods_per_year / len(prices)) if prices else 0
    
    # Average trade
    metrics.avg_trade = np.mean(trades)
    winners = trades[trades > 0]
    losers = trades[trades < 0]
    metrics.avg_winner = np.mean(winners) if len(winners) > 0 else 0
    metrics.avg_loser = np.mean(losers) if len(losers) > 0 else 0
    
    # Profit factor
    gross_profit = np.sum(winners) if len(winners) > 0 else 0
    gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 0.001
    metrics.profit_factor = gross_profit / gross_loss
    
    # Sharpe and Sortino
    if len(trades) > 1:
        std_returns = np.std(trades)
        downside_std = np.std(trades[trades < 0]) if np.sum(trades < 0) > 1 else std_returns
        
        metrics.sharpe_ratio = (metrics.avg_trade / std_returns * np.sqrt(periods_per_year)) if std_returns > 0 else 0
        metrics.sortino_ratio = (metrics.avg_trade / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0
    
    # Max drawdown
    cumulative = np.cumsum(trades)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / (peak + 1)
    metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Consecutive wins/losses
    if len(trades) > 0:
        signs = np.sign(trades)
        metrics.max_consecutive_wins = max_consecutive(signs, 1)
        metrics.max_consecutive_losses = max_consecutive(signs, -1)
    
    # Time in market
    metrics.time_in_market = np.mean(positions) if positions else 0
    
    return metrics


def max_consecutive(arr: np.ndarray, target: int) -> int:
    """Find max consecutive occurrences of target"""
    max_count = 0
    current = 0
    for v in arr:
        if v == target:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# ============================================================================
# BACKTESTER
# ============================================================================

class Backtester:
    """Run backtests on Q-agents"""
    
    def __init__(self, agent_factory: Callable):
        """
        Args:
            agent_factory: Function that creates a fresh agent
        """
        self.agent_factory = agent_factory
        self.results: List[Dict] = []
    
    def run(self, prices: List[float], train_ratio: float = 0.6,
           name: str = "backtest") -> BacktestMetrics:
        """Run single backtest with train/test split"""
        # Split data
        split_idx = int(len(prices) * train_ratio)
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
        
        # Create and train agent
        agent = self.agent_factory()
        
        print(f"📚 Training on {len(train_prices)} bars...")
        _ = agent.run_episode(train_prices)
        
        # Reset for test - keep Q-tables but reset state
        agent.position = False
        agent.entry_price = 0.0
        agent.total_pnl = 0.0
        agent.trades = 0
        agent.wins = 0
        agent.hold_duration = 0
        agent.streak = 0
        
        # Reset analyzer to fresh state
        if hasattr(agent, 'analyzer'):
            from collections import deque
            agent.analyzer.prices = deque(maxlen=100)
            agent.analyzer.returns = deque(maxlen=100)
            agent.analyzer.vol_history = deque(maxlen=50)
        
        # Lower exploration for testing (but not zero - agent needs to act)
        if hasattr(agent, 'ensemble'):
            agent.ensemble.epsilon = 0.05  # Small exploration
        elif hasattr(agent, 'q_table'):
            agent.q_table.epsilon = 0.05
        
        print(f"📊 Testing on {len(test_prices)} bars...")
        
        # Track detailed results
        trade_pnls = []
        positions = []
        entry_price = 0.0
        in_position = False
        
        for price in test_prices:
            action, reward, msg = agent.step(price)
            positions.append(in_position)
            
            if action == 1 and not in_position:  # BUY
                in_position = True
                entry_price = price
            elif action == 2 and in_position:  # SELL
                pnl = (price - entry_price) / entry_price
                trade_pnls.append(pnl)
                in_position = False
        
        # Calculate metrics
        metrics = calculate_metrics(trade_pnls, test_prices, positions)
        
        # Store result
        self.results.append({
            'name': name,
            'metrics': metrics.to_dict(),
            'trade_count': len(trade_pnls)
        })
        
        return metrics
    
    def walk_forward(self, prices: List[float], window_size: int = 500,
                    step_size: int = 100, name: str = "walk_forward") -> List[BacktestMetrics]:
        """Walk-forward optimization"""
        print(f"🚶 Walk-Forward Analysis")
        print(f"   Window: {window_size}, Step: {step_size}")
        
        all_metrics = []
        
        start = 0
        while start + window_size < len(prices):
            window = prices[start:start + window_size]
            
            # Run backtest on window
            metrics = self.run(window, train_ratio=0.7, 
                             name=f"{name}_window_{start}")
            all_metrics.append(metrics)
            
            print(f"   Window {start}-{start+window_size}: "
                  f"Return {metrics.total_return*100:+.2f}%, "
                  f"Sharpe {metrics.sharpe_ratio:.2f}")
            
            start += step_size
        
        # Aggregate metrics
        print(f"\n📊 Walk-Forward Summary ({len(all_metrics)} windows):")
        avg_return = np.mean([m.total_return for m in all_metrics])
        avg_sharpe = np.mean([m.sharpe_ratio for m in all_metrics])
        avg_win_rate = np.mean([m.win_rate for m in all_metrics])
        
        print(f"   Avg Return: {avg_return*100:+.2f}%")
        print(f"   Avg Sharpe: {avg_sharpe:.2f}")
        print(f"   Avg Win Rate: {avg_win_rate*100:.1f}%")
        
        return all_metrics
    
    def compare_strategies(self, prices: List[float], 
                          strategies: Dict[str, Callable]) -> Dict[str, BacktestMetrics]:
        """Compare multiple strategies"""
        print("⚔️ Strategy Comparison")
        print("=" * 50)
        
        results = {}
        
        for name, factory in strategies.items():
            print(f"\n🎯 Testing: {name}")
            self.agent_factory = factory
            metrics = self.run(prices, name=name)
            results[name] = metrics
        
        # Print comparison table
        print("\n" + "=" * 70)
        print("📊 COMPARISON TABLE")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'Win%':>8} {'Trades':>8}")
        print("-" * 70)
        
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1].total_return, reverse=True)
        
        for name, m in sorted_results:
            emoji = "🥇" if name == sorted_results[0][0] else "  "
            print(f"{emoji}{name:<18} {m.total_return*100:>+9.2f}% "
                  f"{m.sharpe_ratio:>8.2f} {m.win_rate*100:>7.1f}% {m.total_trades:>8}")
        
        return results


# ============================================================================
# BENCHMARK STRATEGIES
# ============================================================================

class BuyAndHold:
    """Simple buy and hold benchmark"""
    
    def __init__(self, n_states: int = 100):
        self.position = False
        self.entry_price = 0.0
        self.bought = False
    
    def step(self, price: float) -> Tuple[int, float, str]:
        if not self.bought:
            self.bought = True
            self.position = True
            self.entry_price = price
            return 1, 0, "🟢 BUY"
        return 0, 0, "⏸️ HOLD"
    
    def run_episode(self, prices: List[float]):
        for p in prices:
            self.step(p)
    
    def get_stats(self):
        return {'total_pnl': 0, 'trades': 1, 'wins': 0, 'win_rate': 0}


class RandomTrader:
    """Random trading benchmark"""
    
    def __init__(self, n_states: int = 100, trade_prob: float = 0.05):
        self.position = False
        self.entry_price = 0.0
        self.trade_prob = trade_prob
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
    
    def step(self, price: float) -> Tuple[int, float, str]:
        action = 0
        
        if np.random.random() < self.trade_prob:
            if not self.position:
                self.position = True
                self.entry_price = price
                action = 1
            else:
                pnl = (price - self.entry_price) / self.entry_price
                self.total_pnl += pnl
                self.trades += 1
                if pnl > 0:
                    self.wins += 1
                self.position = False
                action = 2
        
        return action, 0, ["⏸️", "🟢", "🔴"][action]
    
    def run_episode(self, prices: List[float]):
        for p in prices:
            self.step(p)
    
    def get_stats(self):
        return {
            'total_pnl': self.total_pnl,
            'trades': self.trades,
            'wins': self.wins,
            'win_rate': self.wins / max(1, self.trades)
        }


# ============================================================================
# DEMO
# ============================================================================

def run_backtest_demo():
    """Run backtesting demo"""
    from master_agent import MasterAgent, gen_prices
    
    print("=" * 60)
    print("📊 BACKTESTING FRAMEWORK DEMO")
    print("=" * 60)
    
    # Generate test data
    print("\n📈 Generating price data...")
    prices = gen_prices(2000, "mixed")
    print(f"   {len(prices)} price bars generated")
    
    # Define strategies
    strategies = {
        "MasterAgent": lambda: MasterAgent(n_states=300),
        "BuyAndHold": lambda: BuyAndHold(),
        "RandomTrader": lambda: RandomTrader(),
    }
    
    # Run comparison
    backtester = Backtester(lambda: MasterAgent(n_states=300))
    results = backtester.compare_strategies(prices, strategies)
    
    # Detailed report for best strategy
    best_name = max(results, key=lambda x: results[x].total_return)
    print(f"\n🏆 Best Strategy: {best_name}")
    results[best_name].print_report()
    
    # Walk-forward analysis on master agent
    print("\n" + "=" * 60)
    backtester.agent_factory = lambda: MasterAgent(n_states=300)
    wf_results = backtester.walk_forward(prices, window_size=400, step_size=100)
    
    return results


if __name__ == "__main__":
    results = run_backtest_demo()

