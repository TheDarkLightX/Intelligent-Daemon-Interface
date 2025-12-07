#!/usr/bin/env python3
"""
Intelligence Benchmark - Compare Q-Learning Strategies
Tests different Q-table configurations and measures "smartness"
"""

import os
import json
import time
import numpy as np
import sys
sys.path.insert(0, '../phase5_python')
from q_daemon import QTable, LayeredQTable, Action, State, MarketState, generate_market_data


def get_seed() -> int:
    seed_str = os.getenv("SEED")
    try:
        return int(seed_str) if seed_str else 42
    except ValueError:
        return 42


class IntelligenceBenchmark:
    """Benchmark framework for Q-learning agents"""
    
    def __init__(self):
        self.results = {}
    
    def run_strategy(self, name: str, agent, market_data: list, 
                    n_episodes: int = 100) -> dict:
        """Run a strategy through multiple episodes"""
        rewards = []
        trades = []
        wins = []
        
        for _ in range(n_episodes):
            ep_reward = 0
            ep_trades = 0
            ep_wins = 0
            position = False
            
            for price_up, price_down in market_data:
                state = MarketState(price_up, price_down, position)
                state_idx = state.to_state_index()
                
                # Get action
                if hasattr(agent, 'select_action'):
                    if isinstance(agent, LayeredQTable):
                        action, _ = agent.select_action(state_idx)
                    else:
                        action = agent.select_action(state_idx, explore=False)
                else:
                    action = agent(state_idx, price_up, price_down, position)
                
                # Validate
                if action == Action.BUY and position:
                    action = Action.HOLD
                elif action == Action.SELL and not position:
                    action = Action.HOLD
                
                # Execute
                if action == Action.BUY:
                    position = True
                    ep_reward -= 0.1
                elif action == Action.SELL:
                    position = False
                    ep_trades += 1
                    ep_wins += 1
                    ep_reward += 1.0
                else:
                    ep_reward += 0.01
            
            rewards.append(ep_reward)
            trades.append(ep_trades)
            wins.append(ep_wins)
        
        return {
            "name": name,
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_trades": np.mean(trades),
            "win_rate": np.sum(wins) / max(1, np.sum(trades)),
            "total_reward": np.sum(rewards),
            "rewards": rewards
        }
    
    def random_strategy(self, state_idx, price_up, price_down, position):
        """Random baseline"""
        return Action(np.random.randint(4))
    
    def buy_hold_strategy(self, state_idx, price_up, price_down, position):
        """Always buy and hold"""
        if not position:
            return Action.BUY
        return Action.HOLD
    
    def momentum_strategy(self, state_idx, price_up, price_down, position):
        """Buy on up, sell on down"""
        if price_up and not position:
            return Action.BUY
        elif price_down and position:
            return Action.SELL
        return Action.HOLD
    
    def contrarian_strategy(self, state_idx, price_up, price_down, position):
        """Buy on down, sell on up"""
        if price_down and not position:
            return Action.BUY
        elif price_up and position:
            return Action.SELL
        return Action.HOLD
    
    def run_benchmark(self, n_episodes: int = 100, market_steps: int = 50):
        """Run full benchmark comparing strategies"""
        seed = get_seed()
        np.random.seed(seed)
        print("=" * 70)
        print("🧠 Intelligence Benchmark: Q-Learning Strategy Comparison")
        print("=" * 70)
        print(f"Episodes: {n_episodes}, Market steps: {market_steps}, Seed: {seed}")
        print()
        
        # Generate consistent market data for fair comparison
        market_data = generate_market_data(market_steps, 0.6)
        
        strategies = [
            ("Random", self.random_strategy),
            ("Buy & Hold", self.buy_hold_strategy),
            ("Momentum", self.momentum_strategy),
            ("Contrarian", self.contrarian_strategy),
            ("Q-Table (Simple)", QTable()),
            ("Q-Table (Layered)", LayeredQTable()),
        ]
        
        results = []
        for name, agent in strategies:
            print(f"Testing: {name}...", end=" ", flush=True)
            start = time.time()
            result = self.run_strategy(name, agent, market_data, n_episodes)
            elapsed = time.time() - start
            result["time"] = elapsed
            results.append(result)
            print(f"done ({elapsed:.2f}s)")
        
        # Print results
        print("\n" + "=" * 70)
        print("📊 Results")
        print("=" * 70)
        print(f"{'Strategy':<20} | {'Avg Reward':>10} | {'Std':>6} | {'Trades':>6} | {'Win%':>5} | {'Time':>6}")
        print("-" * 70)
        
        for r in sorted(results, key=lambda x: x["avg_reward"], reverse=True):
            print(f"{r['name']:<20} | {r['avg_reward']:>10.2f} | {r['std_reward']:>6.2f} | "
                  f"{r['avg_trades']:>6.1f} | {r['win_rate']*100:>5.1f} | {r['time']:>5.2f}s")
        
        # Intelligence ranking
        print("\n" + "=" * 70)
        print("🏆 Intelligence Ranking")
        print("=" * 70)
        
        ranked = sorted(results, key=lambda x: x["avg_reward"], reverse=True)
        for i, r in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            improvement = (r["avg_reward"] - ranked[-1]["avg_reward"]) / max(0.01, abs(ranked[-1]["avg_reward"])) * 100
            print(f"{medal} {r['name']:<20} - Reward: {r['avg_reward']:.2f} "
                  f"(+{improvement:.0f}% vs worst)")
        
        # Persist results to JSON
        out = {
            "seed": seed,
            "episodes": n_episodes,
            "market_steps": market_steps,
            "results": results,
        }
        with open("benchmark_results.json", "w") as f:
            json.dump(out, f, indent=2)
        # Also write CSV summary
        with open("benchmark_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "avg_reward", "std_reward", "avg_trades", "win_rate", "total_reward", "time"])
            for r in sorted(results, key=lambda x: x["avg_reward"], reverse=True):
                writer.writerow([
                    r["name"],
                    f"{r['avg_reward']:.4f}",
                    f"{r['std_reward']:.4f}",
                    f"{r['avg_trades']:.2f}",
                    f"{r['win_rate']:.4f}",
                    f"{r['total_reward']:.4f}",
                    f"{r['time']:.4f}",
                ])
        return results


class AdaptiveLearningTest:
    """Test how quickly agents adapt to market regime changes"""
    
    def __init__(self):
        pass
    
    def run(self, n_steps: int = 200):
        print("\n" + "=" * 70)
        print("🔄 Adaptive Learning Test")
        print("=" * 70)
        
        # Create agent
        q = LayeredQTable()
        
        # Phase 1: Trending market (high trend prob)
        print("\nPhase 1: Trending Market (steps 0-99)")
        market1 = generate_market_data(100, 0.8)  # Strong trends
        
        rewards1 = []
        position = False
        for i, (up, down) in enumerate(market1):
            state = MarketState(up, down, position)
            action, _ = q.select_action(state.to_state_index())
            
            if action == Action.BUY and not position:
                position = True
                rewards1.append(-0.1)
            elif action == Action.SELL and position:
                position = False
                rewards1.append(1.0)
                q.update_all(state.to_state_index(), action, 1.0, 
                            MarketState(up, down, position).to_state_index())
            else:
                rewards1.append(0.01)
        
        print(f"  Total Reward: {sum(rewards1):.2f}")
        print(f"  Avg per step: {np.mean(rewards1):.3f}")
        
        # Phase 2: Mean-reverting market (low trend prob)
        print("\nPhase 2: Mean-Reverting Market (steps 100-199)")
        market2 = generate_market_data(100, 0.3)  # Choppy
        
        rewards2 = []
        for i, (up, down) in enumerate(market2):
            state = MarketState(up, down, position)
            action, _ = q.select_action(state.to_state_index())
            
            if action == Action.BUY and not position:
                position = True
                rewards2.append(-0.1)
            elif action == Action.SELL and position:
                position = False
                rewards2.append(1.0)
                q.update_all(state.to_state_index(), action, 1.0,
                            MarketState(up, down, position).to_state_index())
            else:
                rewards2.append(0.01)
        
        print(f"  Total Reward: {sum(rewards2):.2f}")
        print(f"  Avg per step: {np.mean(rewards2):.3f}")
        
        # Adaptation score
        adapt_score = np.mean(rewards2[-50:]) / max(0.001, np.mean(rewards2[:50]))
        print(f"\n📈 Adaptation Score: {adapt_score:.2f}x")
        print("  (>1.0 = improved in regime 2, <1.0 = degraded)")
        
        return {"phase1": rewards1, "phase2": rewards2, "adapt_score": adapt_score}


def main():
    print("🤖 Tau Q-Agent Intelligence Testing Framework")
    print("=" * 70)
    
    # Run benchmark
    benchmark = IntelligenceBenchmark()
    benchmark.run_benchmark(n_episodes=50, market_steps=100)
    
    # Run adaptation test
    adapt = AdaptiveLearningTest()
    adapt.run()
    
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

