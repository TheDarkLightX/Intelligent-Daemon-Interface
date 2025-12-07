#!/usr/bin/env python3
"""
Iterative Improvement System
- Multiple training iterations
- Performance measurement
- Automatic hyperparameter adjustment
- Emotion calibration
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
from full_training_pipeline import MegaQAgent, generate_market, Emoji


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    
    # Returns
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Trading
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Efficiency
    trades_per_episode: float = 0.0
    reward_per_trade: float = 0.0
    
    # Learning
    state_coverage: float = 0.0
    q_convergence: float = 0.0
    
    def compute_score(self) -> float:
        """Compute overall score"""
        return (
            self.sharpe_ratio * 0.3 +
            self.win_rate * 0.2 +
            (1 + self.max_drawdown) * 0.2 +  # max_drawdown is negative
            self.profit_factor * 0.2 +
            self.state_coverage * 0.1
        )
    
    def print(self):
        """Print metrics"""
        print(f"\n{Emoji.CHART_UP} Performance Metrics:")
        print(f"  Total Return:    {self.total_return:8.2f}")
        print(f"  Sharpe Ratio:    {self.sharpe_ratio:8.2f}")
        print(f"  Max Drawdown:    {self.max_drawdown*100:7.1f}%")
        print(f"  Win Rate:        {self.win_rate*100:7.1f}%")
        print(f"  Profit Factor:   {self.profit_factor:8.2f}")
        print(f"  Trades/Episode:  {self.trades_per_episode:8.1f}")
        print(f"  Reward/Trade:    {self.reward_per_trade:8.3f}")
        print(f"  Coverage:        {self.state_coverage:7.2f}%")
        print(f"  {Emoji.STAR} Score: {self.compute_score():.3f}")


def compute_metrics(agent: MegaQAgent) -> PerformanceMetrics:
    """Compute comprehensive metrics"""
    stats = agent.get_stats()
    
    # Sharpe ratio
    if len(agent.episode_rewards) > 1:
        returns = np.array(agent.episode_rewards)
        sharpe = np.mean(returns) / (np.std(returns) + 0.01) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    equity = np.array(agent.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = np.min(drawdowns)
    
    # Profit factor (simplified)
    wins = stats["wins"]
    losses = stats["trades"] - wins
    profit_factor = (wins + 0.1) / (losses + 0.1)
    
    return PerformanceMetrics(
        total_return=stats["total_reward"],
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=stats["win_rate"],
        profit_factor=profit_factor,
        avg_win=stats["total_reward"] / max(1, wins),
        avg_loss=0,  # Simplified
        trades_per_episode=stats["trades"] / max(1, stats["episodes"]),
        reward_per_trade=stats["total_reward"] / max(1, stats["trades"]),
        state_coverage=stats["coverage"],
        q_convergence=0.0
    )


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

class AdaptiveHyperparams:
    """Adaptive hyperparameter adjustment based on performance"""
    
    def __init__(self):
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.risk_multiplier = 1.0
        
        self.history: List[Tuple[Dict, float]] = []
    
    def adjust(self, metrics: PerformanceMetrics):
        """Adjust hyperparameters based on metrics"""
        score = metrics.compute_score()
        
        # Store history
        params = {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "risk_multiplier": self.risk_multiplier
        }
        self.history.append((params.copy(), score))
        
        # Adjust based on performance
        if metrics.win_rate < 0.4:
            # Need more exploration
            self.epsilon = min(0.4, self.epsilon * 1.1)
            self.lr = min(0.3, self.lr * 1.05)
        elif metrics.win_rate > 0.6:
            # Exploit more
            self.epsilon = max(0.05, self.epsilon * 0.9)
        
        if metrics.max_drawdown < -0.3:
            # Too risky
            self.risk_multiplier *= 0.9
        elif metrics.max_drawdown > -0.1:
            # Can take more risk
            self.risk_multiplier = min(2.0, self.risk_multiplier * 1.1)
        
        if metrics.state_coverage < 1.0:
            # Need more exploration
            self.epsilon = min(0.5, self.epsilon * 1.1)
        
        return params
    
    def get_best(self) -> Dict:
        """Get best hyperparameters from history"""
        if not self.history:
            return {"lr": self.lr, "gamma": self.gamma, 
                   "epsilon": self.epsilon, "risk_multiplier": self.risk_multiplier}
        
        best = max(self.history, key=lambda x: x[1])
        return best[0]


# ============================================================================
# EMOTION CALIBRATION
# ============================================================================

def calibrate_emotions(agent: MegaQAgent, target_confidence: float = 0.5):
    """Calibrate emotional responses"""
    
    # Reset emotions to balanced state
    agent.emotions.fear = 0.3
    agent.emotions.greed = 0.3
    agent.emotions.confidence = target_confidence
    agent.emotions.patience = 0.5
    
    # Adjust based on recent performance
    if len(agent.episode_rewards) > 10:
        recent = agent.episode_rewards[-10:]
        trend = np.polyfit(range(10), recent, 1)[0]
        
        if trend > 0:
            # Improving - increase confidence
            agent.emotions.confidence = min(0.8, agent.emotions.confidence + 0.1)
            agent.emotions.greed = min(0.5, agent.emotions.greed + 0.05)
        else:
            # Declining - increase caution
            agent.emotions.fear = min(0.6, agent.emotions.fear + 0.1)
            agent.emotions.patience = min(0.8, agent.emotions.patience + 0.1)
    
    return agent


# ============================================================================
# ITERATIVE TRAINING
# ============================================================================

def run_iteration(agent: MegaQAgent, n_episodes: int, 
                 hyperparams: AdaptiveHyperparams) -> PerformanceMetrics:
    """Run one training iteration"""
    
    # Apply hyperparameters
    agent.q_table.lr = hyperparams.lr
    agent.q_table.gamma = hyperparams.gamma
    agent.q_table.epsilon = hyperparams.epsilon
    
    # Training
    regimes = ["mixed", "bull", "bear", "volatile"]
    for ep in range(n_episodes):
        regime = regimes[ep % len(regimes)]
        market = generate_market(200, regime)
        agent.run_episode(market, verbose=False)
    
    return compute_metrics(agent)


def iterative_improvement(n_iterations: int = 5, episodes_per_iter: int = 50):
    """Main iterative improvement loop"""
    
    print(f"{'='*70}")
    print(f"{Emoji.ROCKET} ITERATIVE IMPROVEMENT SYSTEM {Emoji.ROCKET}")
    print(f"{'='*70}")
    print(f"Iterations: {n_iterations}, Episodes/iter: {episodes_per_iter}")
    
    # Initialize
    config = {
        "features": {
            "momentum": 5,
            "momentum_timeframes": 2,
            "volatility": 4,
            "position": 4,
            "duration": 4,
            "streak": 5,
            "regime": 3
        }
    }
    
    agent = MegaQAgent(config)
    hyperparams = AdaptiveHyperparams()
    
    iteration_results: List[PerformanceMetrics] = []
    
    for iteration in range(n_iterations):
        print(f"\n{Emoji.STAR} Iteration {iteration + 1}/{n_iterations}")
        print("-" * 50)
        
        # Run training
        start = time.time()
        metrics = run_iteration(agent, episodes_per_iter, hyperparams)
        elapsed = time.time() - start
        
        iteration_results.append(metrics)
        
        # Print progress
        print(f"  {Emoji.CHECK} Complete in {elapsed:.1f}s")
        print(f"  Score: {metrics.compute_score():.3f} | "
              f"Win: {metrics.win_rate*100:.1f}% | "
              f"Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"  {agent.emotions.get_emoji()} "
              f"Fear={agent.emotions.fear:.2f} "
              f"Greed={agent.emotions.greed:.2f} "
              f"Conf={agent.emotions.confidence:.2f}")
        
        # Adjust hyperparameters
        old_params = hyperparams.adjust(metrics)
        print(f"  {Emoji.GEAR} Adjusted: ε={hyperparams.epsilon:.3f} "
              f"lr={hyperparams.lr:.3f} risk={hyperparams.risk_multiplier:.2f}")
        
        # Calibrate emotions periodically
        if (iteration + 1) % 2 == 0:
            calibrate_emotions(agent)
            print(f"  {Emoji.BRAIN} Emotions calibrated")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{Emoji.TROPHY} FINAL RESULTS")
    print(f"{'='*70}")
    
    # Best iteration
    best_idx = np.argmax([m.compute_score() for m in iteration_results])
    print(f"\n{Emoji.MEDAL_GOLD} Best Iteration: {best_idx + 1}")
    iteration_results[best_idx].print()
    
    # Improvement tracking
    print(f"\n{Emoji.PROGRESS} Improvement Over Iterations:")
    scores = [m.compute_score() for m in iteration_results]
    for i, score in enumerate(scores):
        bar = "█" * int(score * 20)
        improvement = ((score - scores[0]) / max(0.01, scores[0])) * 100 if i > 0 else 0
        print(f"  Iter {i+1}: {bar} {score:.3f} ({improvement:+.1f}%)")
    
    # Best hyperparameters
    best_params = hyperparams.get_best()
    print(f"\n{Emoji.GEAR} Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    
    # Agent status
    agent.print_status()
    
    return agent, iteration_results


# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================

def benchmark_comparison():
    """Compare different agent configurations"""
    
    print(f"\n{'='*70}")
    print(f"{Emoji.TROPHY} AGENT CONFIGURATION BENCHMARK")
    print(f"{'='*70}")
    
    configs = [
        ("Small (10K)", {"momentum": 3, "momentum_timeframes": 2, 
                        "volatility": 3, "position": 3, "duration": 3, 
                        "streak": 3, "regime": 3}),
        ("Medium (100K)", {"momentum": 5, "momentum_timeframes": 2,
                          "volatility": 4, "position": 4, "duration": 4,
                          "streak": 5, "regime": 3}),
        ("Large (500K)", {"momentum": 7, "momentum_timeframes": 3,
                         "volatility": 5, "position": 4, "duration": 5,
                         "streak": 7, "regime": 3}),
    ]
    
    results = []
    
    for name, feature_config in configs:
        print(f"\n{Emoji.STAR} Testing: {name}")
        
        agent = MegaQAgent({"features": feature_config})
        print(f"  States: {agent.features.n_states:,}")
        
        # Quick training
        for _ in range(30):
            market = generate_market(150, "mixed")
            agent.run_episode(market)
        
        metrics = compute_metrics(agent)
        results.append((name, metrics))
        
        print(f"  Score: {metrics.compute_score():.3f} | "
              f"Win: {metrics.win_rate*100:.1f}% | "
              f"Coverage: {metrics.state_coverage:.2f}%")
    
    # Ranking
    print(f"\n{Emoji.MEDAL_GOLD} Ranking:")
    ranked = sorted(results, key=lambda x: x[1].compute_score(), reverse=True)
    medals = [Emoji.MEDAL_GOLD, Emoji.MEDAL_SILVER, Emoji.MEDAL_BRONZE]
    
    for i, (name, metrics) in enumerate(ranked):
        medal = medals[i] if i < 3 else f"{i+1}."
        print(f"  {medal} {name}: {metrics.compute_score():.3f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"{'='*70}")
    print(f"{Emoji.BRAIN} Q-AGENT ITERATIVE IMPROVEMENT SYSTEM")
    print(f"{'='*70}")
    
    # Run iterative improvement
    agent, results = iterative_improvement(n_iterations=5, episodes_per_iter=40)
    
    # Run benchmark
    benchmark_comparison()
    
    # Final message
    print(f"\n{'='*70}")
    print(f"{Emoji.CHECK} ALL COMPLETE!")
    print(f"{'='*70}")
    
    # Show communication samples
    print(f"\n{Emoji.LIGHTNING} Agent Communication Samples:")
    for msg in agent.messages[-15:]:
        print(f"  {msg}")


if __name__ == "__main__":
    main()

