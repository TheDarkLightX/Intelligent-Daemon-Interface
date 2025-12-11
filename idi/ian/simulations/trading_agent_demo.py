#!/usr/bin/env python3
"""
IAN End-to-End Demo: Trading Agent Competition (Simulated Evaluation)

This demo shows the complete IAN workflow over a simulated trading task:
1. Define a goal (trading performance)
2. Create the coordinator with security hardening
3. Submit multiple agent contributions
4. View leaderboard and active policy
5. Simulate Tau Net integration

Run with: python -m idi.ian.examples.trading_agent_demo
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

# IAN imports
from idi.ian import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    Metrics,
    EvaluationLimits,
    Thresholds,
    IANCoordinator,
    CoordinatorConfig,
    SecureCoordinator,
    SecurityLimits,
    TauBridge,
    TauBridgeConfig,
    TauIntegratedCoordinator,
)


# =============================================================================
# Demo Configuration
# =============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    goal_id: str = "TRADING_COMPETITION_2025"
    num_contributors: int = 5
    contributions_per_contributor: int = 3
    leaderboard_size: int = 10
    enable_security: bool = True
    enable_tau: bool = True
    verbose: bool = True


# =============================================================================
# Simulated Trading Agent
# =============================================================================

@dataclass
class TradingStrategy:
    """A simulated trading strategy."""
    name: str
    parameters: dict
    
    def to_agent_pack(self) -> AgentPack:
        """Convert to AgentPack."""
        param_bytes = json.dumps(self.parameters).encode('utf-8')
        return AgentPack(
            version="1.0.0",
            parameters=param_bytes,
            metadata={
                "strategy_name": self.name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def generate_trading_strategy(strategy_type: str, seed: int) -> TradingStrategy:
    """Generate a random trading strategy."""
    random.seed(seed)
    
    strategies = {
        "momentum": {
            "lookback_period": random.randint(5, 50),
            "threshold": random.uniform(0.01, 0.1),
            "position_size": random.uniform(0.1, 0.5),
        },
        "mean_reversion": {
            "window": random.randint(10, 100),
            "std_multiplier": random.uniform(1.0, 3.0),
            "entry_threshold": random.uniform(0.5, 2.0),
        },
        "breakout": {
            "range_period": random.randint(10, 30),
            "breakout_threshold": random.uniform(0.01, 0.05),
            "stop_loss": random.uniform(0.01, 0.03),
        },
        "neural": {
            "hidden_layers": random.randint(2, 5),
            "neurons_per_layer": random.randint(32, 256),
            "learning_rate": random.uniform(0.0001, 0.01),
        },
    }
    
    return TradingStrategy(
        name=f"{strategy_type}_{seed}",
        parameters=strategies.get(strategy_type, strategies["momentum"]),
    )


# =============================================================================
# Simulated Evaluation
# =============================================================================

class TradingEvaluator:
    """
    Simulates trading strategy evaluation.
    
    In a real system, this would:
    - Run the strategy on historical data
    - Compute actual PnL, Sharpe ratio, etc.
    - Use sandboxed execution
    
    For this demo, we simulate with random-but-consistent results.
    """
    
    def __init__(self, goal_spec: GoalSpec):
        self._goal_spec = goal_spec
    
    def evaluate(self, agent_pack: AgentPack, seed: int) -> Metrics:
        """Evaluate a trading strategy."""
        # Use pack hash + seed for deterministic "random" results
        pack_hash = hashlib.sha256(agent_pack.parameters).digest()
        eval_seed = int.from_bytes(pack_hash[:4], 'big') ^ seed
        random.seed(eval_seed)
        
        # Simulate evaluation metrics
        # Better strategies have higher reward, lower risk, lower complexity
        base_reward = random.gauss(0.5, 0.2)
        base_risk = random.gauss(0.3, 0.1)
        base_complexity = random.gauss(0.4, 0.15)
        
        # Clamp to valid ranges
        reward = max(0.0, min(1.0, base_reward))
        risk = max(0.0, min(1.0, base_risk))
        complexity = max(0.0, min(1.0, base_complexity))
        
        return Metrics(
            reward=reward,
            risk=risk,
            complexity=complexity,
            episodes_run=self._goal_spec.eval_limits.max_episodes,
            steps_run=self._goal_spec.eval_limits.max_steps_per_episode * 10,
        )


class DemoEvaluationHarness:
    """Adapter for TradingEvaluator to IANCoordinator interface."""
    
    def __init__(self, evaluator: TradingEvaluator):
        self._evaluator = evaluator
    
    def evaluate(self, agent_pack: AgentPack, goal_spec: GoalSpec, seed: int) -> Metrics:
        return self._evaluator.evaluate(agent_pack, seed)


# =============================================================================
# Mock Tau Sender
# =============================================================================

class DemoTauSender:
    """Mock Tau sender for demo purposes."""
    
    def __init__(self, verbose: bool = True):
        self._verbose = verbose
        self._tx_count = 0
        self._txs: List[dict] = []
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """Simulate sending a transaction to Tau Net."""
        self._tx_count += 1
        tx_hash = hashlib.sha256(tx_data).hexdigest()[:16]
        
        # Try to decode as JSON for display
        try:
            decoded = json.loads(tx_data.decode('utf-8'))
            tx_type = decoded.get('type', 'unknown')
        except:
            decoded = {"raw": tx_data.hex()[:32]}
            tx_type = "raw"
        
        self._txs.append({"hash": tx_hash, "data": decoded})
        
        if self._verbose:
            print(f"    📡 Tau TX #{self._tx_count}: {tx_type} -> {tx_hash}")
        
        return True, tx_hash
    
    def get_transactions(self) -> List[dict]:
        return self._txs


# =============================================================================
# Main Demo
# =============================================================================

def run_demo(config: DemoConfig = None):
    """Run the IAN trading agent competition demo."""
    config = config or DemoConfig()
    
    print("=" * 70)
    print("🤖 IAN Trading Agent Competition Demo")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Define the Goal
    # -------------------------------------------------------------------------
    print("📋 Step 1: Defining Goal")
    print("-" * 40)
    
    goal_spec = GoalSpec(
        goal_id=GoalID(config.goal_id),
        name="Trading Agent Competition 2025",
        description="""
        Optimize a trading strategy for maximum risk-adjusted returns.
        
        Metrics:
        - Reward: Sharpe ratio (higher is better)
        - Risk: Maximum drawdown (lower is better)
        - Complexity: Number of parameters (lower is better)
        
        Thresholds:
        - Minimum Sharpe: 0.1
        - Maximum Drawdown: 90%
        - Maximum Complexity: 90%
        """,
        eval_limits=EvaluationLimits(
            max_episodes=100,
            max_steps_per_episode=1000,
            timeout_seconds=60,
            max_memory_mb=512,
        ),
        thresholds=Thresholds(
            min_reward=0.1,
            max_risk=0.9,
            max_complexity=0.9,
        ),
    )
    
    print(f"  Goal ID: {goal_spec.goal_id}")
    print(f"  Name: {goal_spec.name}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Create Coordinator
    # -------------------------------------------------------------------------
    print("🔧 Step 2: Creating Coordinator")
    print("-" * 40)
    
    # Create base coordinator
    evaluator = TradingEvaluator(goal_spec)
    harness = DemoEvaluationHarness(evaluator)
    
    base_coordinator = IANCoordinator(
        goal_spec=goal_spec,
        config=CoordinatorConfig(
            leaderboard_capacity=config.leaderboard_size,
            use_pareto=False,  # Use scalar ranking for simplicity
        ),
        evaluation_harness=harness,
    )
    
    # Wrap with security if enabled
    if config.enable_security:
        print("  🔒 Security hardening: ENABLED")
        limits = SecurityLimits(
            RATE_LIMIT_TOKENS=10,
            RATE_LIMIT_REFILL_PER_SECOND=1.0,  # 1 token/second
        )
        coordinator = SecureCoordinator(
            base_coordinator,
            limits=limits,
            enable_pow=False,  # Disable PoW for demo
        )
    else:
        print("  🔓 Security hardening: DISABLED")
        coordinator = base_coordinator
    
    # Wrap with Tau integration if enabled
    tau_sender = None
    if config.enable_tau:
        print("  ⛓️  Tau Net integration: ENABLED")
        tau_sender = DemoTauSender(verbose=config.verbose)
        tau_bridge = TauBridge(
            sender=tau_sender,
            config=TauBridgeConfig(
                commit_interval_seconds=30,
                commit_threshold_contributions=5,
            ),
        )
        
        # Register goal on Tau
        tau_bridge.register_goal(goal_spec)
        
        # For Tau integration, we'll manually trigger commits
    else:
        print("  ⛓️  Tau Net integration: DISABLED")
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Submit Contributions
    # -------------------------------------------------------------------------
    print("📤 Step 3: Submitting Contributions")
    print("-" * 40)
    
    strategy_types = ["momentum", "mean_reversion", "breakout", "neural"]
    accepted_count = 0
    rejected_count = 0
    
    for contributor_idx in range(config.num_contributors):
        contributor_id = f"trader_{contributor_idx:03d}"
        
        for contrib_idx in range(config.contributions_per_contributor):
            # Generate a trading strategy
            strategy_type = strategy_types[(contributor_idx + contrib_idx) % len(strategy_types)]
            seed = contributor_idx * 1000 + contrib_idx
            strategy = generate_trading_strategy(strategy_type, seed)
            
            # Create contribution
            contribution = Contribution(
                goal_id=goal_spec.goal_id,
                agent_pack=strategy.to_agent_pack(),
                proofs={},
                contributor_id=contributor_id,
                seed=seed,
            )
            
            # Submit
            result = coordinator.process_contribution(contribution)
            
            if result.accepted:
                accepted_count += 1
                if config.verbose:
                    print(f"  ✅ {contributor_id}/{strategy.name}: "
                          f"score={result.score:.3f}, reward={result.metrics.reward:.3f}")
            else:
                rejected_count += 1
                if config.verbose:
                    print(f"  ❌ {contributor_id}/{strategy.name}: {result.reason}")
    
    print()
    print(f"  Summary: {accepted_count} accepted, {rejected_count} rejected")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: View Leaderboard
    # -------------------------------------------------------------------------
    print("🏆 Step 4: Leaderboard")
    print("-" * 40)
    
    leaderboard = coordinator.get_leaderboard()
    
    for rank, entry in enumerate(leaderboard[:10], 1):
        pack_hash = entry.pack_hash.hex()[:12]
        print(f"  #{rank:2d}: score={entry.score:.4f} | "
              f"contributor={entry.contributor_id} | hash={pack_hash}...")
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Active Policy
    # -------------------------------------------------------------------------
    print("🎯 Step 5: Active Policy")
    print("-" * 40)
    
    active = coordinator.get_active_policy()
    if active:
        print(f"  Pack Hash: {active.pack_hash.hex()[:24]}...")
        print(f"  Score: {active.score:.4f}")
        print(f"  Contributor: {active.contributor_id}")
        print(f"  Log Index: {active.log_index}")
    else:
        print("  No active policy yet")
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Tau Integration Summary
    # -------------------------------------------------------------------------
    if config.enable_tau and tau_sender:
        print("⛓️  Step 6: Tau Net Summary")
        print("-" * 40)
        
        # Manually trigger a commit
        base_coord = coordinator._coordinator if hasattr(coordinator, '_coordinator') else coordinator
        tau_bridge.commit_log(
            goal_id=goal_spec.goal_id,
            log_root=base_coord.get_log_root(),
            log_size=base_coord.state.log.size,
            leaderboard_root=base_coord.get_leaderboard_root(),
            leaderboard_size=len(base_coord.state.leaderboard),
        )
        
        # Trigger an upgrade
        if active:
            tau_bridge.upgrade_policy(
                goal_id=goal_spec.goal_id,
                new_policy=active,
                log_root=base_coord.get_log_root(),
                governance_signatures=[],
            )
        
        txs = tau_sender.get_transactions()
        print(f"  Total transactions: {len(txs)}")
        for tx in txs:
            print(f"    - {tx['hash']}: {tx['data'].get('type', 'unknown')}")
        
        print()
    
    # -------------------------------------------------------------------------
    # Step 7: Statistics
    # -------------------------------------------------------------------------
    print("📊 Step 7: Statistics")
    print("-" * 40)
    
    stats = coordinator.get_stats()
    for key, value in stats.items():
        if isinstance(value, bytes):
            value = value.hex()[:16] + "..."
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    
    return {
        "goal_spec": goal_spec,
        "coordinator": coordinator,
        "leaderboard": leaderboard,
        "active_policy": active,
        "stats": stats,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IAN Trading Agent Competition Demo (Simulated Evaluation)",
    )
    parser.add_argument("--contributors", type=int, default=5, help="Number of contributors")
    parser.add_argument("--contributions", type=int, default=3, help="Contributions per contributor")
    parser.add_argument("--no-security", action="store_true", help="Disable security hardening")
    parser.add_argument("--no-tau", action="store_true", help="Disable Tau integration")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    config = DemoConfig(
        num_contributors=args.contributors,
        contributions_per_contributor=args.contributions,
        enable_security=not args.no_security,
        enable_tau=not args.no_tau,
        verbose=not args.quiet,
    )
    
    run_demo(config)
