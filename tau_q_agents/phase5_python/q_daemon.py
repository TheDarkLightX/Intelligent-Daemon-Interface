#!/usr/bin/env python3
"""
Q-Learning Daemon for Tau Agents
Manages Q-tables, communicates with Tau via subprocess, learns from rewards.
"""

import subprocess
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple
import json
import time


class Action(IntEnum):
    HOLD = 0   # 00
    BUY = 1    # 01
    SELL = 2   # 10
    WAIT = 3   # 11


class State(IntEnum):
    IDLE_NEUTRAL = 0
    IDLE_UP = 1
    IDLE_DOWN = 2
    HOLDING_NEUTRAL = 3
    HOLDING_UP = 4
    HOLDING_DOWN = 5


@dataclass
class MarketState:
    price_up: bool
    price_down: bool
    position: bool  # True = holding
    
    def to_state_index(self) -> int:
        """Convert market state to Q-table index"""
        base = 3 if self.position else 0
        if self.price_up:
            return base + 1
        elif self.price_down:
            return base + 2
        return base


class QTable:
    """Multi-layer Q-table with epsilon-greedy selection"""
    
    def __init__(self, n_states: int = 6, n_actions: int = 4, 
                 learning_rate: float = 0.1, discount: float = 0.95,
                 epsilon: float = 0.1):
        self.q = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Initialize with some prior knowledge
        self._init_priors()
    
    def _init_priors(self):
        """Initialize Q-values with domain knowledge"""
        # IDLE states: prefer BUY on up, HOLD on neutral/down
        self.q[State.IDLE_NEUTRAL, Action.HOLD] = 0.5
        self.q[State.IDLE_UP, Action.BUY] = 1.0
        self.q[State.IDLE_DOWN, Action.HOLD] = 0.6
        
        # HOLDING states: prefer SELL on down, HOLD on up
        self.q[State.HOLDING_NEUTRAL, Action.HOLD] = 0.5
        self.q[State.HOLDING_UP, Action.HOLD] = 0.8
        self.q[State.HOLDING_DOWN, Action.SELL] = 1.0
    
    def select_action(self, state: int, explore: bool = True) -> Action:
        """Epsilon-greedy action selection"""
        if explore and np.random.random() < self.epsilon:
            return Action(np.random.randint(self.n_actions))
        return Action(np.argmax(self.q[state]))
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool = False):
        """Q-learning update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])
        
        self.q[state, action] += self.lr * (target - self.q[state, action])
    
    def get_best_action(self, state: int) -> Action:
        """Get greedy best action"""
        return Action(np.argmax(self.q[state]))
    
    def to_dict(self) -> dict:
        return {
            "q_values": self.q.tolist(),
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'QTable':
        q = cls(learning_rate=data["lr"], discount=data["gamma"], 
                epsilon=data["epsilon"])
        q.q = np.array(data["q_values"])
        return q


class LayeredQTable:
    """Multi-layer Q-table with strategy specialization"""
    
    def __init__(self):
        # Layer 1: Momentum strategy
        self.momentum = QTable(epsilon=0.05)
        self.momentum.q[State.IDLE_UP, Action.BUY] = 2.0  # Strong buy signal
        
        # Layer 2: Contrarian strategy
        self.contrarian = QTable(epsilon=0.05)
        self.contrarian.q[State.IDLE_DOWN, Action.BUY] = 1.5
        self.contrarian.q[State.HOLDING_UP, Action.SELL] = 1.5
        
        # Layer 3: Trend-following
        self.trend = QTable(epsilon=0.05)
        self.trend.q[State.IDLE_UP, Action.BUY] = 1.5
        self.trend.q[State.HOLDING_DOWN, Action.SELL] = 1.5
        
        # Meta-layer: weights for each strategy
        self.weights = np.array([0.4, 0.3, 0.3])  # momentum, contrarian, trend
    
    def select_action(self, state: int) -> Tuple[Action, List[float]]:
        """Weighted voting across layers"""
        q_momentum = self.momentum.q[state]
        q_contrarian = self.contrarian.q[state]
        q_trend = self.trend.q[state]
        
        # Weighted sum
        combined = (self.weights[0] * q_momentum + 
                   self.weights[1] * q_contrarian +
                   self.weights[2] * q_trend)
        
        action = Action(np.argmax(combined))
        return action, combined.tolist()
    
    def update_all(self, state: int, action: int, reward: float, 
                   next_state: int):
        """Update all layers"""
        self.momentum.update(state, action, reward, next_state)
        self.contrarian.update(state, action, reward, next_state)
        self.trend.update(state, action, reward, next_state)
    
    def update_weights(self, performance: List[float]):
        """Update meta-weights based on layer performance"""
        total = sum(performance) + 1e-6
        self.weights = np.array(performance) / total


class TauDaemon:
    """Manages Tau process and Q-learning loop"""
    
    def __init__(self, spec_path: str, use_layered: bool = True):
        self.spec_path = spec_path
        self.q_table = LayeredQTable() if use_layered else QTable()
        self.use_layered = use_layered
        self.history: List[dict] = []
        self.total_reward = 0.0
        self.trades = 0
        self.profitable_trades = 0
    
    def action_to_bits(self, action: Action) -> Tuple[int, int]:
        """Convert action to bit encoding for Tau"""
        return (action & 1, (action >> 1) & 1)
    
    def compute_reward(self, executed_sell: bool, executed_buy: bool,
                      price_up: bool, price_down: bool) -> float:
        """Compute reward signal"""
        reward = 0.0
        
        if executed_sell:
            self.trades += 1
            # Reward for selling (took profit)
            reward = 1.0
            self.profitable_trades += 1
        elif executed_buy:
            # Small cost for entering position
            reward = -0.1
        else:
            # Small reward for patience
            reward = 0.01
        
        return reward
    
    def run_episode(self, market_data: List[Tuple[bool, bool]], 
                   verbose: bool = True) -> dict:
        """Run one episode with market data"""
        state = MarketState(False, False, False)
        state_idx = state.to_state_index()
        episode_reward = 0.0
        actions_taken = []
        
        for step, (price_up, price_down) in enumerate(market_data):
            # Update market state
            state.price_up = price_up
            state.price_down = price_down
            state_idx = state.to_state_index()
            
            # Select action from Q-table
            if self.use_layered:
                action, q_values = self.q_table.select_action(state_idx)
            else:
                action = self.q_table.select_action(state_idx)
                q_values = self.q_table.q[state_idx].tolist()
            
            # Validate action
            valid = True
            if action == Action.BUY and state.position:
                valid = False
                action = Action.HOLD
            elif action == Action.SELL and not state.position:
                valid = False
                action = Action.HOLD
            
            # Execute action
            executed_buy = (action == Action.BUY)
            executed_sell = (action == Action.SELL)
            
            if executed_buy:
                state.position = True
            elif executed_sell:
                state.position = False
            
            # Compute reward
            reward = self.compute_reward(executed_sell, executed_buy,
                                        price_up, price_down)
            episode_reward += reward
            
            # Get next state
            next_state_idx = state.to_state_index()
            
            # Update Q-table
            if self.use_layered:
                self.q_table.update_all(state_idx, action, reward, next_state_idx)
            else:
                self.q_table.update(state_idx, action, reward, next_state_idx)
            
            actions_taken.append({
                "step": step,
                "price_up": price_up,
                "price_down": price_down,
                "action": action.name,
                "valid": valid,
                "reward": reward,
                "position": state.position,
                "q_values": q_values
            })
            
            if verbose:
                print(f"Step {step}: {'↑' if price_up else '↓' if price_down else '-'} "
                      f"| {action.name:4} | Pos: {'H' if state.position else 'I'} "
                      f"| R: {reward:+.2f}")
        
        self.total_reward += episode_reward
        
        return {
            "episode_reward": episode_reward,
            "trades": self.trades,
            "profitable": self.profitable_trades,
            "actions": actions_taken
        }
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            "total_reward": self.total_reward,
            "trades": self.trades,
            "profitable_trades": self.profitable_trades,
            "win_rate": self.profitable_trades / max(1, self.trades),
            "q_table": self.q_table.momentum.to_dict() if self.use_layered 
                      else self.q_table.to_dict()
        }


def generate_market_data(n_steps: int = 50, trend_prob: float = 0.6) -> List[Tuple[bool, bool]]:
    """Generate synthetic market data"""
    data = []
    trend = 1  # 1 = up, -1 = down
    
    for _ in range(n_steps):
        # Trend continuation
        if np.random.random() < trend_prob:
            price_up = trend > 0
        else:
            price_up = trend < 0
            # Trend reversal
            if np.random.random() < 0.3:
                trend *= -1
        
        price_down = not price_up
        data.append((price_up, price_down))
    
    return data


def test_intelligence():
    """Test agent intelligence with multiple episodes"""
    print("=" * 60)
    print("Q-Learning Agent Intelligence Test")
    print("=" * 60)
    
    # Create daemon
    daemon = TauDaemon("", use_layered=True)
    
    # Run multiple episodes
    n_episodes = 10
    episode_rewards = []
    
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1} ---")
        market_data = generate_market_data(30)
        result = daemon.run_episode(market_data, verbose=(ep == n_episodes - 1))
        episode_rewards.append(result["episode_reward"])
        print(f"Episode reward: {result['episode_reward']:.2f}")
    
    # Final stats
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = daemon.get_stats()
    print(f"Total Reward: {stats['total_reward']:.2f}")
    print(f"Total Trades: {stats['trades']}")
    print(f"Profitable: {stats['profitable_trades']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"Avg Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward Trend: {episode_rewards[-1] - episode_rewards[0]:+.2f}")
    
    # Show Q-table
    print("\nQ-Table (Momentum Layer):")
    print("State          | HOLD  | BUY   | SELL  | WAIT")
    print("-" * 50)
    q = daemon.q_table.momentum.q
    for s in State:
        print(f"{s.name:14} | {q[s, 0]:5.2f} | {q[s, 1]:5.2f} | {q[s, 2]:5.2f} | {q[s, 3]:5.2f}")
    
    return daemon


if __name__ == "__main__":
    daemon = test_intelligence()

