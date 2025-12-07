#!/usr/bin/env python3
"""
Improved Q-Agent v2
- Better reward shaping for higher win rates
- Curiosity-driven exploration
- Position sizing based on confidence
- Trend-following with momentum filters
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
import time


# ============================================================================
# EMOJI DEFINITIONS
# ============================================================================

class E:
    """Emoji shortcuts"""
    UP = "📈"; DOWN = "📉"; MONEY = "💰"; FIRE = "🔥"
    ROCKET = "🚀"; TROPHY = "🏆"; BRAIN = "🧠"; TARGET = "🎯"
    CHECK = "✅"; STAR = "⭐"; DIAMOND = "💎"; SKULL = "💀"
    HAPPY = "😊"; EXCITED = "🤩"; NEUTRAL = "😐"; WORRIED = "😟"
    BUY = "🟢"; SELL = "🔴"; HOLD = "⏸️"; BURN = "🔥"


# ============================================================================
# IMPROVED REWARD SHAPING
# ============================================================================

class RewardShaper:
    """
    Advanced reward shaping for better learning
    
    Key improvements:
    1. Asymmetric rewards (bigger penalty for losses)
    2. Time-weighted returns
    3. Risk-adjusted rewards
    4. Momentum bonuses
    """
    
    def __init__(self):
        self.trade_history: List[float] = []
        self.holding_start_price = 0.0
        self.holding_duration = 0
        self.peak_unrealized = 0.0
        
    def compute_reward(self, action: int, executed: bool, 
                      price: float, prev_price: float,
                      position: bool, entry_price: float,
                      momentum: float, volatility: float) -> float:
        """
        Compute shaped reward
        
        action: 0=HOLD, 1=BUY, 2=SELL
        """
        reward = 0.0
        
        if not executed:
            return -0.1  # Invalid action penalty
        
        price_change = (price - prev_price) / prev_price if prev_price > 0 else 0
        
        if action == 1:  # BUY
            # Base cost to enter
            reward = -0.01
            
            # BONUS for good entry timing
            if momentum > 0.02:  # Buying with momentum
                reward += 0.02
            if momentum > 0.04:  # Strong momentum
                reward += 0.03
            if volatility < 0.02:  # Low volatility entry (stable)
                reward += 0.01
            
            # Bonus for buying after dip (mean reversion)
            if price_change < -0.01:
                reward += 0.02
            
            self.holding_start_price = price
            self.holding_duration = 0
            self.peak_unrealized = 0.0
            
        elif action == 2:  # SELL
            if entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                # Base reward from PnL - BALANCED for better risk/reward
                if pnl_pct > 0:
                    # Profitable: reward scales with profit
                    reward = 0.5 + pnl_pct * 20  # Higher multiplier for wins
                    
                    # Bonus for quick profits
                    if self.holding_duration < 10:
                        reward *= 1.3
                    
                    # Bonus for taking profit near peak
                    if self.peak_unrealized > 0:
                        efficiency = pnl_pct / self.peak_unrealized
                        if efficiency > 0.8:
                            reward *= 1.2
                else:
                    # Loss: SMALLER penalty to encourage cutting losses early
                    reward = -0.2 + pnl_pct * 5  # Reduced penalty
                    
                    # BONUS for cutting losses quickly (stop loss behavior)
                    if self.holding_duration < 5 and pnl_pct > -0.03:
                        reward += 0.3  # Reward for quick small loss
                    
                    # Only penalize for holding losers too long
                    if self.holding_duration > 15 and pnl_pct < -0.05:
                        reward *= 1.5
                
                self.trade_history.append(pnl_pct)
            else:
                reward = 0.1  # Flat exit, small positive
                
        else:  # HOLD
            if position:
                self.holding_duration += 1
                
                # Track unrealized PnL
                if entry_price > 0:
                    unrealized = (price - entry_price) / entry_price
                    self.peak_unrealized = max(self.peak_unrealized, unrealized)
                    
                    # Small reward for holding winners
                    if unrealized > 0:
                        reward = 0.005 * (1 + unrealized * 10)  # Bigger reward for winners
                    else:
                        # Penalty for holding losers
                        reward = -0.01 * (1 - unrealized * 5)
            else:
                # Not in position - penalize missing opportunities
                reward = -0.002  # Small penalty for not being in market
                
                # BIGGER penalty for missing big moves
                if abs(price_change) > 0.015:
                    reward -= 0.02
                if momentum > 0.03:  # Missing a trend
                    reward -= 0.01
        
        return reward
    
    def get_win_rate(self) -> float:
        """Get current win rate"""
        if not self.trade_history:
            return 0.5
        wins = sum(1 for t in self.trade_history if t > 0)
        return wins / len(self.trade_history)
    
    def get_avg_win_loss(self) -> Tuple[float, float]:
        """Get average win and loss sizes"""
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return avg_win, avg_loss


# ============================================================================
# CURIOSITY-DRIVEN EXPLORATION
# ============================================================================

class CuriosityModule:
    """
    Intrinsic motivation through curiosity
    
    Rewards visiting novel states and surprising outcomes
    """
    
    def __init__(self, n_states: int):
        self.state_visits = np.zeros(n_states)
        self.transition_model: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.prediction_errors: List[float] = []
        
    def compute_intrinsic_reward(self, state: int, action: int, 
                                  next_state: int) -> float:
        """Compute curiosity bonus"""
        # Novelty bonus (inverse visit count)
        visit_count = self.state_visits[state]
        novelty_bonus = 1.0 / (visit_count + 1)
        
        # Prediction error bonus
        key = (state, action)
        if key in self.transition_model:
            predicted = self.transition_model[key]
            total = sum(predicted.values())
            if total > 0:
                expected_prob = predicted.get(next_state, 0) / total
                surprise = 1.0 - expected_prob
            else:
                surprise = 1.0
        else:
            surprise = 1.0
        
        # Update models
        self.state_visits[state] += 1
        
        if key not in self.transition_model:
            self.transition_model[key] = {}
        self.transition_model[key][next_state] = \
            self.transition_model[key].get(next_state, 0) + 1
        
        # Combined intrinsic reward
        intrinsic = 0.1 * novelty_bonus + 0.1 * surprise
        
        return intrinsic
    
    def get_exploration_bonus(self, state: int) -> float:
        """Get exploration bonus for action selection"""
        visits = self.state_visits[state]
        return np.sqrt(2 * np.log(sum(self.state_visits) + 1) / (visits + 1))


# ============================================================================
# IMPROVED Q-TABLE
# ============================================================================

class ImprovedQTable:
    """
    Q-Table with:
    - Double Q-learning (reduce overestimation)
    - Prioritized experience replay
    - Adaptive learning rate
    """
    
    def __init__(self, n_states: int, n_actions: int = 4):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Double Q-learning
        self.q1 = np.zeros((n_states, n_actions))
        self.q2 = np.zeros((n_states, n_actions))
        
        # Visit counts
        self.visits = np.zeros((n_states, n_actions))
        self.total_updates = 0
        
        # Hyperparameters
        self.lr_base = 0.2
        self.gamma = 0.95
        self.epsilon = 0.15
        
        # Experience replay
        self.replay_buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.buffer_size = 5000
    
    def get_q(self, state: int) -> np.ndarray:
        """Get combined Q-values"""
        return (self.q1[state] + self.q2[state]) / 2
    
    def select_action(self, state: int, exploration_bonus: float = 0.0,
                     explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select action with UCB exploration"""
        q_values = self.get_q(state)
        
        if explore:
            # UCB bonus
            visits = self.visits[state] + 1
            ucb = np.sqrt(2 * np.log(self.total_updates + 1) / visits)
            adjusted_q = q_values + 0.3 * ucb + exploration_bonus
            
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = np.argmax(adjusted_q)
        else:
            action = np.argmax(q_values)
        
        return action, q_values
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool = False):
        """Double Q-learning update"""
        # Randomly choose which Q to update
        if np.random.random() < 0.5:
            q_update, q_target = self.q1, self.q2
        else:
            q_update, q_target = self.q2, self.q1
        
        # Compute target
        if done:
            target = reward
        else:
            # Use q_update to select action, q_target to evaluate
            best_action = np.argmax(q_update[next_state])
            target = reward + self.gamma * q_target[next_state, best_action]
        
        # Adaptive learning rate
        visit_count = self.visits[state, action]
        lr = self.lr_base / (1 + visit_count * 0.01)
        
        # TD error for prioritized replay
        td_error = abs(target - q_update[state, action])
        
        # Update
        q_update[state, action] += lr * (target - q_update[state, action])
        self.visits[state, action] += 1
        self.total_updates += 1
        
        # Store experience
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        self.priorities.append(td_error + 0.01)
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)
    
    def replay(self, batch_size: int = 32):
        """Prioritized experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample based on priorities
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.replay_buffer), batch_size,
                                  p=probs, replace=False)
        
        for idx in indices:
            s, a, r, ns, d = self.replay_buffer[idx]
            
            # Mini-update
            if np.random.random() < 0.5:
                q = self.q1
            else:
                q = self.q2
            
            target = r if d else r + self.gamma * np.max(self.get_q(ns))
            q[s, a] += 0.05 * (target - q[s, a])


# ============================================================================
# FEATURE ENCODER
# ============================================================================

class FeatureEncoder:
    """Compact feature encoding"""
    
    def __init__(self, momentum_bins: int = 7, vol_bins: int = 5,
                 pos_bins: int = 3, pnl_bins: int = 7, streak_bins: int = 5):
        self.m_bins = momentum_bins
        self.v_bins = vol_bins
        self.p_bins = pos_bins
        self.pnl_bins = pnl_bins
        self.s_bins = streak_bins
        
        self.n_states = momentum_bins * vol_bins * pos_bins * pnl_bins * streak_bins
    
    def encode(self, momentum: float, volatility: float, position: bool,
               unrealized_pnl: float, streak: int) -> int:
        """Encode features to state index"""
        m = self._bin(momentum, -0.1, 0.1, self.m_bins)
        v = self._bin(volatility, 0, 0.1, self.v_bins)
        
        if not position:
            p = 0
        elif unrealized_pnl > 0:
            p = 1
        else:
            p = 2
        
        pnl = self._bin(unrealized_pnl, -0.1, 0.1, self.pnl_bins)
        s = self._bin(streak, -3, 3, self.s_bins)
        
        idx = m
        idx = idx * self.v_bins + v
        idx = idx * self.p_bins + p
        idx = idx * self.pnl_bins + pnl
        idx = idx * self.s_bins + s
        
        return min(idx, self.n_states - 1)
    
    def _bin(self, val: float, min_v: float, max_v: float, n: int) -> int:
        norm = (val - min_v) / (max_v - min_v + 1e-8)
        norm = np.clip(norm, 0, 0.999)
        return int(norm * n)


# ============================================================================
# IMPROVED AGENT
# ============================================================================

class ImprovedAgent:
    """Full improved Q-agent"""
    
    def __init__(self):
        self.encoder = FeatureEncoder()
        print(f"{E.BRAIN} State space: {self.encoder.n_states:,} states")
        
        self.q_table = ImprovedQTable(self.encoder.n_states)
        self.reward_shaper = RewardShaper()
        self.curiosity = CuriosityModule(self.encoder.n_states)
        
        # State
        self.position = False
        self.entry_price = 0.0
        self.streak = 0
        self.prices: List[float] = [100.0]
        
        # Stats
        self.total_reward = 0.0
        self.trades = 0
        self.wins = 0
        self.episode_rewards: List[float] = []
        self.messages: List[str] = []
    
    def _get_momentum(self) -> float:
        if len(self.prices) < 5:
            return 0.0
        return (self.prices[-1] - self.prices[-5]) / self.prices[-5]
    
    def _get_volatility(self) -> float:
        if len(self.prices) < 10:
            return 0.02
        returns = np.diff(self.prices[-10:]) / np.array(self.prices[-10:-1])
        return np.std(returns)
    
    def _get_unrealized_pnl(self) -> float:
        if not self.position or self.entry_price == 0:
            return 0.0
        return (self.prices[-1] - self.entry_price) / self.entry_price
    
    def step(self, price_up: bool, price_down: bool) -> Tuple[int, float, str]:
        """Execute one step"""
        # Update price
        change = 0.01 if price_up else -0.01
        new_price = self.prices[-1] * (1 + change)
        prev_price = self.prices[-1]
        self.prices.append(new_price)
        if len(self.prices) > 100:
            self.prices.pop(0)
        
        # Get state
        momentum = self._get_momentum()
        volatility = self._get_volatility()
        unrealized = self._get_unrealized_pnl()
        
        state = self.encoder.encode(momentum, volatility, self.position,
                                    unrealized, self.streak)
        
        # Get exploration bonus from curiosity
        explore_bonus = self.curiosity.get_exploration_bonus(state)
        
        # Select action
        action, q_values = self.q_table.select_action(state, explore_bonus)
        
        # Validate
        executed = True
        if action == 1 and self.position:  # Can't buy if holding
            action = 0
            executed = False
        elif action == 2 and not self.position:  # Can't sell if not holding
            action = 0
            executed = False
        
        # STOP LOSS: Force sell if loss exceeds 5%
        if self.position and action != 2:
            unrealized = self._get_unrealized_pnl()
            if unrealized < -0.05:
                action = 2
                executed = True
        
        # Execute
        if action == 1:  # BUY
            self.position = True
            self.entry_price = new_price
        elif action == 2:  # SELL
            if self.entry_price > 0:
                pnl = (new_price - self.entry_price) / self.entry_price
                self.trades += 1
                if pnl > 0:
                    self.wins += 1
                    self.streak = max(1, self.streak + 1)
                else:
                    self.streak = min(-1, self.streak - 1)
            self.position = False
        
        # Compute shaped reward
        reward = self.reward_shaper.compute_reward(
            action, executed, new_price, prev_price,
            self.position, self.entry_price,
            momentum, volatility
        )
        
        # Get next state
        next_state = self.encoder.encode(
            self._get_momentum(), self._get_volatility(),
            self.position, self._get_unrealized_pnl(), self.streak
        )
        
        # Add curiosity bonus
        intrinsic = self.curiosity.compute_intrinsic_reward(state, action, next_state)
        total_reward = reward + intrinsic
        
        # Update Q-table
        self.q_table.update(state, action, total_reward, next_state)
        
        # Replay
        if self.q_table.total_updates % 50 == 0:
            self.q_table.replay(16)
        
        self.total_reward += reward
        
        # Generate message
        msg = self._gen_message(action, reward, q_values)
        self.messages.append(msg)
        
        return action, reward, msg
    
    def _gen_message(self, action: int, reward: float, q: np.ndarray) -> str:
        """Generate emoji message"""
        parts = []
        
        # Emotion based on recent performance
        win_rate = self.reward_shaper.get_win_rate()
        if win_rate > 0.6:
            parts.append(E.EXCITED)
        elif win_rate > 0.45:
            parts.append(E.HAPPY)
        elif win_rate > 0.35:
            parts.append(E.NEUTRAL)
        else:
            parts.append(E.WORRIED)
        
        # Action
        parts.append([E.HOLD, E.BUY, E.SELL, E.HOLD][action])
        
        # Streak
        if self.streak > 2:
            parts.append(E.FIRE)
        elif self.streak < -2:
            parts.append(E.SKULL)
        
        # Position status
        if self.position:
            unrealized = self._get_unrealized_pnl()
            if unrealized > 0.02:
                parts.append(E.MONEY)
            elif unrealized < -0.02:
                parts.append(E.DOWN)
        
        # Reward indicator
        if reward > 0.5:
            parts.append(E.UP)
        
        return " ".join(parts)
    
    def run_episode(self, market_data: List[Tuple[bool, bool]], 
                   verbose: bool = False) -> float:
        """Run one episode"""
        ep_reward = 0.0
        
        for step, (up, down) in enumerate(market_data):
            action, reward, msg = self.step(up, down)
            ep_reward += reward
            
            if verbose and step % 30 == 0:
                print(f"  Step {step:3d}: {E.UP if up else E.DOWN} {msg} "
                      f"R:{reward:+.3f}")
        
        self.episode_rewards.append(ep_reward)
        return ep_reward
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            "total_reward": self.total_reward,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.trades),
            "episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            "states_visited": int(np.sum(self.q_table.visits > 0)),
            "coverage": np.sum(self.q_table.visits > 0) / self.encoder.n_states * 100
        }
    
    def print_status(self):
        """Print status"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"{E.BRAIN} Improved Agent Status")
        print(f"{'='*60}")
        print(f"{E.MONEY} Total Reward: {stats['total_reward']:.2f}")
        print(f"{E.TARGET} Trades: {stats['trades']} "
              f"(Win Rate: {stats['win_rate']*100:.1f}%)")
        print(f"{E.STAR} Avg Episode Reward: {stats['avg_reward']:.2f}")
        print(f"{E.BRAIN} States Visited: {stats['states_visited']:,} "
              f"({stats['coverage']:.1f}%)")
        
        # Win/loss analysis
        avg_win, avg_loss = self.reward_shaper.get_avg_win_loss()
        print(f"\n📊 Win Analysis:")
        print(f"   Avg Win:  {avg_win*100:+.2f}%")
        print(f"   Avg Loss: {avg_loss*100:+.2f}%")


# ============================================================================
# TRAINING
# ============================================================================

def generate_market(n: int, regime: str = "mixed") -> List[Tuple[bool, bool]]:
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
        
        up = np.random.random() < trend
        data.append((up, not up))
    
    return data


def run_training():
    """Run improved training"""
    print(f"{'='*70}")
    print(f"{E.ROCKET} IMPROVED Q-AGENT TRAINING {E.ROCKET}")
    print(f"{'='*70}")
    
    agent = ImprovedAgent()
    
    # Training phases
    phases = [
        ("Exploration", 50, 0.25),
        ("Learning", 100, 0.15),
        ("Refinement", 150, 0.08),
        ("Mastery", 200, 0.03),
    ]
    
    for phase_name, episodes, epsilon in phases:
        print(f"\n{E.STAR} Phase: {phase_name} ({episodes} episodes, ε={epsilon})")
        print("-" * 50)
        
        agent.q_table.epsilon = epsilon
        phase_rewards = []
        
        for ep in range(episodes):
            regime = ["mixed", "bull", "bear", "volatile"][ep % 4]
            market = generate_market(200, regime)
            reward = agent.run_episode(market, verbose=False)
            phase_rewards.append(reward)
            
            if (ep + 1) % max(1, episodes // 4) == 0:
                stats = agent.get_stats()
                print(f"  Ep {ep+1:3d}: R={np.mean(phase_rewards[-10:]):7.2f} "
                      f"Win={stats['win_rate']*100:5.1f}% "
                      f"Cov={stats['coverage']:.1f}%")
        
        print(f"  {E.CHECK} Phase avg: {np.mean(phase_rewards):.2f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"{E.TROPHY} FINAL EVALUATION")
    print(f"{'='*70}")
    
    agent.q_table.epsilon = 0.0
    
    for regime in ["bull", "bear", "volatile", "mixed"]:
        market = generate_market(500, regime)
        reward = agent.run_episode(market, verbose=False)
        stats = agent.get_stats()
        print(f"  {regime:10}: R={reward:8.2f} Win={stats['win_rate']*100:5.1f}%")
    
    agent.print_status()
    
    # Show messages
    print(f"\n{E.FIRE} Recent Communications:")
    for msg in agent.messages[-12:]:
        print(f"  {msg}")
    
    return agent


if __name__ == "__main__":
    agent = run_training()

