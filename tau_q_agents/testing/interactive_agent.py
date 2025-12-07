#!/usr/bin/env python3
"""
Interactive Q-Agent Testing
Play against the Q-agent or watch it trade
"""

import sys
import time
sys.path.insert(0, '../phase5_python')
from q_daemon import QTable, LayeredQTable, Action, State, MarketState, generate_market_data
import numpy as np


def print_market_viz(history: list, width: int = 60):
    """Print ASCII visualization of market and actions"""
    if not history:
        return
    
    print("\n📈 Market Visualization")
    print("-" * width)
    
    # Price line
    print("Price: ", end="")
    for h in history[-width:]:
        if h["price_up"]:
            print("↑", end="")
        else:
            print("↓", end="")
    print()
    
    # Action line
    print("Action:", end="")
    for h in history[-width:]:
        action = h["action"]
        if action == Action.BUY:
            print("B", end="")
        elif action == Action.SELL:
            print("S", end="")
        elif action == Action.HOLD:
            print("·", end="")
        else:
            print(" ", end="")
    print()
    
    # Position line
    print("Pos:   ", end="")
    for h in history[-width:]:
        if h["position"]:
            print("█", end="")
        else:
            print(" ", end="")
    print()


def print_state(step: int, state: MarketState, action: Action, 
                reward: float, total_reward: float, q_values: list):
    """Print current state"""
    signal = "📈" if state.price_up else "📉"
    pos = "🏠 HOLDING" if state.position else "💰 IDLE"
    
    action_emoji = {
        Action.HOLD: "⏸️  HOLD",
        Action.BUY: "🟢 BUY ",
        Action.SELL: "🔴 SELL",
        Action.WAIT: "⏳ WAIT"
    }
    
    print(f"\n{'='*50}")
    print(f"Step {step}: {signal} Price {'UP' if state.price_up else 'DOWN'}")
    print(f"Position: {pos}")
    print(f"Action: {action_emoji[action]}")
    print(f"Reward: {reward:+.2f} | Total: {total_reward:.2f}")
    print(f"\nQ-Values: HOLD={q_values[0]:.2f} BUY={q_values[1]:.2f} "
          f"SELL={q_values[2]:.2f} WAIT={q_values[3]:.2f}")


def interactive_mode():
    """Interactive trading mode - you provide market signals"""
    print("\n" + "=" * 60)
    print("🎮 Interactive Q-Agent Mode")
    print("=" * 60)
    print("Commands:")
    print("  u/up    - Price goes UP")
    print("  d/down  - Price goes DOWN")
    print("  a/auto  - Run N automatic steps")
    print("  r/reset - Reset agent")
    print("  q/quit  - Exit")
    print("=" * 60)
    
    agent = LayeredQTable()
    position = False
    total_reward = 0.0
    step = 0
    history = []
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        
        if cmd in ('q', 'quit', 'exit'):
            break
        elif cmd in ('r', 'reset'):
            agent = LayeredQTable()
            position = False
            total_reward = 0.0
            step = 0
            history = []
            print("🔄 Agent reset!")
            continue
        elif cmd.startswith('a') or cmd.startswith('auto'):
            # Auto mode
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            print(f"Running {n} auto steps...")
            
            for _ in range(n):
                price_up = np.random.random() > 0.5
                price_down = not price_up
                state = MarketState(price_up, price_down, position)
                action, q_values = agent.select_action(state.to_state_index())
                
                # Validate
                if action == Action.BUY and position:
                    action = Action.HOLD
                elif action == Action.SELL and not position:
                    action = Action.HOLD
                
                # Execute
                reward = 0.01
                if action == Action.BUY:
                    position = True
                    reward = -0.1
                elif action == Action.SELL:
                    position = False
                    reward = 1.0
                    next_state = MarketState(price_up, price_down, position)
                    agent.update_all(state.to_state_index(), action, reward,
                                   next_state.to_state_index())
                
                total_reward += reward
                history.append({
                    "price_up": price_up, "action": action, 
                    "position": position, "reward": reward
                })
                step += 1
            
            print_market_viz(history)
            print(f"\nTotal Reward: {total_reward:.2f} | Steps: {step}")
            continue
        
        # Manual price input
        if cmd in ('u', 'up'):
            price_up, price_down = True, False
        elif cmd in ('d', 'down'):
            price_up, price_down = False, True
        else:
            print("Unknown command. Use: u/d/auto/reset/quit")
            continue
        
        # Process step
        state = MarketState(price_up, price_down, position)
        action, q_values = agent.select_action(state.to_state_index())
        
        # Validate
        valid = True
        if action == Action.BUY and position:
            action = Action.HOLD
            valid = False
        elif action == Action.SELL and not position:
            action = Action.HOLD
            valid = False
        
        # Execute
        reward = 0.01
        if action == Action.BUY:
            position = True
            reward = -0.1
        elif action == Action.SELL:
            position = False
            reward = 1.0
            next_state = MarketState(price_up, price_down, position)
            agent.update_all(state.to_state_index(), action, reward,
                           next_state.to_state_index())
        
        total_reward += reward
        history.append({
            "price_up": price_up, "action": action,
            "position": position, "reward": reward
        })
        step += 1
        
        print_state(step, state, action, reward, total_reward, q_values)
        
        if len(history) > 5:
            print_market_viz(history)
    
    print(f"\n🏁 Final Score: {total_reward:.2f} over {step} steps")
    print("Goodbye!")


def watch_mode():
    """Watch the agent trade automatically"""
    print("\n" + "=" * 60)
    print("👀 Watch Mode - Observe Q-Agent Trading")
    print("=" * 60)
    
    agent = LayeredQTable()
    position = False
    total_reward = 0.0
    history = []
    
    # Generate market
    market = generate_market_data(100, 0.6)
    
    print("Press Ctrl+C to stop\n")
    
    try:
        for step, (price_up, price_down) in enumerate(market):
            state = MarketState(price_up, price_down, position)
            action, q_values = agent.select_action(state.to_state_index())
            
            # Validate
            if action == Action.BUY and position:
                action = Action.HOLD
            elif action == Action.SELL and not position:
                action = Action.HOLD
            
            # Execute
            reward = 0.01
            if action == Action.BUY:
                position = True
                reward = -0.1
            elif action == Action.SELL:
                position = False
                reward = 1.0
                next_state = MarketState(price_up, price_down, position)
                agent.update_all(state.to_state_index(), action, reward,
                               next_state.to_state_index())
            
            total_reward += reward
            history.append({
                "price_up": price_up, "action": action,
                "position": position, "reward": reward
            })
            
            # Display
            signal = "📈" if price_up else "📉"
            act = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️", "WAIT": "⏳"}
            pos = "█" if position else " "
            
            print(f"Step {step:3d}: {signal} {act.get(action.name, '?')} [{pos}] "
                  f"R:{reward:+.2f} Total:{total_reward:6.2f}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    
    print("\n" + "=" * 60)
    print_market_viz(history, 80)
    print(f"\n🏁 Final: {total_reward:.2f} reward, {len(history)} steps")


def main():
    print("🤖 Q-Agent Interactive Testing")
    print("=" * 60)
    print("1. Interactive Mode (you control prices)")
    print("2. Watch Mode (observe agent)")
    print("3. Quit")
    
    try:
        choice = input("\nSelect mode (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        watch_mode()
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()

