# Ensemble Trading Agent Example

This example demonstrates how to create an ensemble trading agent using majority voting.

## Overview

The ensemble agent combines votes from 3 independent trading agents:
- **Agent 1**: Momentum-based trader
- **Agent 2**: Mean-reversion trader  
- **Agent 3**: Regime-aware trader

The agent uses **2-of-3 majority voting** - a trade executes only if at least 2 agents agree.

## Usage

### Generate the Agent

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="ensemble_trading_agent",
    strategy="custom",
    streams=(
        # Inputs: Three independent agents
        StreamConfig(name="agent1_buy", stream_type="sbf"),
        StreamConfig(name="agent2_buy", stream_type="sbf"),
        StreamConfig(name="agent3_buy", stream_type="sbf"),
        StreamConfig(name="agent_sell", stream_type="sbf"),
        
        # Outputs
        StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        # 2-of-3 majority voting
        LogicBlock(
            pattern="majority",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="majority_buy",
            params={"threshold": 2, "total": 3}
        ),
        
        # Position FSM: buy on majority, sell on signal
        LogicBlock(
            pattern="fsm",
            inputs=("majority_buy", "agent_sell"),
            output="position"
        ),
    ),
)

spec = generate_tau_spec(schema)
print(spec)
```

### Run the Agent

1. Create input files in `inputs/`:
   ```bash
   # agent1_buy.in - Momentum agent votes
   echo -e "0\n1\n1\n0\n1" > inputs/agent1_buy.in
   
   # agent2_buy.in - Mean-reversion agent votes
   echo -e "1\n0\n1\n1\n0" > inputs/agent2_buy.in
   
   # agent3_buy.in - Regime-aware agent votes
   echo -e "0\n1\n0\n1\n1" > inputs/agent3_buy.in
   
   # agent_sell.in - Sell signal
   echo -e "0\n0\n0\n1\n0" > inputs/agent_sell.in
   ```

2. Run with Tau:
   ```bash
   tau < ensemble_trading_agent.tau
   ```

3. Check outputs:
   ```bash
   cat outputs/majority_buy.out  # Majority vote result
   cat outputs/position.out      # Final position
   ```

## Expected Behavior

**Majority Logic (2-of-3):**
- `(0,0,0)` → `0` (no majority)
- `(0,0,1)` → `0` (no majority)
- `(0,1,1)` → `1` (2 votes = majority) ✅
- `(1,1,1)` → `1` (3 votes = majority) ✅

**Position Logic:**
- Buy when majority = 1
- Sell when agent_sell = 1
- Maintain position otherwise

## Benefits

1. **Reduced False Positives** - Requires 2+ agents to agree
2. **Fault Tolerance** - One bad agent doesn't break the system
3. **Diversity** - Different strategies complement each other
4. **Confidence** - Majority vote indicates higher confidence

## Files

- `ensemble_trading_agent.tau` - Generated Tau spec
- `inputs/` - Input files for each agent
- `outputs/` - Generated outputs
- `README.md` - This file

