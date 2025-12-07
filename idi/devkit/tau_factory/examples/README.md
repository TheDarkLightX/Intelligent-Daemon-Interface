# Tau Agent Factory Examples

This directory contains example agents demonstrating various patterns and use cases.

## Example Agents

### Basic Examples

- **simple_fsm_agent.py** - Basic FSM agent with buy/sell signals
- **counter_agent.py** - Counter pattern example
- **majority_vote_agent.py** - Majority voting ensemble agent

### Advanced Examples

- **hierarchical_fsm_agent.py** - Supervisor-Worker pattern
- **decomposed_fsm_agent.py** - Decomposed FSM with substates
- **history_state_agent.py** - History state pattern
- **orthogonal_regions_agent.py** - Parallel FSMs

### Domain-Specific Examples

- **trading_agent.py** - Entry-Exit FSM for trading lifecycle
- **risk_management_agent.py** - Risk FSM with multiple risk levels
- **governance_agent.py** - Proposal FSM for governance
- **tcp_connection_agent.py** - TCP connection state machine
- **utxo_tracker_agent.py** - UTXO state machine for Bitcoin

## Running Examples

Each example includes:
- Agent schema definition
- Input file generation
- Spec generation
- Execution instructions

To run an example:

```bash
cd examples
python simple_fsm_agent.py
```

## Tutorials

See `tutorials/` directory for step-by-step guides on building agents.

