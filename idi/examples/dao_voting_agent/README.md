# DAO Voting Agent Example

This example demonstrates DAO-style governance voting using unanimous consensus.

## Overview

The DAO voting agent implements a simple governance mechanism:
- **5 Voters**: Each voter can vote yes (1) or no (0)
- **Unanimous Consensus**: Proposal passes only if ALL voters agree
- **Position Tracking**: Tracks whether proposal is active

## Usage

### Generate the Agent

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="dao_voting_agent",
    strategy="custom",
    streams=(
        # Inputs: 5 voters
        StreamConfig(name="voter1", stream_type="sbf"),
        StreamConfig(name="voter2", stream_type="sbf"),
        StreamConfig(name="voter3", stream_type="sbf"),
        StreamConfig(name="voter4", stream_type="sbf"),
        StreamConfig(name="voter5", stream_type="sbf"),
        StreamConfig(name="reset", stream_type="sbf"),
        
        # Outputs
        StreamConfig(name="consensus", stream_type="sbf", is_input=False),
        StreamConfig(name="proposal_active", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        # Unanimous consensus: all must agree
        LogicBlock(
            pattern="unanimous",
            inputs=("voter1", "voter2", "voter3", "voter4", "voter5"),
            output="consensus"
        ),
        
        # Proposal active FSM: set on consensus, reset on signal
        LogicBlock(
            pattern="fsm",
            inputs=("consensus", "reset"),
            output="proposal_active"
        ),
    ),
)

spec = generate_tau_spec(schema)
print(spec)
```

### Run the Agent

1. Create input files:
   ```bash
   # All voters agree
   echo -e "1\n1\n1\n1\n1" > inputs/voter1.in
   echo -e "1\n1\n1\n1\n1" > inputs/voter2.in
   echo -e "1\n1\n1\n1\n1" > inputs/voter3.in
   echo -e "1\n1\n1\n1\n1" > inputs/voter4.in
   echo -e "1\n1\n1\n1\n1" > inputs/voter5.in
   echo -e "0\n0\n0\n0\n1" > inputs/reset.in
   ```

2. Run with Tau:
   ```bash
   tau < dao_voting_agent.tau
   ```

3. Check outputs:
   ```bash
   cat outputs/consensus.out        # Unanimous consensus
   cat outputs/proposal_active.out  # Proposal status
   ```

## Expected Behavior

**Unanimous Logic:**
- `(1,1,1,1,1)` → `1` (all agree) ✅
- `(1,1,1,1,0)` → `0` (one disagrees)
- `(0,0,0,0,0)` → `0` (all disagree)
- `(1,0,1,1,1)` → `0` (one disagrees)

**Proposal Active Logic:**
- Set to 1 when consensus = 1
- Reset to 0 when reset = 1
- Maintain state otherwise

## Use Cases

1. **DAO Governance** - Require unanimous approval for critical proposals
2. **Multi-Signature Wallets** - All signers must agree
3. **Safety Systems** - All safety checks must pass
4. **Consensus Protocols** - Byzantine fault tolerance

## Files

- `dao_voting_agent.tau` - Generated Tau spec
- `inputs/` - Input files for voters
- `outputs/` - Generated outputs
- `README.md` - This file

