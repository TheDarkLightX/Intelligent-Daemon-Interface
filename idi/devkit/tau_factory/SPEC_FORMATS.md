# Tau Spec Formats: Pure vs Human-Readable

This document explains the two output formats for generated Tau specs.

## Overview

The Tau Factory supports two spec generation modes:

| Mode | Setting | Tau CLI Flag | Variable Names | Use Case |
|------|---------|--------------|----------------|----------|
| **Pure** (default) | `descriptive_names=False` | (none) | `i0`, `o0` | Production, compact |
| **Human-Readable** | `descriptive_names=True` | `--charvar off` | `action`, `confidence` | Documentation, debugging |

## Key Point: `--charvar off`

**Human-readable specs require running Tau with `--charvar off`:**

```bash
# Pure form (default charvar=on)
tau spec.tau

# Human-readable form (requires charvar=off)
tau --charvar off spec.tau
```

## Example: Confidence Gate Pattern

### Pure Form (default)

```python
schema = AgentSchema(
    name="confidence_gate_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="action", stream_type="sbf"),
        StreamConfig(name="confidence", stream_type="sbf"),
        StreamConfig(name="output", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(pattern="confidence_gate", inputs=("action", "confidence"), output="output"),
    ),
    descriptive_names=False,  # Default
)
```

**Generated Spec:**
```tau
# confidence_gate_agent - Generated Tau Agent
# Auto-generated from AgentSchema
# Strategy: custom

i0:sbf = in file("inputs/action.in")
i1:sbf = in file("inputs/confidence.in")

o0:sbf = out file("outputs/output.out")

# confidence_gate pattern: output <- action, confidence
r (o0[t] = i0[t] & i1[t])
```

### Human-Readable Form

```python
schema = AgentSchema(
    # ... same as above ...
    descriptive_names=True,  # Enable human-readable names
)
```

**Generated Spec:**
```tau
# confidence_gate_agent - Generated Tau Agent
# Auto-generated from AgentSchema
# Strategy: custom
# NOTE: This spec uses descriptive names - run with: tau --charvar off

action:sbf = in file("inputs/action.in")
confidence:sbf = in file("inputs/confidence.in")

output:sbf = out file("outputs/output.out")

# confidence_gate pattern: output <- action, confidence
r (output[t] = action[t] & confidence[t])
```

## Example: Safety Override Pattern

### Pure Form
```tau
i0:sbf = in file("inputs/agent_action.in")
i1:sbf = in file("inputs/safe_action.in")
i2:sbf = in file("inputs/safety_trigger.in")

o0:sbf = out file("outputs/final_action.out")

r (o0[t] = (i2[t] & i1[t]) | (i2[t]' & i0[t]))
```

### Human-Readable Form
```tau
# NOTE: Run with: tau --charvar off

agent_action:sbf = in file("inputs/agent_action.in")
safe_action:sbf = in file("inputs/safe_action.in")
safety_trigger:sbf = in file("inputs/safety_trigger.in")

final_action:sbf = out file("outputs/final_action.out")

r (final_action[t] = (safety_trigger[t] & safe_action[t]) | (safety_trigger[t]' & agent_action[t]))
```

## Example: Exploration vs Exploitation

### Pure Form
```tau
i0:sbf = in file("inputs/explore_action.in")
i1:sbf = in file("inputs/exploit_action.in")
i2:sbf = in file("inputs/explore_flag.in")

o0:sbf = out file("outputs/selected_action.out")

r (o0[t] = (i2[t] & i0[t]) | (i2[t]' & i1[t]))
```

### Human-Readable Form
```tau
# NOTE: Run with: tau --charvar off

explore_action:sbf = in file("inputs/explore_action.in")
exploit_action:sbf = in file("inputs/exploit_action.in")
explore_flag:sbf = in file("inputs/explore_flag.in")

selected_action:sbf = out file("outputs/selected_action.out")

r (selected_action[t] = (explore_flag[t] & explore_action[t]) | (explore_flag[t]' & exploit_action[t]))
```

## When to Use Each Format

### Use Pure Form (`descriptive_names=False`)
- Production deployments
- Automated pipelines
- When spec size matters
- Default Tau execution

### Use Human-Readable Form (`descriptive_names=True`)
- Documentation and examples
- Debugging and development
- Code reviews
- Teaching/learning Tau
- **Remember:** Must run with `tau --charvar off`

## Technical Notes

1. **Variable naming in Tau:**
   - With `charvar=on` (default): Variables must be single character + digits (e.g., `i0`, `o1`)
   - With `charvar=off`: Variables can be multi-character identifiers (e.g., `action`, `confidence`)

2. **The `descriptive_names` option:**
   - Part of `AgentSchema` dataclass
   - Default is `False` (pure form)
   - When `True`, generates a comment reminder about `--charvar off`

3. **File paths remain unchanged:**
   - Input/output file paths use the original stream names regardless of mode
   - Only the Tau variable names change
