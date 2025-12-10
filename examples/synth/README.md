# Modular Synth Examples

This directory contains example goal specs and patches for the modular
synth / Auto-QAgent system.

## Files

### Goal Specs (Auto-QAgent)

- **`conservative_qagent_goal.json`**  
  A minimal, low-risk goal spec using synthetic evaluation. Good for
  quick iteration and testing.

- **`research_qagent_goal.json`**  
  A more extensive goal spec with real-mode evaluation, multiple
  environments, and multi-objective ranking. Suitable for research
  experiments.

### Agent Patches

- **`conservative_trader.agentpatch.json`**  
  A production-oriented Q-table agent patch with conservative
  hyperparameters and full spec/ZK profile.

## Usage

### Run Auto-QAgent with a goal spec

```bash
python -m idi.cli dev-auto-qagent \
  --goal examples/synth/conservative_qagent_goal.json \
  --out-patches output/patches
```

### Validate a patch

```bash
python -m idi.cli patch-validate \
  --patch examples/synth/conservative_trader.agentpatch.json
```

### Diff two patches

```bash
python -m idi.cli patch-diff \
  --old examples/synth/conservative_trader.agentpatch.json \
  --new output/patches/candidate-0.agentpatch.json
```

### Create a new patch from scratch

```bash
python -m idi.cli patch-create \
  --out my_patch.json \
  --id my-agent \
  --name "My Agent" \
  --description "Custom agent patch" \
  --agent-type qtable \
  --tag experimental
```

## See Also

- `Internal/AGENT_MODULAR_SYNTHESIZER_DESIGN.md` – Full design document
- `Internal/MODULAR_SYNTH_PRODUCTION_READINESS.md` – Production checklist
- `Internal/POWER_OF_PARAMETERIZATION.md` – Vision document
