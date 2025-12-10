# idi.synth API Reference

This document describes the public, **stable** Python API surface
provided by `idi.synth`. It acts as a façade over several experimental
modules, giving callers a single, future-proof import path.

> For an architectural overview of the modular synth and Auto-QAgent
> stack, see `docs/MODULAR_SYNTH_AND_AUTO_QAGENT.md`.

---

## 1. Importing idi.synth

```python
from idi import synth

# Or import symbols directly
from idi.synth import AgentPatch, QAgentSynthesizer
```

All symbols listed here are re-exported from `idi.synth.__init__`.

---

## 2. AgentPatch Engine

### 2.1 Types

- `AgentPatchMeta`
  - Metadata for patches: `id`, `name`, `description`, `version`,
    `tags: Tuple[str, ...]`.
- `AgentPatch`
  - Generic agent configuration:
    - `meta: AgentPatchMeta`
    - `agent_type: str` (e.g. `"qtable"`)
    - `payload: Mapping[str, Any]`
    - `spec_backend: str` (e.g. `"tau"`)
    - `spec_params: Mapping[str, Any]`
    - `zk_profile: Mapping[str, Any]`

### 2.2 Functions

- `agent_patch_from_dict(data: Mapping[str, Any]) -> AgentPatch`
  - Parse a Python mapping (e.g. decoded JSON) into `AgentPatch`.
- `agent_patch_to_dict(patch: AgentPatch) -> Dict[str, Any]`
  - Convert `AgentPatch` back to a JSON-serializable dict.
- `load_agent_patch(path: Path) -> AgentPatch`
  - Load a patch from `*.agentpatch.json` file.
- `save_agent_patch(patch: AgentPatch, path: Path) -> None`
  - Save a patch to disk (pretty-printed JSON).
- `validate_agent_patch(patch: AgentPatch) -> None`
  - Enforces:
    - Non-empty `id`, `name`, `version`, `agent_type`
    - Bounded lengths for strings
    - Bounded counts for tags, payload/spec/zk keys
  - Raises `ValueError` on violation.
- `diff_agent_patches(a: AgentPatch, b: AgentPatch) -> Dict[str, Any]`
  - Returns a shallow semantic diff mapping field name → `(old, new)`.

---

## 3. QAgent Adapter

These helpers convert between generic `AgentPatch` and QAgent-specific
patches.

- `AGENT_TYPE_QTABLE: str`
  - Canonical `agent_type` for Q-table agents.
- `qagent_patch_to_agent_patch(qpatch: QAgentPatch) -> AgentPatch`
  - Wrap a QAgent patch in generic `AgentPatch` form.
- `agent_patch_to_qagent_patch(patch: AgentPatch) -> QAgentPatch`
  - Extract QAgent parameters from a generic patch.

---

## 4. Generic Synthesizer

These types expose a **planner-agnostic** synthesis interface.

- `AgentSynthConfig`
  - Configuration for a generic agent synthesizer.
- `AgentSynthesizer`
  - High-level interface for synth over `AgentPatch` families.

(Implementation details live under `idi.devkit.experimental.agent_synth`
and may change. The `idi.synth` exports are the stable surface.)

---

## 5. QAgent Synthesizer

- `QAgentSynthConfig`
  - Configuration for QAgent-specific synthesis (beam width, depth,
    etc.).
- `QAgentSynthesizer`
  - Runs bounded beam search over QAgent configurations using KRR-guided
    pruning.
- `load_qagent_patch_preset(name: str) -> QAgentPatch`
  - Load a named preset QAgent configuration.

Typical usage (simplified):

```python
from idi.synth import QAgentSynthConfig, QAgentSynthesizer, qagent_patch_to_agent_patch

base_qpatch = ...  # QAgentPatch
synth = QAgentSynthesizer(base_qpatch, profiles={"conservative"}, evaluator=my_evaluator)
config = QAgentSynthConfig(beam_width=4, max_depth=3)
results = synth.synthesize(config=config)

# Convert to AgentPatch for downstream APIs
agent_patches = [qagent_patch_to_agent_patch(patch) for patch, _score in results]
```

---

## 6. Auto-QAgent API

Although the main Auto-QAgent implementation lives in
`idi.devkit.experimental.auto_qagent`, selected symbols are re-exported
via `idi.synth` for convenience.

### 6.1 Types

- `AutoQAgentGoalSpec`
  - Top-level goal/constraint spec for Auto-QAgent runs.
- `SynthTimeoutError(Exception)`
  - Raised when synth exceeds a wall-clock budget.

### 6.2 Functions

- `load_goal_spec(path: Path) -> AutoQAgentGoalSpec`
  - Load a goal spec JSON file.
- `run_auto_qagent_synth(goal: AutoQAgentGoalSpec, deadline: Optional[float] = None)
  -> List[Tuple[QAgentPatch, Tuple[float, ...]]]`
  - Perform synthesis and return ranked (QAgentPatch, metrics) pairs.
- `run_auto_qagent_synth_agentpatches(goal: AutoQAgentGoalSpec) -> List[AgentPatch]`
  - Convenience wrapper returning `AgentPatch` objects directly.

Typical pattern:

```python
from idi.synth import AutoQAgentGoalSpec, run_auto_qagent_synth_agentpatches
from pathlib import Path
from idi.devkit.experimental.auto_qagent import load_goal_spec

goal = load_goal_spec(Path("examples/synth/conservative_qagent_goal.json"))
patches = run_auto_qagent_synth_agentpatches(goal)
```

---

## 7. Logging & Profiling

### 7.1 Structured Logging

From `idi.devkit.experimental.synth_logging` and re-exported via
`idi.synth`:

- `SynthLogger`
  - Context manager for structured synth logging.
- `SynthRunConfig`
- `SynthRunStats`
- `SynthRunLog`
- `generate_run_id()`

Example:

```python
from idi.synth import SynthLogger, SynthRunConfig, generate_run_id

run_id = generate_run_id()
config = SynthRunConfig(
    beam_width=4,
    max_depth=3,
    profiles=("conservative",),
    eval_mode="synthetic",
)

with SynthLogger(run_id, config, emit_json=True) as slog:
    results = run_auto_qagent_synth(goal)
    # Optionally record stats or candidates
```

### 7.2 Profiling Utilities

While not re-exported from `idi.synth` today, profiling helpers live in
`idi.devkit.experimental.synth_profiler` and are worth knowing about:

- `SynthProfiler` – context manager for profiling
- `profile_synth_run(fn, ...)` – run and profile a synth function
- `benchmark_evaluator(evaluator, patches)` – benchmark an evaluator

---

## 8. Stability Notes

- `idi.synth` is the **preferred import path** for all synth-related
  APIs in production code.
- Experimental modules under `idi.devkit.experimental.*` may change
  without notice; they are implementation details.
- New features should be added behind `idi.synth` re-exports so that
  callers do not need to track internal refactors.
