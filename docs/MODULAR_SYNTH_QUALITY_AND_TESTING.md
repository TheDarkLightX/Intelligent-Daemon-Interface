# Modular Synth Quality, Invariants, and Testing

This document summarizes the **quality strategy** for the modular synth
and Auto-QAgent stack:

- Formal invariants and KRR/Tau alignment
- Property-based, fuzz, and integration testing
- How to run only synth-related tests
- How to interpret logs and profiling output

---

## 1. Formal Invariants

The system enforces a small, explicit set of invariants for QAgent
patches. These are mirrored across KRR packs, Tau specs, and Python
validation.

### I1: State Size Bound

> `state_cells = num_price_bins × num_inventory_bins ≤ MAX_STATE_CELLS`

- **Why:** Prevents Q-table state explosion and ensures tractable search
  and training.
- **Enforced by:**
  - KRR: `build_qagent_base_pack(max_state_cells=...)`
  - Tau: `tau_spec_generator.py` (`invariant_state_size_bound`)
  - Python: `validate_patch_against_spec` and
    `AutoQAgentGoalSpec.from_dict` bounds

### I2: Minimum Discount Factor

> `discount_factor ≥ MIN_DISCOUNT`

- **Why:** Encourages long-term reward consideration and stable
  convergence.
- **Enforced by:**
  - KRR: `discount_constraint` in `qagent_base_pack`
  - Tau: `invariant_discount_bound`

### I3: Maximum Learning Rate (Conservative)

> `learning_rate ≤ MAX_LEARNING_RATE`

- **Why:** Prevents unstable updates in conservative profiles.
- **Enforced by:**
  - KRR: `risk_conservative_pack`
  - Tau: `invariant_learning_rate_bound`

### I4: Exploration Bound

> `epsilon_start ≤ MAX_EXPLORATION`

- **Why:** Prevents the agent from being purely random for too long.
- **Enforced by:**
  - Tau: `invariant_exploration_bound`
  - Python: `AutoQAgentGoalSpec.from_dict` clamping

### I5: Exploration Decay Validity

> `epsilon_end ≤ epsilon_start`

- **Why:** Ensures exploration decays or stays constant, not increasing
  over time.
- **Enforced by:**
  - Tau: `invariant_exploration_decay_valid`
  - Python: `validate_patch_against_spec`

---

## 2. KRR & STRIKE Tests

`test_krr_formal_spec.py` exercises the STRIKE/IKL layer:

- **State size invariant (I1)**
  - `TestStateSizeInvariant` verifies:
    - Valid state sizes pass
    - Excessive state sizes are rejected
    - Boundary case (exactly at limit) passes
- **Discount invariant (I2)**
  - `TestDiscountInvariant` checks discounts above/below threshold
- **Profile-specific constraints**
  - `TestProfileConstraints` ensures conservative profiles reject
    overly aggressive learning rates
- **Beam search + KRR integration**
  - `TestKRRGuidedBeamSearch` confirms that
    `krr_guided_beam_search` only returns KRR-valid candidates and
    properly prunes invalid ones
- **Pack selection & linting**
  - `TestPackSelection` and `TestPackLinting` validate pack metadata and
    selection based on tags and profiles
- **Fixpoint closure**
  - `TestStrikeClosure` checks that derived facts and constraints behave
    as expected

---

## 3. Auto-QAgent Integration Tests

`test_auto_qagent_integration.py` covers end-to-end behavior:

- **Synthetic mode**
  - `TestSyntheticModeIntegration`
    - Runs full Auto-QAgent flow in synthetic mode
    - Asserts on result structure and sorting by objectives
    - Validates AgentPatch exports via `validate_agent_patch`
- **Real mode (mocked)**
  - `TestRealModeMockedIntegration`
    - Mocks `evaluate_patch_real` to:
      - Verify evaluator integration
      - Check multi-env metric aggregation
      - Confirm `_safe_evaluate_patch_real` falls back safely on
        exceptions or degenerate metrics
- **Timeout & budget**
  - `TestTimeoutAndBudget`
    - Confirms `SynthTimeoutError` is raised when deadline is exceeded
    - Verifies clamping of extreme budget values
- **Goal spec loading & robustness**
  - `TestGoalSpecLoading`
    - Checks correct parsing from file
    - Ensures invalid/non-object JSON raises appropriate errors
- **End-to-end invariants**
  - `TestEndToEndInvariants`
    - Ensures empty objectives/envs do not crash
    - Confirms AgentPatch round-trip save/load works via CLI helpers

---

## 4. Tau Spec Tests

`test_tau_spec_generator.py` verifies that Tau spec generation and local
validation are consistent:

- **Spec content**
  - Confirms parameters and invariant definitions appear in the
    generated `.tau` text
- **Warnings**
  - Ensures invalid patches produce meaningful warnings
- **Validation**
  - Checks that `validate_patch_against_spec` passes/fails exactly when
    invariants are satisfied/violated
- **File I/O**
  - Tests `save_tau_spec` creates directories and writes correct
    contents

---

## 5. Profiling & Performance Tests

`test_synth_profiler.py` ensures profiling utilities behave as expected:

- **TimingRecord & ProfileStats**
  - Validate duration calculations and aggregation
- **SynthProfiler**
  - Confirms context manager records operations and timelines
  - Verifies metadata recording
- **ProfileReport**
  - Tests `summary()` and `to_dict()` representations
- **Helper functions**
  - `profile_synth_run` correctly profiles a function call
  - `benchmark_evaluator` records multiple evaluator calls
  - `quick_profile` prints timing information to stderr

These tests guarantee that performance analysis tools remain reliable as
code evolves.

---

## 6. Property-Based & Fuzz Testing

Synth-related property and fuzz tests live in:

- `test_agent_patch_pbt.py`
  - Property-based tests for AgentPatch round-trip and validation
- `test_beam_search_pbt.py`
  - Property tests for `krr_guided_beam_search`:
    - Determinism
    - Monotonicity with respect to beam width/depth
    - No duplicate candidates in results
- `test_goal_spec_fuzz.py`
  - Fuzzing `AutoQAgentGoalSpec.from_dict` with adversarial JSON to
    ensure robustness and input hardening

These tests help catch edge cases that are hard to enumerate manually.

---

## 7. How to Run Synth-Related Tests

From the repository root:

```bash
# Run all synth-related tests
pytest idi/devkit/experimental/test_*synth* idi/devkit/experimental/test_*qagent* -v

# Run specific suites
pytest idi/devkit/experimental/test_auto_qagent_integration.py -v
pytest idi/devkit/experimental/test_krr_formal_spec.py -v
pytest idi/devkit/experimental/test_tau_spec_generator.py -v
pytest idi/devkit/experimental/test_synth_profiler.py -v

# Property-based and fuzz tests
pytest idi/devkit/experimental/test_agent_patch_pbt.py -v
pytest idi/devkit/experimental/test_beam_search_pbt.py -v
pytest idi/devkit/experimental/test_goal_spec_fuzz.py -v
```

You can integrate these commands into CI to enforce quality gates before
merging synth-related changes.

---

## 8. Interpreting Logs and Profiles

- **Structured logs (`SynthLogger`)**
  - Look at `run_id`, `beam_width`, `max_depth`, `profiles`, and
    `duration_seconds` to understand high-level behavior.
- **Profiling (`SynthProfiler`)**
  - Use `ProfileReport.summary()` to find hot spots:
    - If `krr_eval` dominates, consider simplifying packs or reducing
      beam width
    - If `patch_eval` (QTrainer) dominates, adjust budgets or use
      synthetic eval mode for fast iterations

These tools make it easier to both **validate correctness** and
**optimize performance** without changing core logic.
