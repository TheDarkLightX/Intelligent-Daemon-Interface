"""Auto-QAgent automation layer (experimental).

.. deprecated::
    This module is in `idi.devkit.experimental` and may change without
    notice. The Auto-QAgent API is still evolving; use with caution in
    production workflows.

    Key types and functions:
    - AutoQAgentGoalSpec: Goal specification for automated synthesis.
    - run_auto_qagent_synth: Run synthesis and return ranked candidates.
    - run_auto_qagent_synth_agentpatches: Return candidates as AgentPatch.

    For the stable AgentPatch API, import from `idi.synth`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import json

logger = logging.getLogger(__name__)

from idi.devkit.experimental.qagent_synth import QAgentSynthConfig, QAgentSynthesizer
from idi.devkit.experimental.sape_q_patch import (
    QAgentPatch,
    QPatchMeta,
    evaluate_patch_real,
    evaluate_patch_stub,
)


DEFAULT_REAL_EVAL_EPISODES = 16
MAX_REAL_EVAL_EPISODES = 1024

# ---------------------------------------------------------------------------
# Validation bounds (DoS protection)
# ---------------------------------------------------------------------------

MAX_OBJECTIVES = 16
MAX_ENVS = 16
MAX_PROFILES = 16
MAX_PACKS = 32
MAX_AGENTS_LIMIT = 1024
MAX_GENERATIONS_LIMIT = 64
MAX_EPISODES_LIMIT = 10_000
MAX_WALLCLOCK_HOURS = 168.0  # 1 week
MAX_STRING_LENGTH = 256


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clamp_float(value: float, low: float, high: float) -> float:
    """Clamp a float to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


REAL_METRIC_INDEX: Dict[str, int] = {
    "avg_reward": 0,
    "risk_stability": 1,
    "min_reward": 2,
}

SYNTHETIC_METRIC_INDEX: Dict[str, int] = {
    "complexity": 0,
}


ObjectiveDirection = Literal["maximize", "minimize"]
EvalMode = Literal["synthetic", "real"]


def _validated_direction(value: Any) -> ObjectiveDirection:
    """Validate and normalize direction to Literal type.

    Raises:
        ValueError: If direction is not 'maximize' or 'minimize'.
    """
    s = str(value).lower().strip()
    if s in ("maximize", "max"):
        return "maximize"
    if s in ("minimize", "min"):
        return "minimize"
    raise ValueError(f"Invalid objective direction: {value!r}")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Single optimization objective for Auto-QAgent.

    Examples:
    - avg_reward (maximize)
    - risk_stability (maximize)
    - complexity (minimize)
    """

    id: str
    direction: ObjectiveDirection


@dataclass(frozen=True)
class TrainingEnvSpec:
    """Training environment configuration with optional weight."""

    id: str
    weight: float = 1.0


@dataclass(frozen=True)
class TrainingBudgetSpec:
    """Resource limits for automated training runs.

    All values are clamped during parsing to prevent DoS:
    - max_agents: [1, MAX_AGENTS_LIMIT]
    - max_generations: [1, MAX_GENERATIONS_LIMIT]
    - max_episodes_per_agent: [1, MAX_EPISODES_LIMIT]
    - wallclock_hours: [0.01, MAX_WALLCLOCK_HOURS]
    """

    max_agents: int = 16
    max_generations: int = 4
    max_episodes_per_agent: int = 512
    wallclock_hours: float = 1.0


@dataclass(frozen=True)
class TrainingSpec:
    """Training configuration for Auto-QAgent."""

    envs: Tuple[TrainingEnvSpec, ...]
    curriculum_enabled: bool = True
    budget: TrainingBudgetSpec = TrainingBudgetSpec()


@dataclass(frozen=True)
class OutputSpec:
    """Output preferences for Auto-QAgent runs."""

    num_final_agents: int = 3
    bundle_format: str = "wire_v1"


@dataclass(frozen=True)
class PackSelectionSpec:
    """Knowledge pack selection for Auto-QAgent."""

    include: Tuple[str, ...]
    extra: Tuple[str, ...] = ()


def _parse_packs(data: Dict[str, Any]) -> PackSelectionSpec:
    """Parse packs section from raw goal spec data."""
    packs = data.get("packs", {}) or {}
    include_raw = list(packs.get("include", ()) or ())[:MAX_PACKS]
    extra_raw = list(packs.get("extra", ()) or ())[:MAX_PACKS]
    return PackSelectionSpec(
        include=tuple(str(p)[:MAX_STRING_LENGTH] for p in include_raw),
        extra=tuple(str(p)[:MAX_STRING_LENGTH] for p in extra_raw),
    )


def _parse_objectives(data: Dict[str, Any]) -> Tuple[ObjectiveSpec, ...]:
    """Parse objectives section from raw goal spec data."""
    objectives_raw = list(data.get("objectives", []) or [])[:MAX_OBJECTIVES]
    objectives_list: List[ObjectiveSpec] = []
    for o in objectives_raw:
        if not isinstance(o, dict):
            continue
        try:
            direction = _validated_direction(o.get("direction", "maximize"))
        except ValueError:
            # Skip objectives with invalid direction
            continue
        objectives_list.append(
            ObjectiveSpec(
                id=str(o.get("id", ""))[:MAX_STRING_LENGTH],
                direction=direction,
            )
        )
    return tuple(objectives_list)


def _parse_training(data: Dict[str, Any]) -> TrainingSpec:
    """Parse training section from raw goal spec data."""
    training_cfg = (data.get("training", {}) or {})

    envs_raw = list(training_cfg.get("envs", []) or [])[:MAX_ENVS]
    envs = tuple(
        TrainingEnvSpec(
            id=str(e.get("id", ""))[:MAX_STRING_LENGTH],
            weight=_clamp_float(float(e.get("weight", 1.0)), 0.0, 1000.0),
        )
        for e in envs_raw
        if isinstance(e, dict)
    )

    budget_cfg = training_cfg.get("budget", {}) or {}
    budget = TrainingBudgetSpec(
        max_agents=_clamp(int(budget_cfg.get("max_agents", 16)), 1, MAX_AGENTS_LIMIT),
        max_generations=_clamp(
            int(budget_cfg.get("max_generations", 4)),
            1,
            MAX_GENERATIONS_LIMIT,
        ),
        max_episodes_per_agent=_clamp(
            int(budget_cfg.get("max_episodes_per_agent", 512)),
            1,
            MAX_EPISODES_LIMIT,
        ),
        wallclock_hours=_clamp_float(
            float(budget_cfg.get("wallclock_hours", 1.0)),
            0.01,
            MAX_WALLCLOCK_HOURS,
        ),
    )

    curriculum_cfg = training_cfg.get("curriculum", {}) or {}
    curriculum_enabled = bool(curriculum_cfg.get("enabled", True))

    return TrainingSpec(
        envs=envs,
        curriculum_enabled=curriculum_enabled,
        budget=budget,
    )


def _parse_outputs(data: Dict[str, Any]) -> OutputSpec:
    """Parse outputs section from raw goal spec data."""
    outputs_cfg = data.get("outputs", {}) or {}
    num_final_agents = _clamp(int(outputs_cfg.get("num_final_agents", 3)), 1, MAX_AGENTS_LIMIT)
    bundle_format = str(outputs_cfg.get("bundle_format", "wire_v1"))[:MAX_STRING_LENGTH]
    return OutputSpec(num_final_agents=num_final_agents, bundle_format=bundle_format)


def _parse_profiles(data: Dict[str, Any]) -> Tuple[str, ...]:
    """Parse profiles section from raw goal spec data."""
    profiles_raw = list(data.get("profiles", ()) or ())[:MAX_PROFILES]
    return tuple(str(p)[:MAX_STRING_LENGTH] for p in profiles_raw)


def _parse_eval_mode(data: Dict[str, Any]) -> EvalMode:
    """Parse eval_mode from raw goal spec data."""
    eval_mode_str = str(data.get("eval_mode", "synthetic"))
    if eval_mode_str not in ("synthetic", "real"):
        raise ValueError(f"Unsupported eval_mode: {eval_mode_str}")
    return eval_mode_str  # type: ignore[return-value]


@dataclass(frozen=True)
class AutoQAgentGoalSpec:
    """Top-level goal/constraint specification for Auto-QAgent.

    This is a stable, JSON-serializable structure that can be authored
    by humans or higher-level tools. It does not perform any heavy
    lifting itself; instead it guides the Synth + SAPE + training
    pipeline.
    """

    agent_family: str
    profiles: Tuple[str, ...]
    packs: PackSelectionSpec
    objectives: Tuple[ObjectiveSpec, ...]
    training: TrainingSpec
    eval_mode: EvalMode = "synthetic"
    outputs: OutputSpec = OutputSpec()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoQAgentGoalSpec":
        packs_spec = _parse_packs(data)
        objectives = _parse_objectives(data)
        training = _parse_training(data)
        outputs = _parse_outputs(data)
        profiles = _parse_profiles(data)
        eval_mode = _parse_eval_mode(data)
        agent_family = str(data.get("agent_family", "qagent"))[:MAX_STRING_LENGTH]
        return cls(
            agent_family=agent_family,
            profiles=profiles,
            packs=packs_spec,
            objectives=objectives,
            training=training,
            eval_mode=eval_mode,
            outputs=outputs,
        )


def derive_synth_config(budget: TrainingBudgetSpec) -> QAgentSynthConfig:
    """Derive QAgentSynthConfig from a training budget.

    Beam width and depth are scaled from budget parameters but remain
    bounded to keep search tractable.
    """
    raw_beam = max(2, budget.max_agents // 4)
    beam_width = _clamp(raw_beam, 2, 16)

    raw_depth = max(1, budget.max_generations)
    max_depth = _clamp(raw_depth, 1, 8)

    return QAgentSynthConfig(beam_width=beam_width, max_depth=max_depth)


def load_goal_spec(path: Path) -> AutoQAgentGoalSpec:
    """Load an AutoQAgentGoalSpec from a JSON file."""

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):  # pragma: no cover - defensive
        raise ValueError("Goal spec must be a JSON object at top level")
    return AutoQAgentGoalSpec.from_dict(raw)


def _compute_eval_episodes(budget: TrainingBudgetSpec) -> int:
    episodes = int(budget.max_episodes_per_agent)
    if episodes <= 0:
        return DEFAULT_REAL_EVAL_EPISODES
    if episodes > MAX_REAL_EVAL_EPISODES:
        return MAX_REAL_EVAL_EPISODES
    return episodes


def _compute_env_weights(envs: Tuple[TrainingEnvSpec, ...]) -> List[float]:
    if not envs:
        return []

    weights: List[float] = []
    for env in envs:
        w = float(env.weight)
        if w < 0.0:
            w = 0.0
        weights.append(w)

    total = sum(weights)
    if total <= 0.0:
        if not weights:
            return []
        equal = 1.0 / float(len(weights))
        return [equal for _ in weights]

    return [w / total for w in weights]


def _aggregate_weighted_metrics(
    metrics_list: List[Tuple[float, ...]],
    weights: List[float],
) -> Tuple[float, ...]:
    """Aggregate metrics from multiple environments using weighted average.

    Preconditions:
        - metrics_list and weights have the same length.
    Postconditions:
        - Returns tuple of weighted averages, using minimum tuple length.
    """
    if not metrics_list or not weights or len(metrics_list) != len(weights):
        return ()

    # FIX: Use minimum length across all tuples to avoid IndexError
    length = min(len(m) for m in metrics_list) if metrics_list else 0
    if length == 0:
        return ()

    agg: List[float] = []
    for i in range(length):
        value = 0.0
        for (m, w) in zip(metrics_list, weights):
            value += w * m[i]
        agg.append(value)
    return tuple(agg)


# Default fallback metrics for failed evaluations
FALLBACK_REAL_METRICS: Tuple[float, ...] = (-1e6, -1e6, -1e6)
FALLBACK_SYNTHETIC_METRICS: Tuple[float, ...] = (-1e6, -1e6, -1e6, -1e6)


def _safe_evaluate_patch_real(
    patch: QAgentPatch,
    episodes: int,
    seed: int = 0,
) -> Tuple[float, ...]:
    """Evaluate a patch with graceful failure handling.

    If QTrainer fails (exception, no rewards), returns fallback metrics
    instead of crashing. This ensures synth can continue with remaining
    candidates.
    """
    try:
        result = evaluate_patch_real(patch, episodes, seed)
        if not result or all(v == 0.0 for v in result):
            logger.warning(
                "QTrainer returned degenerate metrics for patch %s",
                patch.identifier,
            )
            return FALLBACK_REAL_METRICS
        return result
    except Exception as exc:
        logger.warning(
            "QTrainer failed for patch %s: %s",
            patch.identifier,
            str(exc),
        )
        return FALLBACK_REAL_METRICS


def _make_evaluator(goal: AutoQAgentGoalSpec):
    if goal.eval_mode == "synthetic":
        return evaluate_patch_stub

    episodes = _compute_eval_episodes(goal.training.budget)
    envs = tuple(goal.training.envs)
    env_weights = _compute_env_weights(envs)

    if not env_weights:
        return lambda p, episodes=episodes: _safe_evaluate_patch_real(p, episodes)

    max_envs = min(len(envs), len(env_weights), 4)
    envs = envs[:max_envs]
    env_weights = env_weights[:max_envs]

    def eval_with_envs(patch: QAgentPatch, episodes=episodes, _envs=envs, _weights=env_weights):
        per_env_metrics: List[Tuple[float, ...]] = []
        active_weights: List[float] = []  # FIX: Track weights for non-skipped envs
        for idx, (env, w) in enumerate(zip(_envs, _weights)):
            if w <= 0.0:
                continue
            seed = idx
            m = _safe_evaluate_patch_real(patch, episodes, seed)
            per_env_metrics.append(m)
            active_weights.append(w)  # FIX: Collect corresponding weight

        if not per_env_metrics:
            return _safe_evaluate_patch_real(patch, episodes)

        # FIX: Normalize active_weights to sum to 1.0
        total = sum(active_weights)
        if total > 0:
            active_weights = [aw / total for aw in active_weights]

        aggregated = _aggregate_weighted_metrics(per_env_metrics, active_weights)
        return aggregated

    return eval_with_envs


def _build_objective_indices(
    objectives: Tuple[ObjectiveSpec, ...],
    eval_mode: EvalMode,
) -> List[Tuple[int, float]]:
    if not objectives:
        return []

    if eval_mode == "real":
        mapping = REAL_METRIC_INDEX
    else:
        mapping = SYNTHETIC_METRIC_INDEX

    indices: List[Tuple[int, float]] = []
    for obj in objectives:
        metric_idx = mapping.get(obj.id)
        if metric_idx is None:
            continue
        direction = 1.0 if obj.direction == "maximize" else -1.0
        indices.append((metric_idx, direction))
    return indices


class SynthTimeoutError(Exception):
    """Raised when synth exceeds wallclock budget."""
    pass


def run_auto_qagent_synth(
    goal: AutoQAgentGoalSpec,
    deadline: Optional[float] = None,
) -> List[Tuple[QAgentPatch, Tuple[float, ...]]]:
    """Run a minimal Auto-QAgent synthesis using QAgentSynthesizer.

    This function currently focuses on the design/synthesis phase.
    Evaluation mode is selected via goal.eval_mode:
    - synthetic: fast, deterministic synthetic metric.
    - real: QTrainer-based backtest using a short training run.

    Args:
        goal: The Auto-QAgent goal specification.
        deadline: Optional absolute deadline (time.time()) for the run.
            If not provided, uses goal.training.budget.wallclock_hours.

    Returns:
        List of (QAgentPatch, metrics) tuples, sorted by objectives.

    Raises:
        SynthTimeoutError: If wallclock budget is exceeded.
    """

    if goal.agent_family != "qagent":
        raise ValueError(f"Unsupported agent_family: {goal.agent_family}")

    # Compute deadline from budget if not provided
    if deadline is None:
        wallclock_seconds = goal.training.budget.wallclock_hours * 3600.0
        deadline = time.time() + wallclock_seconds

    def check_deadline() -> None:
        if time.time() > deadline:
            raise SynthTimeoutError(
                f"Synth exceeded wallclock budget of {goal.training.budget.wallclock_hours}h"
            )

    check_deadline()

    meta = QPatchMeta(
        name="auto-qagent-base",
        description="Base patch for Auto-QAgent synthesis",
        version="0.0.1",
        tags=("qagent", "auto", "experimental"),
    )
    base = QAgentPatch(
        identifier="base",
        num_price_bins=10,
        num_inventory_bins=10,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=1000,
        meta=meta,
    )

    profiles = set(goal.profiles) or {"conservative"}
    synth_cfg = derive_synth_config(goal.training.budget)

    # Wrap evaluator with deadline check
    base_evaluator = _make_evaluator(goal)

    def timed_evaluator(patch: QAgentPatch) -> Tuple[float, ...]:
        check_deadline()
        return base_evaluator(patch)

    synth = QAgentSynthesizer(base, profiles=profiles, evaluator=timed_evaluator)

    try:
        results = synth.synthesize(config=synth_cfg)
    except SynthTimeoutError:
        logger.warning("Synth timed out; returning partial results")
        results = []

    check_deadline()

    indices = _build_objective_indices(goal.objectives, goal.eval_mode)
    if not indices:
        return results

    def sort_key(item: Tuple[QAgentPatch, Tuple[float, ...]]) -> Tuple[float, ...]:
        metrics = item[1]
        result: List[float] = []
        for idx, sign in indices:
            # FIX: Guard against index out of bounds
            if idx < len(metrics):
                result.append(sign * metrics[idx])
            else:
                # Use -inf for maximize, +inf for minimize to rank last
                result.append(float("-inf") if sign > 0 else float("inf"))
        return tuple(result)

    return sorted(results, key=sort_key, reverse=True)


def run_auto_qagent_synth_agentpatches(goal: AutoQAgentGoalSpec) -> List["AgentPatch"]:
    from idi.synth import AgentPatch, qagent_patch_to_agent_patch

    results = run_auto_qagent_synth(goal)
    patches: List[AgentPatch] = []
    for qpatch, _score in results:
        patches.append(qagent_patch_to_agent_patch(qpatch))
    return patches
