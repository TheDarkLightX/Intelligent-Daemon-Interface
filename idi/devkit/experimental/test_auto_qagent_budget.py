from __future__ import annotations

from pathlib import Path

from idi.devkit.experimental.auto_qagent import (
    AutoQAgentGoalSpec,
    DEFAULT_REAL_EVAL_EPISODES,
    MAX_REAL_EVAL_EPISODES,
    ObjectiveSpec,
    OutputSpec,
    PackSelectionSpec,
    TrainingBudgetSpec,
    TrainingEnvSpec,
    TrainingSpec,
    _build_objective_indices,
    _aggregate_weighted_metrics,
    _compute_env_weights,
    _compute_eval_episodes,
    _parse_eval_mode,
    _parse_objectives,
    _parse_outputs,
    _parse_packs,
    _parse_profiles,
    _parse_training,
    derive_synth_config,
    run_auto_qagent_synth,
    run_auto_qagent_synth_agentpatches,
)
from idi.synth import AgentPatch, validate_agent_patch, agent_patch_to_dict


def test_compute_eval_episodes_uses_default_for_nonpositive() -> None:
    budget = TrainingBudgetSpec(max_episodes_per_agent=0)
    assert _compute_eval_episodes(budget) == DEFAULT_REAL_EVAL_EPISODES


def test_compute_eval_episodes_clamps_large_values() -> None:
    budget = TrainingBudgetSpec(max_episodes_per_agent=MAX_REAL_EVAL_EPISODES * 10)
    assert _compute_eval_episodes(budget) == MAX_REAL_EVAL_EPISODES


def test_compute_env_weights_empty() -> None:
    assert _compute_env_weights(()) == []


def test_compute_env_weights_normalizes_and_handles_nonpositive() -> None:
    envs = (
        TrainingEnvSpec(id="e1", weight=1.0),
        TrainingEnvSpec(id="e2", weight=3.0),
    )
    weights = _compute_env_weights(envs)
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[1] > weights[0]

    envs_zero = (
        TrainingEnvSpec(id="z1", weight=0.0),
        TrainingEnvSpec(id="z2", weight=-1.0),
    )
    weights_zero = _compute_env_weights(envs_zero)
    assert len(weights_zero) == 2
    assert abs(sum(weights_zero) - 1.0) < 1e-9
    assert abs(weights_zero[0] - weights_zero[1]) < 1e-9


def test_build_objective_indices_real_mode() -> None:
    objectives = (
        ObjectiveSpec(id="avg_reward", direction="maximize"),
        ObjectiveSpec(id="risk_stability", direction="maximize"),
        ObjectiveSpec(id="min_reward", direction="minimize"),
    )
    indices = _build_objective_indices(objectives, "real")
    assert indices == [(0, 1.0), (1, 1.0), (2, -1.0)]


def test_build_objective_indices_ignores_unknown() -> None:
    objectives = (ObjectiveSpec(id="unknown", direction="maximize"),)
    indices = _build_objective_indices(objectives, "real")
    assert indices == []


def test_objective_ranking_lexicographic_real_metrics() -> None:
    objectives = (
        ObjectiveSpec(id="avg_reward", direction="maximize"),
        ObjectiveSpec(id="risk_stability", direction="maximize"),
    )
    indices = _build_objective_indices(objectives, "real")

    metrics_list = [
        (0.5, -0.1, 0.1),
        (0.6, -0.2, 0.05),
        (0.6, -0.05, 0.02),
    ]

    sorted_metrics = sorted(
        metrics_list,
        key=lambda m, idx=indices: tuple(sign * m[i] for i, sign in idx),
        reverse=True,
    )

    assert sorted_metrics[0] == (0.6, -0.05, 0.02)


def _make_synthetic_goal() -> AutoQAgentGoalSpec:
    training = TrainingSpec(
        envs=(),
        curriculum_enabled=False,
        budget=TrainingBudgetSpec(max_episodes_per_agent=0),
    )
    outputs = OutputSpec(num_final_agents=2, bundle_format="wire_v1")
    objectives = (
        ObjectiveSpec(id="complexity", direction="maximize"),
    )
    packs = PackSelectionSpec(include=(), extra=())
    return AutoQAgentGoalSpec(
        agent_family="qagent",
        profiles=("conservative",),
        packs=packs,
        objectives=objectives,
        training=training,
        eval_mode="synthetic",
        outputs=outputs,
    )


def _make_real_goal() -> AutoQAgentGoalSpec:
    training = TrainingSpec(
        envs=(),
        curriculum_enabled=False,
        budget=TrainingBudgetSpec(max_episodes_per_agent=4),
    )
    outputs = OutputSpec(num_final_agents=2, bundle_format="wire_v1")
    objectives = (
        ObjectiveSpec(id="avg_reward", direction="maximize"),
    )
    packs = PackSelectionSpec(include=(), extra=())
    return AutoQAgentGoalSpec(
        agent_family="qagent",
        profiles=("conservative",),
        packs=packs,
        objectives=objectives,
        training=training,
        eval_mode="real",
        outputs=outputs,
    )


def test_auto_qagent_synth_sorts_by_complexity_synthetic() -> None:
    goal = _make_synthetic_goal()
    results = run_auto_qagent_synth(goal)
    assert results, "Auto-QAgent synthetic run should produce candidates"

    # Complexity score is at index 0; ensure non-increasing order.
    scores = [score[0] for _patch, score in results]
    assert scores == sorted(scores, reverse=True)


def test_auto_qagent_agentpatch_export_roundtrip(tmp_path: Path) -> None:
    goal = _make_synthetic_goal()
    patches = run_auto_qagent_synth_agentpatches(goal)
    assert patches, "AgentPatch export should produce at least one patch"

    for ap in patches:
        assert isinstance(ap, AgentPatch)
        validate_agent_patch(ap)


def test_aggregate_weighted_metrics_basic() -> None:
    metrics = [
        (1.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    weights = [0.25, 0.75]
    agg = _aggregate_weighted_metrics(metrics, weights)
    assert len(agg) == 3
    assert abs(agg[0] - 2.5) < 1e-9


def test_aggregate_weighted_metrics_ignores_mismatched_lengths() -> None:
    metrics = [(1.0, 0.0, 0.0)]
    weights = [0.5, 0.5]
    agg = _aggregate_weighted_metrics(metrics, weights)
    assert agg == ()


def test_auto_qagent_synth_sorts_by_avg_reward_real_mode() -> None:
    goal = _make_real_goal()
    results = run_auto_qagent_synth(goal)
    assert results, "Auto-QAgent real-mode run should produce candidates"

    scores = [score[0] for _patch, score in results]
    assert scores == sorted(scores, reverse=True)


def test_auto_qagent_real_mode_agentpatch_export() -> None:
    goal = _make_real_goal()
    patches = run_auto_qagent_synth_agentpatches(goal)
    assert patches, "Real-mode AgentPatch export should produce at least one patch"

    for ap in patches:
        assert isinstance(ap, AgentPatch)
        validate_agent_patch(ap)


def test_helper_parsers_match_from_dict_structures() -> None:
    """Helper parsers should be consistent with from_dict.

    This pins the semantics so refactors can safely split logic without
    changing behavior.
    """
    raw = {
        "agent_family": "qagent",
        "profiles": ["conservative", "research"],
        "packs": {"include": ["qagent_base"], "extra": ["risk_conservative"]},
        "objectives": [
            {"id": "avg_reward", "direction": "maximize"},
            {"id": "risk_stability", "direction": "maximize"},
        ],
        "training": {
            "envs": [
                {"id": "env_a", "weight": 1.0},
                {"id": "env_b", "weight": 3.0},
            ],
            "budget": {
                "max_agents": 64,
                "max_generations": 10,
                "max_episodes_per_agent": 1000,
                "wallclock_hours": 10.0,
            },
        },
        "eval_mode": "synthetic",
        "outputs": {"num_final_agents": 5, "bundle_format": "wire_v1"},
    }

    spec = AutoQAgentGoalSpec.from_dict(raw)

    assert _parse_packs(raw) == spec.packs
    assert _parse_objectives(raw) == spec.objectives
    assert _parse_training(raw) == spec.training
    assert _parse_outputs(raw) == spec.outputs
    assert _parse_profiles(raw) == spec.profiles
    assert _parse_eval_mode(raw) == spec.eval_mode


def test_derive_synth_config_scales_with_budget() -> None:
    """Synth config should scale with budget but remain bounded."""
    small_budget = TrainingBudgetSpec(max_agents=4, max_generations=2)
    medium_budget = TrainingBudgetSpec(max_agents=64, max_generations=10)
    huge_budget = TrainingBudgetSpec(max_agents=10**6, max_generations=1000)

    small_cfg = derive_synth_config(small_budget)
    medium_cfg = derive_synth_config(medium_budget)
    huge_cfg = derive_synth_config(huge_budget)

    # Beam width should be at least 2 and capped to a small constant.
    assert small_cfg.beam_width >= 2
    assert medium_cfg.beam_width >= small_cfg.beam_width
    assert huge_cfg.beam_width == medium_cfg.beam_width

    # Depth should be at least 1 and capped to a small constant.
    assert small_cfg.max_depth >= 1
    assert medium_cfg.max_depth >= small_cfg.max_depth
    assert huge_cfg.max_depth == medium_cfg.max_depth
