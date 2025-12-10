from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from idi.devkit.experimental.strike_krr import KnowledgeBase, evaluate_with_krr
from idi.devkit.experimental.synth_krr_planner import build_kb_for_qpatch


@dataclass(frozen=True)
class QPatchMeta:
    """Metadata for experimental patches, aligned with AgentPatch schema."""

    name: str
    description: str
    version: str
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QAgentPatch:
    """Minimal experimental patch for a Q-table style agent.

    Design by Contract:
    - Preconditions:
      - All numeric fields are finite.
      - Bin counts are positive integers.
    - Invariants:
      - epsilon_start in [0, 1], epsilon_end in [0, 1].
      - epsilon_decay_steps > 0.
      - 0 < learning_rate <= 1.
      - 0 < discount_factor <= 1.
    """

    identifier: str
    num_price_bins: int
    num_inventory_bins: int
    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    meta: QPatchMeta = field(
        default_factory=lambda: QPatchMeta(
            name="unnamed",
            description="experimental QAgentPatch",
            version="0.0.0",
            tags=("experimental",),
        )
    )


@dataclass(frozen=True)
class QPatchMetrics:
    """Metrics associated with a QAgentPatch.

    Examples of metrics:
    - average_reward: higher is better.
    - negative_max_drawdown: higher is better (less drawdown).
    """

    patch: QAgentPatch
    values: Tuple[float, ...]


def qpatch_to_ikl_facts(patch: QAgentPatch) -> Dict[str, Set[Tuple]]:
    facts: Dict[str, Set[Tuple]] = {
        "patch": {(patch.identifier,)},
        "param_value": {
            (patch.identifier, "num_price_bins", patch.num_price_bins),
            (patch.identifier, "num_inventory_bins", patch.num_inventory_bins),
            (patch.identifier, "learning_rate", patch.learning_rate),
            (patch.identifier, "discount_factor", patch.discount_factor),
            (patch.identifier, "epsilon_start", patch.epsilon_start),
            (patch.identifier, "epsilon_end", patch.epsilon_end),
            (patch.identifier, "epsilon_decay_steps", patch.epsilon_decay_steps),
        },
    }
    return facts


def check_patch_with_krr(
    patch: QAgentPatch,
    kb: KnowledgeBase | None = None,
    max_state_cells: int = 256,
    min_discount: float = 0.9,
) -> Tuple[bool, List[str]]:
    profiles: Set[str] = {"conservative"}
    if kb is None:
        kb = build_kb_for_qpatch(
            patch.meta,
            requested_profiles=profiles,
        )
    base_facts = qpatch_to_ikl_facts(patch)
    params = {"max_state_cells": max_state_cells, "min_discount": min_discount}
    allowed, reasons = evaluate_with_krr(kb, base_facts, params, active_profiles=profiles)
    return allowed, reasons


def validate_patch_fast(patch: QAgentPatch) -> bool:
    """Lightweight structural validation for QAgentPatch.

    Preconditions:
        - patch is a well-formed QAgentPatch instance.
    Postconditions:
        - Returns True iff basic type/range constraints hold.
    """
    if patch.num_price_bins <= 0:
        return False
    if patch.num_inventory_bins <= 0:
        return False
    if not (0.0 < patch.learning_rate <= 1.0):
        return False
    if not (0.0 < patch.discount_factor <= 1.0):
        return False
    if not (0.0 <= patch.epsilon_start <= 1.0):
        return False
    if not (0.0 <= patch.epsilon_end <= 1.0):
        return False
    if patch.epsilon_decay_steps <= 0:
        return False
    return True


def check_spec_stub(patch: QAgentPatch) -> bool:
    """Experimental stand-in for a Tau-backed spec check.

    This function is deliberately conservative and should be replaced with
    a real Tau integration.

    Current invariants enforced:
    - Structural validity (validate_patch_fast).
    - Bounded state space size to limit Q-table dimensionality.
    - Simple discount floor via IKL/STRIKE KRR.
    """
    if not validate_patch_fast(patch):
        return False

    # Simple spec-style constraint: cap total state cells to keep the
    # underlying Q-table and Tau traces tractable in experiments.
    max_state_cells = 256
    state_cells = patch.num_price_bins * patch.num_inventory_bins
    if state_cells > max_state_cells:
        return False

    # Delegate additional checks to the experimental KRR layer.
    allowed, _reasons = check_patch_with_krr(
        patch,
        max_state_cells=max_state_cells,
        min_discount=0.9,
    )
    if not allowed:
        return False

    return True


def mutate_patch(patch: QAgentPatch) -> Sequence[QAgentPatch]:
    """Generate a small neighborhood of patches around the given patch.

    This is an experimental, local mutation operator. It avoids
    large jumps to keep search stable.

    Postconditions:
        - All returned patches are structurally valid or will be
          rejected by validate_patch_fast.
    """
    step_lr = 0.1 * patch.learning_rate
    if step_lr <= 0.0:
        step_lr = 0.01

    candidates: List[QAgentPatch] = []

    def clamp(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value

    # Learning rate up/down
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-lr+",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=clamp(patch.learning_rate + step_lr, 1e-4, 1.0),
            discount_factor=patch.discount_factor,
            epsilon_start=patch.epsilon_start,
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-lr-",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=clamp(patch.learning_rate - step_lr, 1e-4, 1.0),
            discount_factor=patch.discount_factor,
            epsilon_start=patch.epsilon_start,
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )

    # Epsilon start tweaks
    eps_step = 0.1
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-eps+",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=patch.learning_rate,
            discount_factor=patch.discount_factor,
            epsilon_start=clamp(patch.epsilon_start + eps_step, 0.0, 1.0),
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-eps-",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=patch.learning_rate,
            discount_factor=patch.discount_factor,
            epsilon_start=clamp(patch.epsilon_start - eps_step, 0.0, 1.0),
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )

    # Price bin count mutations
    for delta in (-2, 2):
        candidates.append(
            QAgentPatch(
                identifier=f"{patch.identifier}-pb{delta:+d}",
                num_price_bins=max(1, patch.num_price_bins + delta),
                num_inventory_bins=patch.num_inventory_bins,
                learning_rate=patch.learning_rate,
                discount_factor=patch.discount_factor,
                epsilon_start=patch.epsilon_start,
                epsilon_end=patch.epsilon_end,
                epsilon_decay_steps=patch.epsilon_decay_steps,
            )
        )

    # Inventory bin count mutations
    for delta in (-2, 2):
        candidates.append(
            QAgentPatch(
                identifier=f"{patch.identifier}-ib{delta:+d}",
                num_price_bins=patch.num_price_bins,
                num_inventory_bins=max(1, patch.num_inventory_bins + delta),
                learning_rate=patch.learning_rate,
                discount_factor=patch.discount_factor,
                epsilon_start=patch.epsilon_start,
                epsilon_end=patch.epsilon_end,
                epsilon_decay_steps=patch.epsilon_decay_steps,
            )
        )

    # Discount factor tweaks
    df_step = 0.02
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-df+",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=patch.learning_rate,
            discount_factor=clamp(patch.discount_factor + df_step, 0.8, 0.999),
            epsilon_start=patch.epsilon_start,
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )
    candidates.append(
        QAgentPatch(
            identifier=f"{patch.identifier}-df-",
            num_price_bins=patch.num_price_bins,
            num_inventory_bins=patch.num_inventory_bins,
            learning_rate=patch.learning_rate,
            discount_factor=clamp(patch.discount_factor - df_step, 0.8, 0.999),
            epsilon_start=patch.epsilon_start,
            epsilon_end=patch.epsilon_end,
            epsilon_decay_steps=patch.epsilon_decay_steps,
        )
    )

    return [c for c in candidates if validate_patch_fast(c)]


def evaluate_patch_stub(patch: QAgentPatch) -> Tuple[float, ...]:
    """Experimental synthetic evaluation for QAgentPatch.

    This function is kept for comparison and testing purposes. It
    implements a deterministic synthetic metric vector that rewards:
    - state complexity near 100 cells,
    - learning_rate near 0.1,
    - epsilon_start near 0.2,
    - discount_factor near 0.99.
    """
    state_complexity = float(patch.num_price_bins * patch.num_inventory_bins)
    complexity_score = -abs(state_complexity - 100.0)

    lr_score = -abs(patch.learning_rate - 0.1)
    eps_score = -abs(patch.epsilon_start - 0.2)
    df_score = -abs(patch.discount_factor - 0.99)

    return (complexity_score, lr_score, eps_score, df_score)


def _build_training_config_from_patch(episodes: int | None = None) -> "TrainingConfig":
    """Create a small TrainingConfig tuned for fast evaluation.

    This helper isolates the dependency on idi_iann.config and keeps
    evaluation focused on a short training run. It intentionally does
    not depend on QAgentPatch so that we can later add mappings in a
    single place without changing the TrainingConfig creation logic.
    """
    from idi.training.python.idi_iann.config import TrainingConfig

    base_episodes = 16
    if episodes is not None and episodes > 0:
        base_episodes = episodes

    # Use a small number of episodes and short episode length to keep
    # evaluations fast and bounded.
    return TrainingConfig(episodes=base_episodes, episode_length=32)


def evaluate_patch_real(
    patch: QAgentPatch,
    episodes_override: int | None = None,
    seed_override: int | None = 0,
) -> Tuple[float, ...]:
    """Evaluate a patch using the real QTrainer on a synthetic market.

    This function maps QAgentPatch parameters into a TrainingConfig,
    runs a short Q-learning session, and summarizes performance using
    simple reward statistics.
    """
    from idi.training.python.idi_iann.config import QuantizerConfig, TrainingConfig
    from idi.training.python.idi_iann.trainer import QTrainer

    base_cfg: TrainingConfig = _build_training_config_from_patch(episodes_override)

    quantizer = QuantizerConfig(
        price_buckets=patch.num_price_bins,
        volume_buckets=patch.num_inventory_bins,
        trend_buckets=base_cfg.quantizer.trend_buckets,
        scarcity_buckets=base_cfg.quantizer.scarcity_buckets,
        mood_buckets=base_cfg.quantizer.mood_buckets,
    )

    # Approximate an exploration decay that moves epsilon from start to
    # end over the configured number of episodes, when possible.
    exploration_decay = base_cfg.exploration_decay
    if (
        patch.epsilon_start > 0.0
        and patch.epsilon_end > 0.0
        and patch.epsilon_end < patch.epsilon_start
    ):
        episodes = float(base_cfg.episodes)
        try:
            exploration_decay = (patch.epsilon_end / patch.epsilon_start) ** (1.0 / episodes)
        except (ZeroDivisionError, OverflowError, ValueError):
            exploration_decay = base_cfg.exploration_decay

    cfg = TrainingConfig(
        episodes=base_cfg.episodes,
        episode_length=base_cfg.episode_length,
        discount=patch.discount_factor,
        learning_rate=patch.learning_rate,
        exploration_decay=exploration_decay,
        quantizer=quantizer,
        rewards=base_cfg.rewards,
        emote=base_cfg.emote,
        layers=base_cfg.layers,
        tile_coder=base_cfg.tile_coder,
        communication=base_cfg.communication,
        fractal=base_cfg.fractal,
        multi_layer=base_cfg.multi_layer,
        episodic=base_cfg.episodic,
    )

    cfg.validate()

    trainer = QTrainer(cfg, use_crypto_env=False, seed=seed_override)
    _policy, _trace = trainer.run()

    rewards = list(getattr(trainer, "_episode_rewards", []))
    if not rewards:
        # Degenerate case: no rewards recorded, treat as worst.
        return (0.0, -0.0, -0.0)

    avg_reward = float(mean(rewards))
    dispersion = float(-pstdev(rewards) if len(rewards) > 1 else 0.0)
    min_reward = float(min(rewards))

    return (avg_reward, dispersion, min_reward)


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """Return True if metric vector a dominates b.

    Dominance is defined as:
    - a is >= b in all components, and
    - a is > b in at least one component.
    """
    better_or_equal_all = True
    strictly_better_one = False

    for ai, bi in zip(a, b):
        if ai < bi:
            better_or_equal_all = False
            break
        if ai > bi:
            strictly_better_one = True

    return better_or_equal_all and strictly_better_one


def select_pareto_front(
    patches: Iterable[QPatchMetrics],
    limit: int,
) -> List[QPatchMetrics]:
    """Select up to `limit` non-dominated patches.

    This implementation is O(n^2) in the number of patches and is
    intended for small experimental populations.
    """
    items = list(patches)
    non_dominated: List[QPatchMetrics] = []

    for i, cand in enumerate(items):
        dominated = False
        for j, other in enumerate(items):
            if i == j:
                continue
            if dominates(other.values, cand.values):
                dominated = True
                break
        if not dominated:
            non_dominated.append(cand)

    if len(non_dominated) <= limit:
        return non_dominated

    return non_dominated[:limit]


def evolve_q_patches(
    base_patch: QAgentPatch,
    population_size: int,
    iterations: int,
    evaluator: Callable[[QAgentPatch], Tuple[float, ...]] = evaluate_patch_real,
) -> List[QPatchMetrics]:
    """Run a small SAPE-style evolution on QAgentPatch space.

    Preconditions:
        - base_patch is structurally valid and passes spec stub.
    Postconditions:
        - Returns non-empty list of Pareto-consistent, spec-safe patches.
    """
    if not validate_patch_fast(base_patch):
        raise ValueError("Base patch is structurally invalid")
    if not check_spec_stub(base_patch):
        raise ValueError("Base patch does not satisfy spec stub")

    base_metrics = evaluator(base_patch)
    population: List[QPatchMetrics] = [QPatchMetrics(base_patch, base_metrics)]

    for _ in range(iterations):
        candidates: List[QAgentPatch] = []

        for entry in population:
            for neighbor in mutate_patch(entry.patch):
                if not validate_patch_fast(neighbor):
                    continue
                if not check_spec_stub(neighbor):
                    continue
                candidates.append(neighbor)

        if not candidates:
            break

        evaluated_candidates = [
            QPatchMetrics(p, evaluator(p)) for p in candidates
        ]

        merged: List[QPatchMetrics] = population + evaluated_candidates
        population = select_pareto_front(merged, population_size)

    return population
