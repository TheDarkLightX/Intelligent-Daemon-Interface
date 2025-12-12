"""Off-Policy Evaluation (OPE) module for Q-table policies.

Supports Direct Method (DM), Importance Sampling (IPS/WIS), and Doubly Robust (DR)
estimators for evaluating candidate policies against logged trajectories.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import TrainingConfig
from .domain import Action, StateKey
from .policy import LookupPolicy


@dataclass
class LoggedTransition:
    """A single transition from logged data."""

    state: StateKey
    action: Action
    reward: float
    next_state: StateKey
    behavior_prob: float  # Probability of action under behavior policy
    done: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "LoggedTransition":
        """Create from dictionary representation."""
        action = Action.from_str(d["action"]) if isinstance(d["action"], str) else d["action"]
        return cls(
            state=tuple(d["state"]),
            action=action,
            reward=d["reward"],
            next_state=tuple(d["next_state"]),
            behavior_prob=d.get("behavior_prob", 1.0),
            done=d.get("done", False),
        )


@dataclass
class LoggedEpisode:
    """A sequence of transitions forming an episode."""

    episode_id: str
    transitions: List[LoggedTransition]
    behavior_policy_id: str
    config_hash: str
    data_version: str

    @property
    def total_reward(self) -> float:
        """Sum of rewards in episode."""
        return sum(t.reward for t in self.transitions)

    @classmethod
    def from_dict(cls, d: Dict) -> "LoggedEpisode":
        """Create from dictionary representation."""
        return cls(
            episode_id=d["episode_id"],
            transitions=[LoggedTransition.from_dict(t) for t in d["transitions"]],
            behavior_policy_id=d.get("behavior_policy_id", "unknown"),
            config_hash=d.get("config_hash", ""),
            data_version=d.get("data_version", "1.0"),
        )


@dataclass
class LoggedDataset:
    """Collection of logged episodes with metadata."""

    episodes: List[LoggedEpisode]
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "LoggedDataset":
        """Load dataset from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            episodes=[LoggedEpisode.from_dict(ep) for ep in data["episodes"]],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "transitions": [
                        {
                            "state": list(t.state),
                            "action": t.action.value if isinstance(t.action, Action) else t.action,
                            "reward": t.reward,
                            "next_state": list(t.next_state),
                            "behavior_prob": t.behavior_prob,
                            "done": t.done,
                        }
                        for t in ep.transitions
                    ],
                    "behavior_policy_id": ep.behavior_policy_id,
                    "config_hash": ep.config_hash,
                    "data_version": ep.data_version,
                }
                for ep in self.episodes
            ],
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(data, indent=2))


@dataclass
class OPEResult:
    """Results from off-policy evaluation."""

    estimator: str
    value_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    n_episodes: int
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "estimator": self.estimator,
            "value_estimate": self.value_estimate,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "n_episodes": self.n_episodes,
            "details": self.details,
        }


class OPEEvaluator:
    """Off-policy evaluation for Q-table policies."""

    def __init__(
        self,
        candidate_policy: LookupPolicy,
        discount: float = 0.99,
        confidence_level: float = 0.95,
    ):
        """Initialize evaluator.

        Args:
            candidate_policy: Policy to evaluate
            discount: Discount factor for returns
            confidence_level: Confidence level for intervals (e.g., 0.95)
        """
        self.policy = candidate_policy
        self.discount = discount
        self.confidence_level = confidence_level
        self._z_score = 1.96  # For 95% CI; could be computed from confidence_level

    def _policy_prob(self, state: StateKey, action: Action) -> float:
        """Get probability of action under candidate policy (greedy = 1.0 or 0.0)."""
        best = self.policy.best_action(state)
        return 1.0 if action == best or (hasattr(action, "value") and action.value == best.value) else 0.0

    def _compute_discounted_return(self, episode: LoggedEpisode) -> float:
        """Compute discounted return for an episode."""
        ret = 0.0
        discount_factor = 1.0
        for t in episode.transitions:
            ret += discount_factor * t.reward
            discount_factor *= self.discount
        return ret

    def direct_method(self, dataset: LoggedDataset) -> OPEResult:
        """Direct Method (DM): Use Q-table to estimate policy value.

        Estimates V(s) = max_a Q(s,a) under candidate policy for initial states.
        """
        if not dataset.episodes:
            return OPEResult("DM", 0.0, 0.0, (0.0, 0.0), 0)

        values = []
        for ep in dataset.episodes:
            if ep.transitions:
                initial_state = ep.transitions[0].state
                # Value under greedy policy = Q(s, best_action)
                best_action = self.policy.best_action(initial_state)
                v = self.policy.q_value(initial_state, best_action)
                values.append(v)

        n = len(values)
        if n == 0:
            return OPEResult("DM", 0.0, 0.0, (0.0, 0.0), 0)

        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / max(1, n - 1)
        std_err = math.sqrt(variance / n) if n > 1 else 0.0
        ci = (mean_val - self._z_score * std_err, mean_val + self._z_score * std_err)

        return OPEResult(
            estimator="DM",
            value_estimate=mean_val,
            standard_error=std_err,
            confidence_interval=ci,
            n_episodes=n,
            details={"variance": variance},
        )

    def importance_sampling(self, dataset: LoggedDataset, weighted: bool = True) -> OPEResult:
        """Importance Sampling (IPS) or Weighted Importance Sampling (WIS).

        Args:
            dataset: Logged trajectories
            weighted: If True, use WIS; otherwise plain IPS
        """
        if not dataset.episodes:
            return OPEResult("WIS" if weighted else "IPS", 0.0, 0.0, (0.0, 0.0), 0)

        weighted_returns = []
        weights = []

        for ep in dataset.episodes:
            # Compute importance weight as product of ratios
            weight = 1.0
            for t in ep.transitions:
                pi_prob = self._policy_prob(t.state, t.action)
                if t.behavior_prob <= 0:
                    weight = 0.0
                    break
                weight *= pi_prob / t.behavior_prob

            discounted_return = self._compute_discounted_return(ep)
            weighted_returns.append(weight * discounted_return)
            weights.append(weight)

        n = len(weighted_returns)
        if n == 0:
            return OPEResult("WIS" if weighted else "IPS", 0.0, 0.0, (0.0, 0.0), 0)

        if weighted:
            total_weight = sum(weights)
            if total_weight <= 0:
                mean_val = 0.0
            else:
                mean_val = sum(weighted_returns) / total_weight
        else:
            mean_val = sum(weighted_returns) / n

        # Compute standard error via bootstrap approximation
        variance = sum((wr - mean_val) ** 2 for wr in weighted_returns) / max(1, n - 1)
        std_err = math.sqrt(variance / n) if n > 1 else 0.0
        ci = (mean_val - self._z_score * std_err, mean_val + self._z_score * std_err)

        estimator_name = "WIS" if weighted else "IPS"
        return OPEResult(
            estimator=estimator_name,
            value_estimate=mean_val,
            standard_error=std_err,
            confidence_interval=ci,
            n_episodes=n,
            details={"total_weight": sum(weights), "variance": variance},
        )

    def doubly_robust(self, dataset: LoggedDataset) -> OPEResult:
        """Doubly Robust (DR) estimator combining DM and IPS.

        DR = DM + IPS correction, reducing variance when Q estimates are accurate.
        """
        if not dataset.episodes:
            return OPEResult("DR", 0.0, 0.0, (0.0, 0.0), 0)

        dr_values = []

        for ep in dataset.episodes:
            if not ep.transitions:
                continue

            # Start with DM estimate
            initial_state = ep.transitions[0].state
            best_action = self.policy.best_action(initial_state)
            dm_value = self.policy.q_value(initial_state, best_action)

            # Compute IPS correction term
            weight = 1.0
            correction = 0.0
            discount_factor = 1.0

            for t in ep.transitions:
                pi_prob = self._policy_prob(t.state, t.action)
                if t.behavior_prob <= 0:
                    weight = 0.0
                    break
                ratio = pi_prob / t.behavior_prob
                weight *= ratio

                # Q(s,a) under candidate policy
                q_sa = self.policy.q_value(t.state, t.action)
                # V(s') under candidate policy
                if not t.done:
                    best_next = self.policy.best_action(t.next_state)
                    v_next = self.policy.q_value(t.next_state, best_next)
                else:
                    v_next = 0.0

                # TD-like correction
                td_error = t.reward + self.discount * v_next - q_sa
                correction += discount_factor * weight * td_error
                discount_factor *= self.discount

            dr_value = dm_value + correction
            dr_values.append(dr_value)

        n = len(dr_values)
        if n == 0:
            return OPEResult("DR", 0.0, 0.0, (0.0, 0.0), 0)

        mean_val = sum(dr_values) / n
        variance = sum((v - mean_val) ** 2 for v in dr_values) / max(1, n - 1)
        std_err = math.sqrt(variance / n) if n > 1 else 0.0
        ci = (mean_val - self._z_score * std_err, mean_val + self._z_score * std_err)

        return OPEResult(
            estimator="DR",
            value_estimate=mean_val,
            standard_error=std_err,
            confidence_interval=ci,
            n_episodes=n,
            details={"variance": variance},
        )

    def evaluate_all(self, dataset: LoggedDataset) -> Dict[str, OPEResult]:
        """Run all estimators and return results."""
        return {
            "DM": self.direct_method(dataset),
            "IPS": self.importance_sampling(dataset, weighted=False),
            "WIS": self.importance_sampling(dataset, weighted=True),
            "DR": self.doubly_robust(dataset),
        }


def run_ope(
    policy: LookupPolicy,
    dataset_path: Path,
    discount: float = 0.99,
    output_path: Optional[Path] = None,
) -> Dict[str, OPEResult]:
    """Run OPE evaluation and optionally save results.

    Args:
        policy: Candidate policy to evaluate
        dataset_path: Path to logged dataset JSON
        discount: Discount factor
        output_path: Optional path to save results JSON

    Returns:
        Dictionary of estimator name to OPEResult
    """
    dataset = LoggedDataset.from_json(dataset_path)
    evaluator = OPEEvaluator(policy, discount=discount)
    results = evaluator.evaluate_all(dataset)

    if output_path:
        output = {name: result.to_dict() for name, result in results.items()}
        output_path.write_text(json.dumps(output, indent=2))

    return results

