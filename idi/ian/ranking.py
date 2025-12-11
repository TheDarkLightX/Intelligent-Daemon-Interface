"""
Ranking functions for IAN leaderboard scoring.

Provides pluggable ranking strategies:
- ScalarRanking: Weighted sum of metrics
- ParetoRanking: Multi-objective non-dominated comparison

These are used by GoalSpec.compute_score() and the coordinator
to determine candidate quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import Metrics


class RankingFunction(ABC):
    """Abstract base class for ranking functions."""
    
    @abstractmethod
    def score(self, metrics: Metrics) -> float:
        """
        Compute a scalar score from metrics.
        
        Higher scores are better.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Scalar score
        """
        ...
    
    @abstractmethod
    def compare(self, m1: Metrics, m2: Metrics) -> int:
        """
        Compare two metrics.
        
        Returns:
            1 if m1 > m2, -1 if m1 < m2, 0 if equal
        """
        ...


@dataclass
class ScalarRanking(RankingFunction):
    """
    Weighted linear combination of metrics.
    
    score = w_reward * reward + w_risk * risk + w_complexity * complexity + ...
    
    Typically: w_reward > 0, w_risk < 0, w_complexity < 0
    """
    
    weight_reward: float = 1.0
    weight_risk: float = -0.5
    weight_complexity: float = -0.01
    weight_sharpe: float = 0.0
    weight_drawdown: float = 0.0
    
    def score(self, metrics: Metrics) -> float:
        """Compute weighted score."""
        s = 0.0
        s += self.weight_reward * metrics.reward
        s += self.weight_risk * metrics.risk
        s += self.weight_complexity * metrics.complexity
        
        if metrics.sharpe_ratio is not None:
            s += self.weight_sharpe * metrics.sharpe_ratio
        if metrics.max_drawdown is not None:
            s += self.weight_drawdown * metrics.max_drawdown
        
        return s
    
    def compare(self, m1: Metrics, m2: Metrics) -> int:
        """Compare by scalar score."""
        s1 = self.score(m1)
        s2 = self.score(m2)
        
        if s1 > s2:
            return 1
        elif s1 < s2:
            return -1
        return 0
    
    @classmethod
    def from_weights(cls, weights: Dict[str, float]) -> "ScalarRanking":
        """Create from a weights dictionary."""
        return cls(
            weight_reward=weights.get("reward", 1.0),
            weight_risk=weights.get("risk", -0.5),
            weight_complexity=weights.get("complexity", -0.01),
            weight_sharpe=weights.get("sharpe_ratio", 0.0),
            weight_drawdown=weights.get("max_drawdown", 0.0),
        )


class ParetoRanking(RankingFunction):
    """
    Multi-objective Pareto ranking.
    
    Does not produce a total ordering, but can determine:
    - Dominance relationships
    - Whether a candidate is on the Pareto frontier
    
    For score(), returns the distance from the "anti-ideal" point
    (worst possible in all dimensions).
    """
    
    def __init__(
        self,
        objectives: Optional[List[str]] = None,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Initialize Pareto ranking.
        
        Args:
            objectives: List of metric names to consider
            maximize: Dict mapping objective name to whether it should be maximized
        """
        self.objectives = objectives or ["reward", "risk", "complexity"]
        self.maximize = maximize or {
            "reward": True,  # Higher reward is better
            "risk": False,  # Lower risk is better
            "complexity": False,  # Lower complexity is better
            "sharpe_ratio": True,
            "max_drawdown": False,
        }
    
    def _get_objective_value(self, metrics: Metrics, obj: str) -> Optional[float]:
        """
        Get value of an objective from metrics.
        
        Uses getattr for extensibility - any numeric attribute of Metrics
        can be used as an objective.
        """
        value = getattr(metrics, obj, None)
        # Only return if it's a valid numeric value
        if isinstance(value, (int, float)):
            return float(value)
        return None
    
    def dominates(self, m1: Metrics, m2: Metrics) -> bool:
        """
        Check if m1 Pareto-dominates m2.
        
        m1 dominates m2 iff:
        - m1 is at least as good as m2 in all objectives
        - m1 is strictly better in at least one objective
        """
        at_least_as_good = True
        strictly_better = False
        
        for obj in self.objectives:
            v1 = self._get_objective_value(m1, obj)
            v2 = self._get_objective_value(m2, obj)
            
            if v1 is None or v2 is None:
                continue
            
            maximize = self.maximize.get(obj, True)
            
            if maximize:
                if v1 < v2:
                    at_least_as_good = False
                if v1 > v2:
                    strictly_better = True
            else:
                if v1 > v2:
                    at_least_as_good = False
                if v1 < v2:
                    strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def score(self, metrics: Metrics) -> float:
        """
        Compute a scalar score for Pareto ranking.
        
        Uses distance from anti-ideal point (normalized).
        This provides a rough ordering but is not the primary
        comparison method for Pareto ranking.
        """
        # Simple weighted sum as fallback score
        score = 0.0
        for obj in self.objectives:
            v = self._get_objective_value(metrics, obj)
            if v is None:
                continue
            
            maximize = self.maximize.get(obj, True)
            if maximize:
                score += v
            else:
                score -= v
        
        return score
    
    def compare(self, m1: Metrics, m2: Metrics) -> int:
        """
        Compare two metrics using Pareto dominance.
        
        Returns:
            1 if m1 dominates m2
            -1 if m2 dominates m1
            0 if neither dominates (incomparable or equal)
        """
        if self.dominates(m1, m2):
            return 1
        elif self.dominates(m2, m1):
            return -1
        return 0


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def rank_contributions(
    metrics_list: List[Metrics],
    ranking: RankingFunction,
) -> List[int]:
    """
    Rank a list of metrics.
    
    Args:
        metrics_list: List of Metrics objects
        ranking: Ranking function to use
        
    Returns:
        List of indices sorted by rank (best first)
    """
    scored = [(i, ranking.score(m)) for i, m in enumerate(metrics_list)]
    scored.sort(key=lambda x: -x[1])  # Descending by score
    return [i for i, _ in scored]


def is_pareto_optimal(
    metrics: Metrics,
    others: List[Metrics],
) -> bool:
    """
    Check if metrics is Pareto-optimal among others.
    
    Args:
        metrics: Metrics to check
        others: List of other Metrics
        
    Returns:
        True if no other metric dominates this one
    """
    ranking = ParetoRanking()
    for other in others:
        if ranking.dominates(other, metrics):
            return False
    return True
