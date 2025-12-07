"""Fractal state abstraction utilities (self-similar recursive decomposition).

Based on goi/demos/kernel_shell/fractal/product_mdp.py and fractal_q_agent.py patterns.
Implements hierarchical state encoding with fractal backoff strategy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np

from .config import TileCoderConfig
from .abstraction import TileCoder
from .domain import Action


@dataclass
class FractalLevelConfig:
    """Configuration for a single abstraction level."""
    features: List[Tuple[str, int]]  # [(feature_name, num_buckets), ...]
    scale_factor: float  # Scaling factor for this level
    visit_threshold: int = 5  # Minimum visits before using this level


@dataclass
class FractalConfig:
    """Configuration for fractal state abstraction."""
    levels: List[FractalLevelConfig]
    backoff_enabled: bool = True
    hierarchical_updates: bool = True


class FractalStateEncoder:
    """
    Recursive state encoder with self-similar structure.
    
    Based on goi/demos/kernel_shell/fractal/product_mdp.py fractal_discretize pattern.
    
    Example hierarchy:
    Level 0 (coarse): (regime: 4 values) = 4 states
    Level 1 (medium): (regime: 4, trend: 4) = 16 states  
    Level 2 (fine): (regime: 4, trend: 4, momentum: 8) = 128 states
    
    Each level's states decompose into sub-states at next level.
    """
    
    def __init__(self, config: FractalConfig):
        self.config = config
        self.num_levels = len(config.levels)
        self._visit_counts: Dict[int, Dict[Tuple[int, ...], int]] = {}
        for level in range(self.num_levels):
            self._visit_counts[level] = {}
    
    def fractal_discretize(self, raw_features: Dict[str, float], level: int) -> Tuple[int, ...]:
        """
        Discretize features at specific abstraction level.
        
        Uses product MDP approach: each dimension discretized independently,
        then combined into product state space.
        
        Based on fractal_discretize from product_mdp.py:
        - Regime: 3 bins (bear, neutral, bull)
        - Momentum: 5 bins
        - Position: 3 bins
        """
        level_config = self.config.levels[level]
        buckets = []
        
        for feature_name, num_buckets in level_config.features:
            value = raw_features.get(feature_name, 0.0)
            # Scale by level's scale factor
            scaled_value = value * level_config.scale_factor
            
            # Discretize into buckets (similar to product_mdp.py logic)
            if feature_name == "regime" or "regime" in feature_name.lower():
                # Regime: 3 bins (bear < -0.01, neutral, bull > 0.01)
                if scaled_value < -0.01:
                    bucket = 0
                elif scaled_value > 0.01:
                    bucket = 2
                else:
                    bucket = 1
                bucket = min(bucket, num_buckets - 1)
            elif feature_name == "momentum" or "momentum" in feature_name.lower():
                # Momentum: clip to range and discretize
                bucket = int(np.clip((scaled_value + 0.03) * 50, 0, num_buckets - 1))
            elif feature_name == "position" or "position" in feature_name.lower():
                # Position: simple binning
                bucket = min(int(abs(scaled_value) * num_buckets), num_buckets - 1)
            else:
                # Generic binning
                bucket = int(scaled_value * num_buckets) % num_buckets
            
            buckets.append(bucket)
        
        return tuple(buckets)
    
    def encode(self, raw_state: Dict[str, float], level: int) -> Tuple[int, ...]:
        """Encode state at specific abstraction level."""
        encoded = self.fractal_discretize(raw_state, level)
        # Track visits for backoff strategy
        if self.config.backoff_enabled:
            self._visit_counts[level][encoded] = self._visit_counts[level].get(encoded, 0) + 1
        return encoded
    
    def decompose(self, state: Tuple[int, ...], level: int) -> List[Tuple[int, ...]]:
        """
        Decompose state into sub-states at next level.
        
        Example: Level 0 state (regime=2) → Level 1 sub-states 
        (regime=2, trend=0), (regime=2, trend=1), ..., (regime=2, trend=3)
        """
        if level >= self.num_levels - 1:
            return [state]  # Can't decompose further
        
        next_level_config = self.config.levels[level + 1]
        sub_states = []
        
        # Generate all combinations of next-level features for this state
        # This creates self-similar structure
        next_feature_buckets = [nb for _, nb in next_level_config.features]
        for combination in itertools.product(*[range(nb) for nb in next_feature_buckets]):
            sub_state = state + combination
            sub_states.append(sub_state)
        
        return sub_states
    
    def aggregate_q(self, q_values: Dict[Tuple[int, ...], float], level: int) -> Dict[Tuple[int, ...], float]:
        """
        Aggregate Q-values from fine to coarse levels.
        
        Uses weighted average: Q_coarse = mean(Q_fine for all sub-states)
        """
        aggregated = {}
        # Get all states at this level
        for state in self._visit_counts[level].keys():
            sub_states = self.decompose(state, level)
            if sub_states:
                q_sum = sum(q_values.get(sub, 0.0) for sub in sub_states)
                aggregated[state] = q_sum / len(sub_states)
        return aggregated
    
    def get_visit_count(self, state: Tuple[int, ...], level: int) -> int:
        """Get visit count for a state at a specific level."""
        return self._visit_counts[level].get(state, 0)
    
    def should_use_level(self, state: Tuple[int, ...], level: int) -> bool:
        """Check if level should be used based on visit threshold."""
        threshold = self.config.levels[level].visit_threshold
        return self.get_visit_count(state, level) >= threshold


class HierarchicalTileCoder:
    """
    Multi-scale tile coding with recursive structure.
    
    Extends TileCoder with fractal backoff strategy.
    
    Level 0: Large tiles (coarse abstraction, e.g., 4x4x4)
    Level 1: Medium tiles (medium abstraction, e.g., 8x8x8)  
    Level 2: Small tiles (fine abstraction, e.g., 16x16x16)
    
    Tiles overlap across levels for generalization.
    Backoff: if fine-level state unseen, use coarse-level Q-value.
    """
    
    def __init__(self, base_config: TileCoderConfig, num_levels: int, scale_ratios: List[float]):
        """
        Args:
            base_config: Base tile coder configuration
            num_levels: Number of abstraction levels
            scale_ratios: Scale factors for each level [coarse, medium, fine, ...]
        """
        self.base_config = base_config
        self.num_levels = num_levels
        self.scale_ratios = scale_ratios  # [1.0, 0.5, 0.25] for 3 levels
        self._coders: List[TileCoder] = []
        self._build_levels()
    
    def _build_levels(self) -> None:
        """Build tile coders for each abstraction level."""
        for level in range(self.num_levels):
            scaled_sizes = [
                max(1, int(s * self.scale_ratios[level])) 
                for s in self.base_config.tile_sizes
            ]
            level_config = TileCoderConfig(
                num_tilings=self.base_config.num_tilings,
                tile_sizes=scaled_sizes,
                offsets=self.base_config.offsets
            )
            self._coders.append(TileCoder(level_config))
    
    def encode(self, state: Sequence[int], level: int) -> Tuple[int, ...]:
        """Encode state at specific level with scaled tile sizes."""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} out of range [0, {self.num_levels})")
        return self._coders[level].encode(state)
    
    def encode_all_levels(self, state: Sequence[int]) -> Dict[int, Tuple[int, ...]]:
        """Encode state at all levels simultaneously."""
        return {level: self.encode(state, level) for level in range(self.num_levels)}
    
    def backoff_q_value(
        self, 
        state: Sequence[int], 
        action: Action, 
        q_tables: Dict[int, Dict[Tuple[int, ...], Dict[Action, float]]],
        visit_counts: Optional[Dict[int, Dict[Tuple[int, ...], int]]] = None,
        visit_thresholds: Optional[Dict[int, int]] = None
    ) -> float:
        """
        Get Q-value with fractal backoff: try fine level first, fall back to coarse.
        
        This implements the "fractal backoff" pattern from fractal_q_agent.py:
        - Prefer fine level if visited enough
        - Fall back to medium if fine unseen
        - Fall back to coarse if medium unseen
        """
        if visit_counts is None:
            visit_counts = {}
        if visit_thresholds is None:
            visit_thresholds = {level: 5 for level in range(self.num_levels)}
        
        # Try fine level first (highest level = finest)
        for level in range(self.num_levels - 1, -1, -1):
            encoded = self.encode(state, level)
            level_visits = visit_counts.get(level, {})
            threshold = visit_thresholds.get(level, 5)
            
            # Check if this level has been visited enough
            if level_visits.get(encoded, 0) >= threshold:
                if encoded in q_tables[level] and action in q_tables[level][encoded]:
                    return q_tables[level][encoded][action]
        
        return 0.0  # Default if no level has this state


class FractalQTable:
    """
    Q-table with self-similar hierarchical structure.
    
    Implements fractal backoff: prefer fine states, fall back to coarse.
    Based on pattern from goi/demos/kernel_shell/fractal/fractal_q_agent.py.
    
    Q-values stored at multiple abstraction levels.
    Updates propagate both up (aggregation) and down (refinement).
    """
    
    def __init__(self, encoder: FractalStateEncoder, tile_coder: Optional[HierarchicalTileCoder] = None):
        self.encoder = encoder
        self.tile_coder = tile_coder
        self._tables: Dict[int, Dict[Tuple[int, ...], Dict[Action, float]]] = {}
        self._visit_counts: Dict[int, Dict[Tuple[int, ...], int]] = {}
        for level in range(encoder.num_levels):
            self._tables[level] = {}
            self._visit_counts[level] = {}
    
    def q_value(self, raw_state: Dict[str, float], action: Action, level: Optional[int] = None) -> float:
        """
        Get Q-value with fractal backoff.
        
        If level specified, get Q-value at that level.
        If level=None, use backoff strategy: try fine → medium → coarse.
        """
        if level is not None:
            encoded = self.encoder.encode(raw_state, level)
            return self._tables[level].get(encoded, {}).get(action, 0.0)
        
        # Fractal backoff: try fine level first
        if self.tile_coder:
            state_tuple = tuple(raw_state.values())
            return self.tile_coder.backoff_q_value(
                state_tuple, action, self._tables, 
                self._visit_counts,
                {level: self.encoder.config.levels[level].visit_threshold 
                 for level in range(self.encoder.num_levels)}
            )
        
        # Fallback: try levels in order
        for level_idx in range(self.encoder.num_levels - 1, -1, -1):
            encoded = self.encoder.encode(raw_state, level_idx)
            if encoded in self._tables[level_idx] and action in self._tables[level_idx][encoded]:
                if self.encoder.should_use_level(encoded, level_idx):
                    return self._tables[level_idx][encoded][action]
        
        return 0.0
    
    def update(self, raw_state: Dict[str, float], action: Action, delta: float, level: Optional[int] = None):
        """
        Update Q-value with hierarchical propagation.
        
        Based on fractal_q_agent.py update_q pattern:
        - Fine level gets full update (alpha)
        - Medium level gets 0.5 * alpha update
        - Coarse level gets 0.25 * alpha update
        
        If level specified, update at that level only.
        If level=None, update at finest level that has this state, propagate to coarser levels.
        """
        if level is not None:
            encoded = self.encoder.encode(raw_state, level)
            entry = self._tables[level].setdefault(encoded, {})
            entry[action] = entry.get(action, 0.0) + delta
            self._visit_counts[level][encoded] = self._visit_counts[level].get(encoded, 0) + 1
            return
        
        # Update at all levels with different learning rates (fractal pattern)
        update_weights = [1.0, 0.5, 0.25]  # Fine, medium, coarse
        
        for level_idx in range(self.encoder.num_levels - 1, -1, -1):
            encoded = self.encoder.encode(raw_state, level_idx)
            entry = self._tables[level_idx].setdefault(encoded, {})
            weight = update_weights[level_idx] if level_idx < len(update_weights) else 1.0
            entry[action] = entry.get(action, 0.0) + (delta * weight)
            self._visit_counts[level_idx][encoded] = self._visit_counts[level_idx].get(encoded, 0) + 1
    
    def best_action(self, raw_state: Dict[str, float], level: Optional[int] = None) -> Action:
        """
        Select best action with fractal backoff.
        
        Uses backoff strategy: try fine level first, fall back to coarse.
        """
        best_action = None
        best_q = float('-inf')
        
        for action in Action:
            q = self.q_value(raw_state, action, level)
            if q > best_q:
                best_q = q
                best_action = action
        
        return best_action or Action.HOLD
    
    def get_table_size(self, level: int) -> int:
        """Get number of states in Q-table at specific level."""
        return len(self._tables[level])


class FractalPolicy:
    """
    Policy that uses fractal abstraction with backoff strategy.
    
    Selects abstraction level based on uncertainty/exploration,
    uses backoff for unseen states.
    """
    
    def __init__(self, fractal_q_table: FractalQTable):
        self.fractal_q_table = fractal_q_table
        self.encoder = fractal_q_table.encoder
    
    def choose_action(self, raw_state: Dict[str, float], exploration: float = 0.0) -> Action:
        """
        Choose action using fractal backoff strategy.
        
        Based on fractal_q_agent.py select_action pattern:
        - Prefer fine level if visited enough
        - Fall back to medium if fine unseen
        - Fall back to coarse if medium unseen
        - Random exploration if all levels unseen
        """
        import random
        
        # Exploration
        if random.random() < exploration:
            return random.choice(list(Action))
        
        # Try to find best level to use
        for level in range(self.encoder.num_levels - 1, -1, -1):
            encoded = self.encoder.encode(raw_state, level)
            if self.encoder.should_use_level(encoded, level):
                return self.fractal_q_table.best_action(raw_state, level)
        
        # All levels unseen, use random
        return random.choice(list(Action))

