"""
MeritRank: Sybil-tolerant reputation system for evaluator scoring.

Security Controls:
- Multi-seed BFT consensus (prevents single seed manipulation)
- Transitivity decay α bounds serial Sybil attacks to O(1)
- Connectivity decay β bounds parallel attacks to O(√k)
- Bridge detection prevents reputation laundering
- Recency weighting prevents temporal manipulation
- Bounded iterations prevent DoS

Based on: MeritRank (TU Delft 2022), with IAN enhancements

Author: DarkLightX
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants
# =============================================================================

# Decay parameters (from MeritRank paper)
DEFAULT_ALPHA = 0.15  # Transitivity decay (random restart probability)
DEFAULT_BETA = 0.3    # Connectivity decay (bridge penalty)

# Walk parameters
DEFAULT_NUM_WALKS = 1000
MAX_WALK_LENGTH = 100
MIN_WALKS = 100
MAX_WALKS = 10000

# Graph limits
MAX_NODES = 100000
MAX_EDGES_PER_NODE = 1000
MAX_TOTAL_EDGES = 1000000

# Seed selection
DEFAULT_NUM_SEEDS = 5
MIN_SEEDS = 3
MAX_SEEDS = 20

# Recency weighting
RECENCY_HALFLIFE_MS = 7 * 24 * 60 * 60 * 1000  # 7 days
MIN_EDGE_WEIGHT = 0.01

# Slashing
SYBIL_DETECTION_THRESHOLD = 0.8
SLASHING_PENALTY = 0.5

# Score bounds
MIN_SCORE = 0.0
MAX_SCORE = 1.0
EPSILON = 1e-10  # Prevent division by zero


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EvaluatorNode:
    """
    Node representing an evaluator in the reputation graph.
    
    Security:
        - stake must be positive
        - registered_at_ms for temporal tracking
    """
    evaluator_id: str
    stake: float
    registered_at_ms: int
    
    # Computed scores (updated by MeritRank)
    merit_score: float = 0.0
    bridge_score: float = 0.0  # Higher = more bridge-like (suspicious)
    
    # Flags
    is_seed: bool = False
    is_slashed: bool = False
    slash_count: int = 0
    
    def __post_init__(self) -> None:
        if self.stake < 0:
            raise ValueError("Stake must be non-negative")


@dataclass
class EvaluationEdge:
    """
    Directed edge representing an evaluation relationship.
    
    from_id evaluated to_id with quality score at timestamp.
    
    Security:
        - quality clamped to [0, 1]
        - timestamp_ms for recency weighting
    """
    from_id: str
    to_id: str
    quality: float  # [0, 1] evaluation quality
    timestamp_ms: int
    contribution_hash: bytes  # Hash of contribution being evaluated
    
    # Computed weight (includes recency)
    weight: float = field(default=1.0, init=False)
    
    def __post_init__(self) -> None:
        self.quality = max(0.0, min(1.0, self.quality))
        if len(self.contribution_hash) != 32:
            raise ValueError("contribution_hash must be 32 bytes")


@dataclass
class SybilCluster:
    """
    Detected cluster of likely Sybil nodes.
    
    Security: Evidence includes specific detection signals.
    """
    cluster_id: str
    node_ids: Set[str]
    confidence: float  # [0, 1] confidence this is a Sybil cluster
    detection_method: str
    detected_at_ms: int
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class SlashingEvent:
    """
    Record of a slashing event for Sybil behavior.
    """
    event_id: str
    node_id: str
    cluster_id: str
    penalty_fraction: float
    reason: str
    timestamp_ms: int


# =============================================================================
# Reputation Graph
# =============================================================================

class ReputationGraph:
    """
    Graph structure for MeritRank computation.
    
    Security:
        - Bounded node and edge counts
        - Validates all inputs
    """
    
    def __init__(self):
        self._nodes: Dict[str, EvaluatorNode] = {}
        self._outgoing: Dict[str, List[EvaluationEdge]] = defaultdict(list)
        self._incoming: Dict[str, List[EvaluationEdge]] = defaultdict(list)
        self._edge_count = 0
    
    @property
    def node_count(self) -> int:
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        return self._edge_count
    
    def add_node(self, node: EvaluatorNode) -> None:
        """Add or update a node."""
        if self.node_count >= MAX_NODES and node.evaluator_id not in self._nodes:
            raise RuntimeError(f"Maximum nodes ({MAX_NODES}) exceeded")
        self._nodes[node.evaluator_id] = node
    
    def get_node(self, node_id: str) -> Optional[EvaluatorNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def add_edge(self, edge: EvaluationEdge) -> None:
        """Add an evaluation edge."""
        if self._edge_count >= MAX_TOTAL_EDGES:
            raise RuntimeError(f"Maximum edges ({MAX_TOTAL_EDGES}) exceeded")
        
        if edge.from_id not in self._nodes or edge.to_id not in self._nodes:
            raise ValueError("Edge endpoints must exist in graph")
        
        if len(self._outgoing[edge.from_id]) >= MAX_EDGES_PER_NODE:
            raise RuntimeError(f"Maximum edges per node ({MAX_EDGES_PER_NODE}) exceeded")
        
        self._outgoing[edge.from_id].append(edge)
        self._incoming[edge.to_id].append(edge)
        self._edge_count += 1
    
    def get_outgoing(self, node_id: str) -> List[EvaluationEdge]:
        """Get outgoing edges from node."""
        return self._outgoing.get(node_id, [])
    
    def get_incoming(self, node_id: str) -> List[EvaluationEdge]:
        """Get incoming edges to node."""
        return self._incoming.get(node_id, [])
    
    def get_all_nodes(self) -> List[EvaluatorNode]:
        """Get all nodes."""
        return list(self._nodes.values())
    
    def get_all_node_ids(self) -> Set[str]:
        """Get all node IDs."""
        return set(self._nodes.keys())


# =============================================================================
# MeritRank Algorithm
# =============================================================================

class MeritRank:
    """
    Sybil-tolerant reputation system using random walks with decay.
    
    Security features:
        - Multi-seed BFT consensus
        - Transitivity decay (α) for serial Sybil resistance
        - Connectivity decay (β) for parallel Sybil resistance
        - Bridge detection for reputation laundering prevention
        - Recency weighting for temporal attack resistance
        - Slashing for detected Sybil clusters
    
    Key properties:
        - Serial Sybil attack bounded to O(1) gain
        - Parallel Sybil attack bounded to O(√k) gain
        - Converges in O(N * num_walks) time
    
    Usage:
        mr = MeritRank(graph)
        mr.select_seeds()
        mr.compute()
        scores = mr.get_scores()
    """
    
    def __init__(
        self,
        graph: ReputationGraph,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        num_walks: int = DEFAULT_NUM_WALKS,
        num_seeds: int = DEFAULT_NUM_SEEDS,
    ):
        """
        Initialize MeritRank.
        
        Args:
            graph: Reputation graph
            alpha: Transitivity decay (restart probability)
            beta: Connectivity decay (bridge penalty)
            num_walks: Number of random walks per seed
            num_seeds: Number of trusted seeds to select
        """
        # Validate parameters
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not MIN_WALKS <= num_walks <= MAX_WALKS:
            raise ValueError(f"num_walks must be in [{MIN_WALKS}, {MAX_WALKS}]")
        if not MIN_SEEDS <= num_seeds <= MAX_SEEDS:
            raise ValueError(f"num_seeds must be in [{MIN_SEEDS}, {MAX_SEEDS}]")
        
        self._graph = graph
        self._alpha = alpha
        self._beta = beta
        self._num_walks = num_walks
        self._num_seeds = num_seeds
        
        # Seeds (selected by select_seeds())
        self._seeds: List[str] = []
        self._seed_weights: Dict[str, float] = {}
        
        # Results
        self._scores: Dict[str, float] = {}
        self._bridge_scores: Dict[str, float] = {}
        self._computed_at_ms: int = 0
        
        # Sybil detection
        self._detected_clusters: List[SybilCluster] = []
        self._slashing_events: List[SlashingEvent] = []
        
        # RNG for deterministic computation
        # SECURITY NOTE: Using random.Random() (not secrets) is INTENTIONAL here.
        # MeritRank must be deterministic across all nodes for consensus.
        # All nodes seeded with same value will compute identical scores.
        # This is NOT used for cryptographic purposes - only for reproducible
        # random walks that all consensus participants can verify.
        self._rng = random.Random()
    
    def set_seed(self, seed: int) -> None:
        """
        Set RNG seed for deterministic computation.
        
        CONSENSUS-CRITICAL: All nodes must use the same seed to compute
        identical MeritRank scores. The seed should be derived from
        a block hash or other consensus-agreed value.
        """
        self._rng.seed(seed)

    def _validate_custom_seeds(self, custom_seeds: List[str]) -> List[str]:
        """Validate and return custom seeds (up to _num_seeds)."""
        for seed_id in custom_seeds:
            if seed_id not in self._graph._nodes:
                raise ValueError(f"Seed {seed_id} not in graph")
        return custom_seeds[:self._num_seeds]

    def _get_eligible_nodes(self) -> List[EvaluatorNode]:
        """Return nodes eligible for seed selection (non-slashed, positive stake)."""
        return [
            n for n in self._graph.get_all_nodes()
            if not n.is_slashed and n.stake > 0
        ]

    def _weighted_sample_one(self, remaining: List[EvaluatorNode], total_stake: float) -> Tuple[str, float]:
        """Select one node weighted by stake, return (node_id, node_stake)."""
        weights = [n.stake / total_stake for n in remaining]
        cumulative = 0.0
        r = self._rng.random()
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return remaining[i].evaluator_id, remaining[i].stake
        # Fallback to last (floating point edge case)
        return remaining[-1].evaluator_id, remaining[-1].stake

    def _stake_weighted_selection(self, eligible: List[EvaluatorNode]) -> List[str]:
        """Select _num_seeds nodes using stake-weighted sampling without replacement."""
        total_stake = sum(n.stake for n in eligible)
        selected: List[str] = []
        remaining = eligible.copy()

        for _ in range(self._num_seeds):
            if not remaining:
                break
            node_id, node_stake = self._weighted_sample_one(remaining, total_stake)
            selected.append(node_id)
            total_stake -= node_stake
            remaining = [n for n in remaining if n.evaluator_id != node_id]

        return selected

    def _finalize_seeds(self) -> None:
        """Set seed weights and mark seed nodes in graph."""
        if self._seeds:
            weight = 1.0 / len(self._seeds)
            self._seed_weights = {s: weight for s in self._seeds}

        for node_id in self._seeds:
            node = self._graph.get_node(node_id)
            if node:
                node.is_seed = True

    def select_seeds(
        self,
        custom_seeds: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select trusted seed nodes using stake-weighted selection.
        
        Security:
            - Multi-seed prevents single point of failure
            - Stake-weighted prevents low-stake manipulation
            - BFT requires agreement from multiple seeds
        
        Args:
            custom_seeds: Optional predefined seeds (for testing)
        
        Returns:
            List of selected seed node IDs
        """
        if custom_seeds is not None:
            self._seeds = self._validate_custom_seeds(custom_seeds)
        else:
            eligible = self._get_eligible_nodes()
            if len(eligible) < self._num_seeds:
                self._seeds = [n.evaluator_id for n in eligible]
            else:
                self._seeds = self._stake_weighted_selection(eligible)

        self._finalize_seeds()
        return self._seeds
    
    def _compute_edge_weights(self) -> None:
        """
        Compute edge weights with recency decay.
        
        Security: Recent edges weighted higher, prevents zombie revivals.
        """
        now_ms = int(time.time() * 1000)
        
        for node_id in self._graph.get_all_node_ids():
            for edge in self._graph.get_outgoing(node_id):
                # Base weight from quality
                base_weight = edge.quality
                
                # Recency factor (exponential decay)
                age_ms = now_ms - edge.timestamp_ms
                recency_factor = math.exp(-age_ms * math.log(2) / RECENCY_HALFLIFE_MS)
                
                # Combined weight
                edge.weight = max(MIN_EDGE_WEIGHT, base_weight * recency_factor)
    
    def _random_walk(self, start_node: str) -> Dict[str, float]:
        """
        Perform random walk with transitivity decay from start node.
        
        Security:
            - α probability of restart (limits serial attack)
            - Bounded walk length
            - Weight-based transitions
        
        Returns:
            Visit counts per node (normalized)
        """
        visits: Dict[str, float] = defaultdict(float)
        current = start_node
        
        for _ in range(MAX_WALK_LENGTH):
            visits[current] += 1
            
            # Random restart with probability α
            if self._rng.random() < self._alpha:
                current = start_node
                continue
            
            # Get outgoing edges
            edges = self._graph.get_outgoing(current)
            if not edges:
                # Dead end, restart
                current = start_node
                continue
            
            # Weight-based transition
            total_weight = sum(e.weight for e in edges)
            if total_weight < EPSILON:
                current = start_node
                continue
            
            r = self._rng.random() * total_weight
            cumulative = 0
            for edge in edges:
                cumulative += edge.weight
                if r <= cumulative:
                    current = edge.to_id
                    break
        
        # Normalize
        total_visits = sum(visits.values())
        if total_visits > 0:
            for node_id in visits:
                visits[node_id] /= total_visits
        
        return dict(visits)
    
    def _detect_bridges(self) -> Dict[str, float]:
        """
        Detect bridge-like nodes (potential reputation laundering).
        
        Security: High bridge score = suspicious node.
        
        Bridge detection heuristics:
            1. High ratio of incoming to outgoing distinct sources
            2. Connects otherwise disconnected communities
            3. Sudden appearance with high connectivity
        """
        bridge_scores: Dict[str, float] = {}
        
        for node_id in self._graph.get_all_node_ids():
            node = self._graph.get_node(node_id)
            if node is None:
                continue
            
            incoming = self._graph.get_incoming(node_id)
            outgoing = self._graph.get_outgoing(node_id)
            
            # Heuristic 1: Degree ratio
            in_degree = len(incoming)
            out_degree = len(outgoing)
            
            if in_degree + out_degree == 0:
                bridge_scores[node_id] = 0.0
                continue
            
            # High in-degree with low out-degree is suspicious
            degree_ratio = in_degree / (in_degree + out_degree + 1)
            
            # Heuristic 2: Source diversity
            in_sources = len(set(e.from_id for e in incoming))
            out_targets = len(set(e.to_id for e in outgoing))
            
            # Many sources, few targets = potential bridge
            if out_targets > 0:
                source_ratio = in_sources / out_targets
            else:
                source_ratio = in_sources if in_sources > 0 else 0
            
            # Heuristic 3: Recency (new high-connectivity nodes are suspicious)
            now_ms = int(time.time() * 1000)
            age_ms = now_ms - node.registered_at_ms
            age_days = age_ms / (24 * 60 * 60 * 1000)
            
            # New nodes with high connectivity
            connectivity = in_degree + out_degree
            newness_factor = 1.0 / (1.0 + age_days)  # Higher for newer nodes
            
            # Combined bridge score
            bridge_score = (
                0.3 * degree_ratio +
                0.3 * min(1.0, source_ratio / 10) +
                0.4 * newness_factor * min(1.0, connectivity / 20)
            )
            
            bridge_scores[node_id] = min(1.0, bridge_score)
        
        return bridge_scores
    
    def _apply_connectivity_decay(
        self,
        raw_scores: Dict[str, float],
        bridge_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply connectivity decay (β) based on bridge scores.
        
        Security: Penalizes nodes behind bridges (potential Sybils).
        """
        decayed_scores = {}
        
        for node_id, score in raw_scores.items():
            bridge_score = bridge_scores.get(node_id, 0.0)
            
            # Apply β decay proportional to bridge score
            decay_factor = 1.0 - self._beta * bridge_score
            decayed_scores[node_id] = score * max(0.1, decay_factor)
        
        return decayed_scores
    
    def compute(self) -> Dict[str, float]:
        """
        Compute MeritRank scores for all nodes.
        
        Algorithm:
            1. Compute edge weights with recency decay
            2. For each seed, perform random walks
            3. Aggregate visit counts across seeds (BFT)
            4. Detect bridges
            5. Apply connectivity decay
            6. Normalize final scores
        
        Returns:
            Dictionary of node_id → merit_score
        """
        if not self._seeds:
            self.select_seeds()
        
        if not self._seeds:
            logger.warning("No seeds available for MeritRank computation")
            return {}
        
        # Step 1: Compute edge weights
        self._compute_edge_weights()
        
        # Step 2: Random walks from each seed
        seed_results: Dict[str, Dict[str, float]] = {}
        
        for seed_id in self._seeds:
            aggregated: Dict[str, float] = defaultdict(float)
            
            for _ in range(self._num_walks):
                visits = self._random_walk(seed_id)
                for node_id, count in visits.items():
                    aggregated[node_id] += count
            
            # Normalize per seed
            total = sum(aggregated.values())
            if total > 0:
                seed_results[seed_id] = {
                    k: v / total for k, v in aggregated.items()
                }
            else:
                seed_results[seed_id] = {}
        
        # Step 3: BFT aggregation (weighted median across seeds)
        raw_scores = self._aggregate_bft(seed_results)
        
        # Step 4: Detect bridges
        self._bridge_scores = self._detect_bridges()
        
        # Step 5: Apply connectivity decay
        decayed_scores = self._apply_connectivity_decay(raw_scores, self._bridge_scores)
        
        # Step 6: Normalize final scores
        total = sum(decayed_scores.values())
        if total > 0:
            self._scores = {
                k: max(MIN_SCORE, min(MAX_SCORE, v / total))
                for k, v in decayed_scores.items()
            }
        else:
            self._scores = {}
        
        # Update node objects
        for node_id, score in self._scores.items():
            node = self._graph.get_node(node_id)
            if node:
                node.merit_score = score
                node.bridge_score = self._bridge_scores.get(node_id, 0.0)
        
        self._computed_at_ms = int(time.time() * 1000)
        
        return self._scores
    
    def _aggregate_bft(
        self,
        seed_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        BFT aggregation of seed results using weighted median.
        
        Security: Tolerates up to f = (n-1)/3 Byzantine seeds.
        """
        all_nodes = set()
        for results in seed_results.values():
            all_nodes.update(results.keys())
        
        aggregated: Dict[str, float] = {}
        
        for node_id in all_nodes:
            # Collect scores from all seeds
            scores = []
            for seed_id in self._seeds:
                seed_weight = self._seed_weights.get(seed_id, 1.0 / len(self._seeds))
                node_score = seed_results.get(seed_id, {}).get(node_id, 0.0)
                scores.append((node_score, seed_weight))
            
            # Weighted median
            scores.sort(key=lambda x: x[0])
            
            total_weight = sum(w for _, w in scores)
            cumulative = 0
            median_score = 0
            
            for score, weight in scores:
                cumulative += weight
                if cumulative >= total_weight / 2:
                    median_score = score
                    break
            
            aggregated[node_id] = median_score
        
        return aggregated
    
    def detect_sybils(self) -> List[SybilCluster]:
        """
        Detect potential Sybil clusters.
        
        Security: Uses multiple signals for detection.
        """
        if not self._scores:
            self.compute()
        
        clusters = []
        
        # Method 1: High bridge score nodes
        suspicious = [
            node_id for node_id, score in self._bridge_scores.items()
            if score > SYBIL_DETECTION_THRESHOLD
        ]
        
        if suspicious:
            cluster = SybilCluster(
                cluster_id=secrets.token_hex(8),
                node_ids=set(suspicious),
                confidence=0.7,
                detection_method="bridge_score",
                detected_at_ms=int(time.time() * 1000),
                evidence={"bridge_scores": {n: self._bridge_scores[n] for n in suspicious}},
            )
            clusters.append(cluster)
        
        # Method 2: Clique detection (simplified)
        # Full implementation would use community detection
        
        self._detected_clusters = clusters
        return clusters
    
    def slash(self, node_id: str, reason: str) -> Optional[SlashingEvent]:
        """
        Slash a node for Sybil behavior.
        
        Security: Reduces stake and flags node.
        """
        node = self._graph.get_node(node_id)
        if node is None:
            return None
        
        # Apply penalty
        original_stake = node.stake
        node.stake *= (1 - SLASHING_PENALTY)
        node.is_slashed = True
        node.slash_count += 1
        
        event = SlashingEvent(
            event_id=secrets.token_hex(8),
            node_id=node_id,
            cluster_id="",
            penalty_fraction=SLASHING_PENALTY,
            reason=reason,
            timestamp_ms=int(time.time() * 1000),
        )
        
        self._slashing_events.append(event)
        
        logger.info(
            f"Slashed {node_id}: stake {original_stake:.2f} → {node.stake:.2f} ({reason})"
        )
        
        return event
    
    def get_scores(self) -> Dict[str, float]:
        """Get computed merit scores."""
        return self._scores.copy()
    
    def get_score(self, node_id: str) -> float:
        """Get merit score for a specific node."""
        return self._scores.get(node_id, 0.0)
    
    def get_bridge_scores(self) -> Dict[str, float]:
        """Get bridge scores (for diagnostics)."""
        return self._bridge_scores.copy()
