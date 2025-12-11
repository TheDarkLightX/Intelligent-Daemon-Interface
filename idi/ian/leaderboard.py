"""
Leaderboard implementations for IAN.

Provides two ranking strategies:
1. Leaderboard: Scalar top-K using min-heap
2. ParetoFrontier: Multi-objective non-dominated set

Both maintain bounded size and provide deterministic ranking.

Complexity:
- Leaderboard (top-K heap):
  - Insert: O(log K)
  - Query top-K: O(K log K) for sorted, O(K) unsorted
  - Space: O(K)
  
- ParetoFrontier:
  - Insert: O(K) domination check
  - Query frontier: O(K)
  - Space: O(K) bounded or O(N) unbounded
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import ContributionMeta, Metrics


class Leaderboard:
    """
    Top-K leaderboard using a min-heap.
    
    The heap root contains the *lowest* score among the top K.
    When a new candidate arrives with score > root, we replace root.
    
    Invariant:
    - After processing contribution i, the heap contains exactly the top-K
      contributions by score among all valid contributions [0..i].
    
    Tie-breaking: Earlier submissions (lower timestamp) win ties.
    """
    
    def __init__(self, capacity: int = 100) -> None:
        """
        Initialize leaderboard.
        
        Args:
            capacity: Maximum number of entries (K)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        # Heap entries: (score, -timestamp, pack_hash_hex)
        # We negate timestamp so earlier entries sort higher on ties
        self._heap: List[Tuple[float, int, str]] = []
        # Full metadata indexed by pack_hash
        self._entries: Dict[bytes, ContributionMeta] = {}
    
    def add(self, meta: ContributionMeta) -> bool:
        """
        Attempt to add a candidate to the leaderboard.
        
        Args:
            meta: Contribution metadata with score
            
        Returns:
            True if added (either new entry or replaced worse entry)
            
        Complexity: O(log K)
        """
        pack_hash = meta.pack_hash
        score = meta.score
        timestamp = meta.timestamp_ms
        
        # Already in leaderboard?
        if pack_hash in self._entries:
            return False
        
        heap_entry = (score, -timestamp, pack_hash.hex())
        
        if len(self._heap) < self.capacity:
            # Room available
            heapq.heappush(self._heap, heap_entry)
            self._entries[pack_hash] = meta
            return True
        
        # Check if better than worst (root of min-heap)
        worst_score, _, _ = self._heap[0]
        if score > worst_score:
            # Replace worst
            _, _, removed_hash_hex = heapq.heapreplace(self._heap, heap_entry)
            removed_hash = bytes.fromhex(removed_hash_hex)
            del self._entries[removed_hash]
            self._entries[pack_hash] = meta
            return True
        elif score == worst_score:
            # Tie: prefer earlier timestamp
            worst_neg_ts = self._heap[0][1]
            if -timestamp > worst_neg_ts:  # Earlier timestamp = larger negative
                _, _, removed_hash_hex = heapq.heapreplace(self._heap, heap_entry)
                removed_hash = bytes.fromhex(removed_hash_hex)
                del self._entries[removed_hash]
                self._entries[pack_hash] = meta
                return True
        
        return False
    
    def top_k(self, sorted_desc: bool = True) -> List[ContributionMeta]:
        """
        Get the top-K entries.
        
        Args:
            sorted_desc: If True, sort by score descending
            
        Returns:
            List of ContributionMeta
            
        Complexity: O(K log K) if sorted, O(K) otherwise
        """
        entries = list(self._entries.values())
        if sorted_desc:
            entries.sort(key=lambda m: (-m.score, m.timestamp_ms))
        return entries
    
    def worst_score(self) -> Optional[float]:
        """
        Get the score of the worst entry in the leaderboard.
        
        Returns:
            Score of worst entry, or None if empty
            
        Complexity: O(1)
        """
        if not self._heap:
            return None
        return self._heap[0][0]
    
    def best_score(self) -> Optional[float]:
        """
        Get the score of the best entry in the leaderboard.
        
        Returns:
            Score of best entry, or None if empty
            
        Complexity: O(K)
        """
        if not self._entries:
            return None
        return max(m.score for m in self._entries.values())
    
    def contains(self, pack_hash: bytes) -> bool:
        """Check if pack_hash is in the leaderboard."""
        return pack_hash in self._entries
    
    def get(self, pack_hash: bytes) -> Optional[ContributionMeta]:
        """Get entry by pack_hash."""
        return self._entries.get(pack_hash)
    
    def get_active_policy(self) -> Optional[ContributionMeta]:
        """
        Get the current best (active) policy.
        
        Returns:
            Best ContributionMeta or None if empty
        """
        if not self._entries:
            return None
        return max(self._entries.values(), key=lambda m: (m.score, -m.timestamp_ms))
    
    def get_root(self) -> bytes:
        """
        Compute a deterministic hash of the leaderboard state.
        
        Used for anchoring leaderboard state to Tau.
        """
        if not self._entries:
            return b"\x00" * 32
        
        # Sort entries deterministically
        sorted_hashes = sorted(self._entries.keys())
        hasher = hashlib.sha256()
        for h in sorted_hashes:
            hasher.update(h)
            hasher.update(str(self._entries[h].score).encode())
        return hasher.digest()
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __contains__(self, pack_hash: bytes) -> bool:
        return pack_hash in self._entries
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "capacity": self.capacity,
            "entries": [meta.to_dict() for meta in self._entries.values()],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Leaderboard":
        """Deserialize from stored state."""
        lb = cls(capacity=data["capacity"])
        for entry_data in data["entries"]:
            meta = ContributionMeta.from_dict(entry_data)
            lb.add(meta)
        return lb
    
    def __repr__(self) -> str:
        return f"Leaderboard(size={len(self)}, capacity={self.capacity}, best={self.best_score()})"


class ParetoFrontier:
    """
    Multi-objective leaderboard maintaining the Pareto frontier.
    
    A contribution c1 dominates c2 iff:
    - c1.reward >= c2.reward AND
    - c1.risk <= c2.risk AND
    - c1.complexity <= c2.complexity AND
    - at least one inequality is strict
    
    The frontier contains all non-dominated contributions.
    
    Complexity:
    - Insert: O(K) to check domination
    - Query: O(K)
    - Space: O(K) if bounded, O(N) if unbounded
    """
    
    def __init__(self, max_size: Optional[int] = None) -> None:
        """
        Initialize Pareto frontier.
        
        Args:
            max_size: Optional capacity limit. If set, dominated entries
                      are removed first, then worst by tie-breaker.
        """
        self.max_size = max_size
        self._entries: Dict[bytes, ContributionMeta] = {}
    
    @staticmethod
    def dominates(m1: Metrics, m2: Metrics) -> bool:
        """
        Check if m1 dominates m2.
        
        m1 dominates m2 iff:
        - m1 is at least as good as m2 in all objectives
        - m1 is strictly better in at least one objective
        """
        # Higher reward is better
        # Lower risk is better
        # Lower complexity is better
        at_least_as_good = (
            m1.reward >= m2.reward and
            m1.risk <= m2.risk and
            m1.complexity <= m2.complexity
        )
        strictly_better = (
            m1.reward > m2.reward or
            m1.risk < m2.risk or
            m1.complexity < m2.complexity
        )
        return at_least_as_good and strictly_better
    
    def add(self, meta: ContributionMeta) -> bool:
        """
        Attempt to add a candidate to the Pareto frontier.
        
        Args:
            meta: Contribution metadata
            
        Returns:
            True if added to frontier (not dominated by existing entries)
            
        Complexity: O(K) where K is current frontier size
        """
        pack_hash = meta.pack_hash
        
        # Already in frontier?
        if pack_hash in self._entries:
            return False
        
        # Check if dominated by any existing entry
        for existing in self._entries.values():
            if self.dominates(existing.metrics, meta.metrics):
                return False  # Dominated, reject
        
        # Remove entries dominated by new one
        to_remove = []
        for h, existing in self._entries.items():
            if self.dominates(meta.metrics, existing.metrics):
                to_remove.append(h)
        
        for h in to_remove:
            del self._entries[h]
        
        # Add new entry
        self._entries[pack_hash] = meta
        
        # Enforce max_size if set
        if self.max_size is not None and len(self._entries) > self.max_size:
            self._evict_worst()
        
        return True
    
    def _evict_worst(self) -> None:
        """Remove the 'worst' entry by tie-breaker (lowest reward, then highest risk)."""
        if not self._entries:
            return
        
        worst_hash = min(
            self._entries.keys(),
            key=lambda h: (
                self._entries[h].metrics.reward,
                -self._entries[h].metrics.risk,
                -self._entries[h].timestamp_ms,
            )
        )
        del self._entries[worst_hash]
    
    def frontier(self) -> List[ContributionMeta]:
        """
        Get all entries on the Pareto frontier.
        
        Returns:
            List of non-dominated ContributionMeta
        """
        return list(self._entries.values())
    
    def contains(self, pack_hash: bytes) -> bool:
        """Check if pack_hash is in the frontier."""
        return pack_hash in self._entries
    
    def get(self, pack_hash: bytes) -> Optional[ContributionMeta]:
        """Get entry by pack_hash."""
        return self._entries.get(pack_hash)
    
    def get_root(self) -> bytes:
        """Compute deterministic hash of frontier state."""
        if not self._entries:
            return b"\x00" * 32
        
        sorted_hashes = sorted(self._entries.keys())
        hasher = hashlib.sha256()
        for h in sorted_hashes:
            hasher.update(h)
        return hasher.digest()
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __contains__(self, pack_hash: bytes) -> bool:
        return pack_hash in self._entries
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "max_size": self.max_size,
            "entries": [meta.to_dict() for meta in self._entries.values()],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ParetoFrontier":
        """Deserialize from stored state."""
        pf = cls(max_size=data.get("max_size"))
        for entry_data in data["entries"]:
            meta = ContributionMeta.from_dict(entry_data)
            pf._entries[meta.pack_hash] = meta  # Direct add to avoid re-checking
        return pf
    
    def __repr__(self) -> str:
        return f"ParetoFrontier(size={len(self)}, max_size={self.max_size})"
