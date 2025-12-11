"""
De-duplication service for IAN contributions.

Provides two-tier de-duplication:
1. BloomFilter: Fast probabilistic pre-filter (O(1), compact)
2. DedupIndex: Authoritative hash index (O(1) lookup)

This combination gives O(1) average-case de-dup with bounded false-positive fallback.

Complexity:
- BloomFilter:
  - Add: O(k) where k = number of hash functions
  - Check: O(k)
  - Space: O(m) bits where m = filter size
  
- DedupIndex:
  - Add: O(1) amortized
  - Check: O(1)
  - Space: O(n) where n = number of entries
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Set


class BloomFilter:
    """
    Probabilistic data structure for approximate set membership.
    
    Properties:
    - No false negatives: if check returns False, the item is definitely not in the set
    - Possible false positives: if check returns True, the item might be in the set
    - Compact: uses much less memory than storing all items
    
    The false positive rate depends on:
    - m: number of bits in the filter
    - k: number of hash functions
    - n: number of items inserted
    
    Optimal k = (m/n) * ln(2)
    False positive rate ≈ (1 - e^(-kn/m))^k
    """
    
    def __init__(
        self,
        expected_items: int = 100_000,
        false_positive_rate: float = 0.01,
    ) -> None:
        """
        Initialize Bloom filter with target false positive rate.
        
        Args:
            expected_items: Expected number of items to store
            false_positive_rate: Target false positive probability (0 < p < 1)
        """
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be in (0, 1)")
        
        # Calculate optimal size and number of hash functions
        # m = -n * ln(p) / (ln(2)^2)
        # k = (m/n) * ln(2)
        self._n_expected = expected_items
        self._fp_rate = false_positive_rate
        
        ln2 = math.log(2)
        self._m = int(-expected_items * math.log(false_positive_rate) / (ln2 ** 2))
        self._m = max(self._m, 64)  # Minimum size
        
        self._k = max(1, int((self._m / expected_items) * ln2))
        self._k = min(self._k, 16)  # Cap at 16 hash functions
        
        # Bit array (using bytearray for efficiency)
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0
    
    def _get_hash_positions(self, key: bytes) -> list[int]:
        """
        Compute k hash positions for a key.
        
        Uses double hashing: h(i) = h1 + i * h2 (mod m)
        """
        # Use SHA-256 and split into two 128-bit halves
        digest = hashlib.sha256(key).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")
        
        positions = []
        for i in range(self._k):
            pos = (h1 + i * h2) % self._m
            positions.append(pos)
        return positions
    
    def _set_bit(self, pos: int) -> None:
        """Set bit at position."""
        byte_idx = pos // 8
        bit_idx = pos % 8
        self._bits[byte_idx] |= (1 << bit_idx)
    
    def _get_bit(self, pos: int) -> bool:
        """Get bit at position."""
        byte_idx = pos // 8
        bit_idx = pos % 8
        return bool(self._bits[byte_idx] & (1 << bit_idx))
    
    def add(self, key: bytes) -> "BloomFilter":
        """
        Add a key to the filter.
        
        Args:
            key: Key to add (typically a hash)
            
        Returns:
            self for chaining
        """
        for pos in self._get_hash_positions(key):
            self._set_bit(pos)
        self._count += 1
        return self
    
    def maybe_contains(self, key: bytes) -> bool:
        """
        Check if key might be in the filter.
        
        Args:
            key: Key to check
            
        Returns:
            False = definitely not in set
            True = probably in set (may be false positive)
        """
        for pos in self._get_hash_positions(key):
            if not self._get_bit(pos):
                return False
        return True
    
    def __contains__(self, key: bytes) -> bool:
        """Allow `key in bloom_filter` syntax."""
        return self.maybe_contains(key)
    
    @property
    def count(self) -> int:
        """Number of items added (approximate)."""
        return self._count
    
    @property
    def estimated_false_positive_rate(self) -> float:
        """Estimate current false positive rate based on items added."""
        if self._count == 0:
            return 0.0
        # FPR ≈ (1 - e^(-kn/m))^k
        exp_term = math.exp(-self._k * self._count / self._m)
        return (1 - exp_term) ** self._k
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "m": self._m,
            "k": self._k,
            "count": self._count,
            "bits": self._bits.hex(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BloomFilter":
        """Deserialize from stored state."""
        bf = cls.__new__(cls)
        bf._m = data["m"]
        bf._k = data["k"]
        bf._count = data["count"]
        bf._bits = bytearray.fromhex(data["bits"])
        bf._n_expected = bf._count or 100_000
        bf._fp_rate = 0.01
        return bf
    
    def __repr__(self) -> str:
        return f"BloomFilter(count={self._count}, m={self._m}, k={self._k}, est_fpr={self.estimated_false_positive_rate:.4f})"


class DedupIndex:
    """
    Authoritative hash index for de-duplication.
    
    Maps pack_hash -> log_index for all accepted contributions.
    """
    
    def __init__(self) -> None:
        self._index: Dict[bytes, int] = {}
    
    def add(self, pack_hash: bytes, log_index: int) -> None:
        """
        Add a pack_hash to the index.
        
        Args:
            pack_hash: 32-byte hash of the AgentPack
            log_index: Position in the experiment log
        """
        if len(pack_hash) != 32:
            raise ValueError(f"pack_hash must be 32 bytes, got {len(pack_hash)}")
        self._index[pack_hash] = log_index
    
    def contains(self, pack_hash: bytes) -> bool:
        """Check if pack_hash is in the index."""
        return pack_hash in self._index
    
    def get_log_index(self, pack_hash: bytes) -> Optional[int]:
        """Get log index for a pack_hash, or None if not found."""
        return self._index.get(pack_hash)
    
    def __contains__(self, pack_hash: bytes) -> bool:
        return pack_hash in self._index
    
    def __len__(self) -> int:
        return len(self._index)
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "entries": {h.hex(): idx for h, idx in self._index.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DedupIndex":
        """Deserialize from stored state."""
        di = cls()
        for h_hex, idx in data["entries"].items():
            di._index[bytes.fromhex(h_hex)] = idx
        return di
    
    def __repr__(self) -> str:
        return f"DedupIndex(size={len(self)})"


class DedupService:
    """
    Two-tier de-duplication service.
    
    Combines BloomFilter (fast, probabilistic) with DedupIndex (authoritative).
    
    Algorithm:
    1. Check BloomFilter first
    2. If Bloom says "not present" -> definitely new (fast path)
    3. If Bloom says "maybe present" -> check DedupIndex (authoritative)
    
    This gives O(1) average case with bounded false positive fallback.
    """
    
    def __init__(
        self,
        expected_contributions: int = 100_000,
        bloom_fp_rate: float = 0.01,
    ) -> None:
        """
        Initialize de-duplication service.
        
        Args:
            expected_contributions: Expected number of contributions
            bloom_fp_rate: False positive rate for Bloom filter
        """
        self._bloom = BloomFilter(
            expected_items=expected_contributions,
            false_positive_rate=bloom_fp_rate,
        )
        self._index = DedupIndex()
    
    def is_duplicate(self, pack_hash: bytes) -> bool:
        """
        Check if pack_hash is a duplicate.
        
        Args:
            pack_hash: 32-byte hash of the AgentPack
            
        Returns:
            True if this pack has already been seen
            
        Complexity: O(1) average
        """
        # Fast path: Bloom filter says definitely not present
        if not self._bloom.maybe_contains(pack_hash):
            return False
        
        # Bloom says maybe present -> check authoritative index
        return self._index.contains(pack_hash)
    
    def add(self, pack_hash: bytes, log_index: int) -> None:
        """
        Record a new pack_hash as seen.
        
        Args:
            pack_hash: 32-byte hash of the AgentPack
            log_index: Position in the experiment log
        """
        self._bloom.add(pack_hash)
        self._index.add(pack_hash, log_index)
    
    def get_log_index(self, pack_hash: bytes) -> Optional[int]:
        """Get log index for a pack_hash, or None if not found."""
        return self._index.get_log_index(pack_hash)
    
    @property
    def count(self) -> int:
        """Number of items in the index."""
        return len(self._index)
    
    def __contains__(self, pack_hash: bytes) -> bool:
        return self.is_duplicate(pack_hash)
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "bloom": self._bloom.to_dict(),
            "index": self._index.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DedupService":
        """Deserialize from stored state."""
        ds = cls.__new__(cls)
        ds._bloom = BloomFilter.from_dict(data["bloom"])
        ds._index = DedupIndex.from_dict(data["index"])
        return ds
    
    def __repr__(self) -> str:
        return f"DedupService(count={self.count}, bloom_fpr={self._bloom.estimated_false_positive_rate:.4f})"
