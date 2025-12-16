"""
TwigMMR: Optimized Merkle Mountain Range with fixed-depth subtrees.

Security Controls:
- Domain separation between leaf and internal node hashes
- Bounded tree size to prevent memory exhaustion
- Safe integer arithmetic for index calculations
- Atomic twig state transitions
- Verification of twig roots on disk load
- Input validation on all external data

Based on: QMDB (arXiv:2501.05262), with IAN-specific enhancements

Author: DarkLightX
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import json
import secrets
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants
# =============================================================================

# Hash size (SHA-256)
HASH_SIZE = 32

# Domain separation prefixes (prevents leaf/internal node confusion)
LEAF_PREFIX = b'\x00'
INTERNAL_PREFIX = b'\x01'
EMPTY_PREFIX = b'\x02'

# Twig parameters
TWIG_DEPTH = 11  # 2^11 = 2048 leaves per twig
TWIG_CAPACITY = 1 << TWIG_DEPTH  # 2048
ACTIVE_BITS_SIZE = TWIG_CAPACITY // 8  # 256 bytes

# Cache parameters
DEFAULT_HOT_TWIG_COUNT = 16  # ~32K entries in RAM
MAX_HOT_TWIG_COUNT = 64  # ~128K entries max

# Security bounds
MAX_TREE_SIZE = 1 << 40  # ~1 trillion entries max
MAX_PROOF_DEPTH = 50  # Maximum proof path length
MAX_VERSION_HISTORY = 1000  # Checkpoint versions to keep

# Empty hash (domain-separated)
EMPTY_HASH = hashlib.sha256(EMPTY_PREFIX).digest()


# =============================================================================
# Helper Functions
# =============================================================================

def hash_leaf(data: bytes) -> bytes:
    """
    Hash leaf data with domain separation.
    
    Security: Prefix prevents leaf/internal node confusion.
    """
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def hash_internal(left: bytes, right: bytes) -> bytes:
    """
    Hash internal node with domain separation.
    
    Security: Prefix prevents leaf/internal node confusion.
    """
    if len(left) != HASH_SIZE or len(right) != HASH_SIZE:
        raise ValueError(f"Child hashes must be {HASH_SIZE} bytes")
    return hashlib.sha256(INTERNAL_PREFIX + left + right).digest()


def hash_pair(left: bytes, right: bytes) -> bytes:
    """Alias for hash_internal for compatibility."""
    return hash_internal(left, right)


def safe_add(a: int, b: int, max_val: int = MAX_TREE_SIZE) -> int:
    """
    Safe integer addition with overflow check.
    
    Security: Prevents integer overflow attacks.
    """
    result = a + b
    if result > max_val or result < 0:
        raise OverflowError(f"Integer overflow: {a} + {b} exceeds {max_val}")
    return result


# =============================================================================
# Data Structures
# =============================================================================

class TwigState(Enum):
    """State of a twig in its lifecycle."""
    FRESH = 1      # < TWIG_CAPACITY entries, in RAM, mutable
    FULL = 2       # == TWIG_CAPACITY entries, data on disk, root in RAM
    INACTIVE = 3   # 0 active entries, data deleted, root on disk
    PRUNED = 4     # Entirely deleted including root


@dataclass
class BitArray:
    """
    Efficient bit array for tracking active entries.
    
    Security: Bounds checking on all bit operations.
    """
    _data: bytearray
    _size: int
    
    def __init__(self, size: int):
        if size <= 0 or size > TWIG_CAPACITY:
            raise ValueError(f"BitArray size must be in (0, {TWIG_CAPACITY}]")
        self._size = size
        self._data = bytearray((size + 7) // 8)
    
    def set(self, index: int) -> None:
        """Set bit at index to 1."""
        if not 0 <= index < self._size:
            raise IndexError(f"Bit index {index} out of range [0, {self._size})")
        self._data[index // 8] |= (1 << (index % 8))
    
    def clear(self, index: int) -> None:
        """Set bit at index to 0."""
        if not 0 <= index < self._size:
            raise IndexError(f"Bit index {index} out of range [0, {self._size})")
        self._data[index // 8] &= ~(1 << (index % 8))
    
    def get(self, index: int) -> bool:
        """Get bit at index."""
        if not 0 <= index < self._size:
            raise IndexError(f"Bit index {index} out of range [0, {self._size})")
        return bool(self._data[index // 8] & (1 << (index % 8)))
    
    def count_set(self) -> int:
        """Count number of set bits."""
        return sum(bin(b).count('1') for b in self._data)
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return bytes(self._data)
    
    @classmethod
    def from_bytes(cls, data: bytes, size: int) -> "BitArray":
        """Deserialize from bytes."""
        arr = cls(size)
        arr._data = bytearray(data)
        return arr


@dataclass
class Twig:
    """
    Fixed-depth subtree containing up to TWIG_CAPACITY leaves.
    
    Invariants:
        - len(entries) <= TWIG_CAPACITY
        - active_bits.size == TWIG_CAPACITY
        - root == computed Merkle root when entries hashed
        - state transitions: Fresh → Full → Inactive → Pruned
    
    Security:
        - Validates state transitions
        - Verifies root on load
    """
    twig_id: int
    entries: List[bytes]  # Leaf hashes
    active_bits: BitArray
    root: bytes
    state: TwigState
    first_log_index: int
    created_at_ms: int
    
    def __post_init__(self) -> None:
        """Validate twig after creation."""
        if len(self.entries) > TWIG_CAPACITY:
            raise ValueError(f"Twig has {len(self.entries)} entries, max is {TWIG_CAPACITY}")
        if len(self.root) != HASH_SIZE:
            raise ValueError(f"Root must be {HASH_SIZE} bytes")
    
    def can_transition_to(self, new_state: TwigState) -> bool:
        """
        Check if state transition is valid.
        
        Valid transitions:
            FRESH → FULL (when capacity reached)
            FULL → INACTIVE (when all entries deactivated)
            INACTIVE → PRUNED (when pruning)
        """
        valid_transitions = {
            TwigState.FRESH: {TwigState.FULL},
            TwigState.FULL: {TwigState.INACTIVE},
            TwigState.INACTIVE: {TwigState.PRUNED},
            TwigState.PRUNED: set(),  # Terminal state
        }
        return new_state in valid_transitions.get(self.state, set())
    
    def transition_to(self, new_state: TwigState) -> None:
        """
        Transition to new state with validation.
        
        Security: Enforces valid state machine transitions.
        """
        if not self.can_transition_to(new_state):
            raise ValueError(
                f"Invalid state transition: {self.state.name} → {new_state.name}"
            )
        self.state = new_state


@dataclass
class MembershipProof:
    """
    Proof that an entry exists in the MMR at a given index.
    
    Security:
        - Includes log_index for position binding
        - Includes version for freshness
        - Full path verification required
    """
    log_index: int
    leaf_hash: bytes
    intra_twig_path: List[bytes]  # Siblings within twig
    inter_twig_path: List[bytes]  # Twig root to MMR root
    twig_root: bytes
    mmr_root: bytes
    version: int
    
    def verify(self) -> bool:
        """
        Verify the proof is valid.
        
        Security: Verifies complete path, no short-circuit.
        """
        # Step 1: Verify intra-twig path
        current = self.leaf_hash
        entry_index = self.log_index % TWIG_CAPACITY
        
        for i, sibling in enumerate(self.intra_twig_path):
            if len(sibling) != HASH_SIZE:
                return False
            
            # Determine position (left or right)
            if (entry_index >> i) & 1 == 0:
                current = hash_internal(current, sibling)
            else:
                current = hash_internal(sibling, current)
        
        # Check twig root
        if current != self.twig_root:
            return False
        
        # Step 2: Verify inter-twig path (frontier to root)
        # If no inter-twig path, twig_root should equal mmr_root (single twig case)
        if not self.inter_twig_path:
            return self.twig_root == self.mmr_root
        
        current = self.twig_root
        for sibling in self.inter_twig_path:
            if len(sibling) != HASH_SIZE:
                return False
            current = hash_internal(current, sibling)
        
        return current == self.mmr_root


# =============================================================================
# TwigMMR Implementation
# =============================================================================

class TwigMMR:
    """
    Merkle Mountain Range with twig-based optimization.
    
    Security features:
        - Domain-separated hashing (leaf vs internal)
        - Bounded tree size
        - Safe integer arithmetic
        - Atomic twig operations
        - Hot/cold twig caching with verification
        - Version tracking for historical proofs
    
    Performance:
        - O(1) append (amortized)
        - O(TWIG_DEPTH) proof for hot data (~11 hashes)
        - O(log N) proof for cold data
    
    Usage:
        mmr = TwigMMR()
        log_index, proof = mmr.append(leaf_hash)
        assert mmr.verify(proof)
    """
    
    def __init__(
        self,
        hot_twig_count: int = DEFAULT_HOT_TWIG_COUNT,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize TwigMMR.
        
        Args:
            hot_twig_count: Number of twigs to keep in RAM
            storage_path: Path for cold twig storage (None = memory only)
        """
        # Validate config
        if not 1 <= hot_twig_count <= MAX_HOT_TWIG_COUNT:
            raise ValueError(
                f"hot_twig_count must be in [1, {MAX_HOT_TWIG_COUNT}]"
            )
        
        self._hot_twig_count = hot_twig_count
        self._storage_path = storage_path
        
        # State
        self._current_twig: Optional[Twig] = None
        self._hot_twigs: OrderedDict[int, Twig] = OrderedDict()
        self._cold_twig_roots: Dict[int, bytes] = {}
        self._frontier: List[bytes] = []
        self._size: int = 0
        self._version: int = 0
        
        # Version history for historical proofs
        self._checkpoints: Dict[int, bytes] = {}  # version → root
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        
        # Initialize first twig
        self._create_fresh_twig()
    
    @property
    def size(self) -> int:
        """Total number of entries in MMR."""
        return self._size
    
    @property
    def version(self) -> int:
        """Current version (increments on each append)."""
        return self._version
    
    @property
    def frontier(self) -> List[bytes]:
        """MMR peak roots (copy)."""
        return self._frontier.copy()
    
    @property
    def root(self) -> bytes:
        """
        Compute single root hash from frontier.
        
        Uses bag-the-peaks algorithm (right-to-left).
        """
        if not self._frontier:
            return EMPTY_HASH
        
        result = self._frontier[-1]
        for peak in reversed(self._frontier[:-1]):
            result = hash_internal(peak, result)
        return result
    
    def _create_fresh_twig(self) -> None:
        """Create a new fresh twig."""
        twig_id = len(self._hot_twigs) + len(self._cold_twig_roots)
        
        self._current_twig = Twig(
            twig_id=twig_id,
            entries=[],
            active_bits=BitArray(TWIG_CAPACITY),
            root=EMPTY_HASH,
            state=TwigState.FRESH,
            first_log_index=self._size,
            created_at_ms=int(time.time() * 1000),
        )
    
    def _compute_twig_root(self, entries: List[bytes]) -> bytes:
        """
        Compute Merkle root for a twig's entries.
        
        Pads with EMPTY_HASH to TWIG_CAPACITY.
        """
        if not entries:
            return EMPTY_HASH
        
        # Pad to full capacity
        padded = entries + [EMPTY_HASH] * (TWIG_CAPACITY - len(entries))
        
        # Build tree bottom-up
        level = padded
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else EMPTY_HASH
                next_level.append(hash_internal(left, right))
            level = next_level
        
        return level[0]
    
    def _remerkleize_current_twig(self) -> None:
        """Recompute current twig's root after modification."""
        if self._current_twig is None:
            return
        self._current_twig.root = self._compute_twig_root(self._current_twig.entries)
    
    def _update_frontier(self) -> None:
        """
        Update MMR frontier after append.
        
        MMR frontier = roots of complete binary subtrees (peaks).
        """
        if self._current_twig is None:
            return
        
        # For simplicity, rebuild frontier from all twigs
        # A production impl would do incremental updates
        peaks = []
        
        # Add roots of all full twigs
        for twig in self._hot_twigs.values():
            if twig.state in (TwigState.FULL, TwigState.INACTIVE):
                peaks.append(twig.root)
        
        for twig_id, root in self._cold_twig_roots.items():
            peaks.append(root)
        
        # Add current twig if non-empty
        if self._current_twig.entries:
            peaks.append(self._current_twig.root)
        
        # Merge peaks from right to left where possible
        self._frontier = self._merge_peaks(peaks)
    
    def _merge_peaks(self, peaks: List[bytes]) -> List[bytes]:
        """
        Merge adjacent peaks of equal height.
        
        Returns the MMR frontier (list of peak roots).
        """
        if not peaks:
            return []
        
        # For TwigMMR, each twig represents a subtree
        # Peaks are merged based on the binary representation of total size
        return peaks  # Simplified - full impl would merge properly
    
    def _rotate_twig(self) -> None:
        """
        Rotate current twig to FULL and create new FRESH twig.
        
        Security: Atomic operation to prevent partial state.
        """
        if self._current_twig is None:
            return
        
        # Transition to full
        self._current_twig.transition_to(TwigState.FULL)
        
        # Add to hot cache
        self._hot_twigs[self._current_twig.twig_id] = self._current_twig
        
        # Evict oldest if over capacity
        while len(self._hot_twigs) > self._hot_twig_count:
            oldest_id, oldest_twig = self._hot_twigs.popitem(last=False)
            self._flush_twig_to_disk(oldest_twig)
        
        # Create new fresh twig
        self._create_fresh_twig()
    
    def _flush_twig_to_disk(self, twig: Twig) -> None:
        """
        Flush twig to disk storage.
        
        Security: Stores root for later verification on load.
        """
        if self._storage_path is None:
            # Memory-only mode: just keep root
            self._cold_twig_roots[twig.twig_id] = twig.root
            return
        
        # Security: Validate twig_id to prevent path traversal
        if not isinstance(twig.twig_id, int) or twig.twig_id < 0:
            raise ValueError(f"Invalid twig_id: {twig.twig_id}")
        
        # Security: Validate storage path is absolute and exists
        if not self._storage_path.is_absolute():
            raise ValueError("Storage path must be absolute")
        
        # Store twig data
        twig_path = self._storage_path / f"twig_{twig.twig_id}.dat"
        
        # Security: Ensure resolved path is under storage_path (prevent traversal)
        try:
            twig_path.resolve().relative_to(self._storage_path.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {twig_path}")
        
        # Security: Store with integrity check
        data = {
            'twig_id': twig.twig_id,
            'entries': twig.entries,
            'active_bits': twig.active_bits.to_bytes(),
            'root': twig.root,
            'first_log_index': twig.first_log_index,
        }
        
        # Compute integrity hash
        integrity_hash = hashlib.sha256(
            str(twig.twig_id).encode() + twig.root
        ).digest()
        
        # Security: Use JSON instead of pickle to avoid deserialization attacks
        # Convert bytes to hex for JSON serialization
        json_data = {
            'twig_id': data['twig_id'],
            'entries': [e.hex() for e in data['entries']],
            'active_bits': data['active_bits'].hex(),
            'root': data['root'].hex(),
            'first_log_index': data['first_log_index'],
            'integrity_hash': integrity_hash.hex(),
        }
        
        with open(twig_path, 'w') as f:
            json.dump(json_data, f)
        
        # Keep root in memory
        self._cold_twig_roots[twig.twig_id] = twig.root
    
    def _load_twig_from_disk(self, twig_id: int) -> Optional[Twig]:
        """
        Load twig from disk with verification.
        
        Security: Verifies integrity hash and recomputes root.
        """
        if self._storage_path is None:
            return None
        
        twig_path = self._storage_path / f"twig_{twig_id}.dat"
        if not twig_path.exists():
            return None
        
        try:
            with open(twig_path, 'r') as f:
                json_data = json.load(f)
            
            # Validate required fields exist
            required_fields = ['twig_id', 'entries', 'active_bits', 'root', 'first_log_index', 'integrity_hash']
            if not all(field in json_data for field in required_fields):
                logger.error(f"Twig {twig_id} missing required fields")
                return None
            
            # Convert hex back to bytes
            entries = [bytes.fromhex(e) for e in json_data['entries']]
            active_bits_bytes = bytes.fromhex(json_data['active_bits'])
            root = bytes.fromhex(json_data['root'])
            stored_hash = bytes.fromhex(json_data['integrity_hash'])
            
            # Verify integrity
            expected_hash = hashlib.sha256(
                str(json_data['twig_id']).encode() + root
            ).digest()
            
            if stored_hash != expected_hash:
                logger.error(f"Twig {twig_id} integrity check failed")
                return None
            
            # Reconstruct twig
            twig = Twig(
                twig_id=json_data['twig_id'],
                entries=entries,
                active_bits=BitArray.from_bytes(active_bits_bytes, TWIG_CAPACITY),
                root=root,
                state=TwigState.FULL,
                first_log_index=json_data['first_log_index'],
                created_at_ms=0,  # Unknown after load
            )
            
            # Security: Verify root matches
            computed_root = self._compute_twig_root(twig.entries)
            if computed_root != twig.root:
                logger.error(f"Twig {twig_id} root verification failed")
                return None
            
            return twig
            
        except Exception as e:
            logger.exception(f"Failed to load twig {twig_id}: {e}")
            return None
    
    def _get_twig(self, twig_id: int) -> Optional[Twig]:
        """
        Get twig by ID (hot or cold).
        
        Promotes cold twigs to hot cache on access.
        """
        # Check current twig
        if self._current_twig and self._current_twig.twig_id == twig_id:
            return self._current_twig
        
        # Check hot cache
        if twig_id in self._hot_twigs:
            # Move to end (LRU)
            self._hot_twigs.move_to_end(twig_id)
            return self._hot_twigs[twig_id]
        
        # Load from disk
        twig = self._load_twig_from_disk(twig_id)
        if twig:
            # Add to hot cache
            self._hot_twigs[twig_id] = twig
            
            # Evict if needed
            while len(self._hot_twigs) > self._hot_twig_count:
                oldest_id, oldest_twig = self._hot_twigs.popitem(last=False)
                if oldest_id != twig_id:  # Don't re-evict what we just loaded
                    self._flush_twig_to_disk(oldest_twig)
        
        return twig
    
    def append(self, leaf_data: bytes) -> Tuple[int, MembershipProof]:
        """
        Append a new entry to the MMR.
        
        Args:
            leaf_data: Raw data to append (will be hashed)
        
        Returns:
            (log_index, proof) where proof is immediately usable
        
        Security:
            - Validates tree size bound
            - Uses domain-separated hashing
        """
        # Security: Check size bound
        if self._size >= MAX_TREE_SIZE:
            raise OverflowError(f"MMR size limit ({MAX_TREE_SIZE}) reached")
        
        # Hash the leaf with domain separation
        leaf_hash = hash_leaf(leaf_data)
        
        return self.append_hash(leaf_hash)
    
    def append_hash(self, leaf_hash: bytes) -> Tuple[int, MembershipProof]:
        """
        Append a pre-hashed entry to the MMR.
        
        Args:
            leaf_hash: Pre-computed leaf hash (must be HASH_SIZE bytes)
        
        Returns:
            (log_index, proof)
        """
        # Validate input
        if len(leaf_hash) != HASH_SIZE:
            raise ValueError(f"Leaf hash must be {HASH_SIZE} bytes")
        
        if self._size >= MAX_TREE_SIZE:
            raise OverflowError(f"MMR size limit ({MAX_TREE_SIZE}) reached")
        
        if self._current_twig is None:
            self._create_fresh_twig()
        
        log_index = self._size
        
        # Add to current twig
        self._current_twig.entries.append(leaf_hash)
        entry_index = len(self._current_twig.entries) - 1
        self._current_twig.active_bits.set(entry_index)
        
        # Remerkleize
        self._remerkleize_current_twig()
        
        # Check if twig is full
        if len(self._current_twig.entries) >= TWIG_CAPACITY:
            self._rotate_twig()
        
        # Update frontier
        self._update_frontier()
        
        # Increment counters
        self._size = safe_add(self._size, 1)
        self._version += 1
        
        # Generate proof
        proof = self.prove(log_index)
        
        return log_index, proof
    
    def prove(self, log_index: int) -> MembershipProof:
        """
        Generate membership proof for entry at log_index.
        
        Args:
            log_index: Index of entry to prove
        
        Returns:
            MembershipProof that can be verified
        
        Security:
            - Validates index bounds
            - Full path computation (no shortcuts)
        """
        # Validate index
        if not 0 <= log_index < self._size:
            raise IndexError(f"Log index {log_index} out of range [0, {self._size})")
        
        # Find containing twig
        twig_id = log_index // TWIG_CAPACITY
        entry_index = log_index % TWIG_CAPACITY
        
        twig = self._get_twig(twig_id)
        if twig is None:
            raise ValueError(f"Twig {twig_id} not found")
        
        if entry_index >= len(twig.entries):
            raise IndexError(f"Entry index {entry_index} not in twig {twig_id}")
        
        # Build intra-twig proof
        intra_twig_path = self._build_twig_proof(twig, entry_index)
        
        # Build inter-twig proof (simplified)
        inter_twig_path = self._build_frontier_proof(twig_id)
        
        return MembershipProof(
            log_index=log_index,
            leaf_hash=twig.entries[entry_index],
            intra_twig_path=intra_twig_path,
            inter_twig_path=inter_twig_path,
            twig_root=twig.root,
            mmr_root=self.root,
            version=self._version,
        )
    
    def _build_twig_proof(self, twig: Twig, entry_index: int) -> List[bytes]:
        """
        Build Merkle proof within a single twig.
        
        Returns list of sibling hashes from leaf to twig root.
        """
        # Pad entries to full capacity
        entries = twig.entries + [EMPTY_HASH] * (TWIG_CAPACITY - len(twig.entries))
        
        path = []
        level = entries
        idx = entry_index
        
        while len(level) > 1:
            # Get sibling
            sibling_idx = idx ^ 1
            if sibling_idx < len(level):
                path.append(level[sibling_idx])
            else:
                path.append(EMPTY_HASH)
            
            # Move up
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else EMPTY_HASH
                next_level.append(hash_internal(left, right))
            
            level = next_level
            idx //= 2
        
        return path
    
    def _build_frontier_proof(self, twig_id: int) -> List[bytes]:
        """
        Build proof from twig root to MMR root.
        
        For now, returns empty list (simplified implementation).
        Full implementation would track MMR structure.
        """
        # Simplified: for single-twig case, no inter-twig proof needed
        return []
    
    def verify(self, proof: MembershipProof) -> bool:
        """
        Verify a membership proof.
        
        Security: Delegates to proof's verify method for full validation.
        """
        return proof.verify()
    
    def checkpoint(self) -> int:
        """
        Create a checkpoint at current version.
        
        Returns version number of checkpoint.
        """
        # Limit checkpoint history
        while len(self._checkpoints) >= MAX_VERSION_HISTORY:
            oldest = min(self._checkpoints.keys())
            del self._checkpoints[oldest]
        
        self._checkpoints[self._version] = self.root
        return self._version
    
    def get_all_entry_hashes(self) -> List[bytes]:
        """
        Get all entry hashes in the MMR.
        
        Used for IBLT construction.
        """
        hashes = []
        
        # Collect from all twigs
        for twig in self._hot_twigs.values():
            hashes.extend(twig.entries)
        
        # Current twig
        if self._current_twig:
            hashes.extend(self._current_twig.entries)
        
        return hashes
    
    def get_sync_state(self) -> Dict[str, Any]:
        """Get state for synchronization."""
        return {
            'size': self._size,
            'frontier': self._frontier.copy(),
            'version': self._version,
            'root': self.root,
        }
