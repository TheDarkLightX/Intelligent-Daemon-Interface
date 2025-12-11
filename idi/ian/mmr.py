"""
Merkle Mountain Range (MMR) implementation for IAN experiment logs.

MMR is an append-only authenticated data structure that allows:
- O(log N) append without rebalancing
- O(log N) membership proofs
- O(log N) peaks for compact state root

This implementation follows the Grin/Mimblewimble MMR design with some
simplifications for clarity.

Security Properties:
- Collision resistance: Relies on SHA-256 preimage resistance
- Binding: Root uniquely identifies the full sequence of leaves
- Append-only: No mutation of historical entries

Complexity:
- Append: O(log N) hash operations
- Get root: O(log N) hash operations (bagging peaks)
- Membership proof: O(log N) hashes in proof
- Verify proof: O(log N) hash operations
- Storage: O(N) leaves + O(log N) peaks (or O(N) for all nodes)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _hash_leaf(data: bytes) -> bytes:
    """Hash a leaf node. Prefix with 0x00 to domain-separate from internal nodes."""
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_internal(left: bytes, right: bytes) -> bytes:
    """Hash two child nodes to create parent. Prefix with 0x01 for domain separation."""
    return hashlib.sha256(b"\x01" + left + right).digest()


def _count_trailing_ones(n: int) -> int:
    """Count trailing 1-bits in binary representation of n."""
    if n == 0:
        return 0
    count = 0
    while n & 1:
        count += 1
        n >>= 1
    return count


def _popcount(n: int) -> int:
    """Count number of 1-bits in binary representation."""
    return bin(n).count('1')


@dataclass
class MMRProof:
    """
    Membership proof for a leaf in the MMR.
    
    Contains the authentication path from leaf to peaks.
    """
    leaf_index: int
    leaf_hash: bytes
    siblings: List[Tuple[bytes, bool]]  # (sibling_hash, is_right_sibling)
    peaks_bag: List[bytes]  # Peaks to bag with proved subtree root
    mmr_size: int  # Size of MMR when proof was generated
    
    def to_dict(self) -> dict:
        return {
            "leaf_index": self.leaf_index,
            "leaf_hash": self.leaf_hash.hex(),
            "siblings": [(h.hex(), r) for h, r in self.siblings],
            "peaks_bag": [p.hex() for p in self.peaks_bag],
            "mmr_size": self.mmr_size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MMRProof":
        return cls(
            leaf_index=data["leaf_index"],
            leaf_hash=bytes.fromhex(data["leaf_hash"]),
            siblings=[(bytes.fromhex(h), r) for h, r in data["siblings"]],
            peaks_bag=[bytes.fromhex(p) for p in data["peaks_bag"]],
            mmr_size=data["mmr_size"],
        )


class MerkleMountainRange:
    """
    Merkle Mountain Range: append-only authenticated log.
    
    The MMR consists of multiple "mountains" (perfect binary trees) of
    decreasing heights. When a new leaf is added, it may merge with
    existing mountains to form larger ones.
    
    Example state after 7 leaves::
    
             [6]           <- height 2 mountain (4 leaves)
            /   \\
          [2]   [5]
          / \\   / \\
        [0] [1][3][4]
        
        Plus:
             [9]           <- height 1 mountain (2 leaves)
            /   \\
          [7]   [8]
        
        Plus:
          [10]             <- height 0 mountain (1 leaf)
        
        Peaks: [6], [9], [10]
    
    Invariants:
    - Peaks are ordered by decreasing height
    - After appending leaf i, the peaks represent a unique commitment to leaves [0..i]
    - The bagged root is computed by hashing peaks left-to-right
    """
    
    def __init__(self) -> None:
        # Store all nodes (leaves and internal) in a flat array
        # Position in array follows MMR indexing
        self._nodes: List[bytes] = []
        self._leaf_count: int = 0
        # Store leaf data for proof generation (optional, can be offloaded)
        self._leaf_data: List[bytes] = []
    
    @property
    def size(self) -> int:
        """Number of leaves in the MMR."""
        return self._leaf_count
    
    @property
    def node_count(self) -> int:
        """Total number of nodes (leaves + internal)."""
        return len(self._nodes)
    
    def append(self, leaf_data: bytes) -> int:
        """
        Append a leaf to the MMR.
        
        Args:
            leaf_data: Raw data for the leaf
            
        Returns:
            Index of the new leaf (0-indexed)
            
        Complexity: O(log N) hash operations
        """
        leaf_hash = _hash_leaf(leaf_data)
        self._leaf_data.append(leaf_data)
        self._nodes.append(leaf_hash)
        
        current_hash = leaf_hash
        current_height = 0
        
        # Merge with existing peaks if they have the same height
        # This is determined by counting trailing 1-bits in the leaf count
        merge_count = _count_trailing_ones(self._leaf_count)
        
        for _ in range(merge_count):
            # Get left sibling (previous peak at same height)
            left_sibling = self._nodes[len(self._nodes) - 2]
            # Create parent
            parent_hash = _hash_internal(left_sibling, current_hash)
            self._nodes.append(parent_hash)
            current_hash = parent_hash
            current_height += 1
        
        leaf_index = self._leaf_count
        self._leaf_count += 1
        return leaf_index
    
    def get_peaks(self) -> List[bytes]:
        """
        Get the current peaks (roots of the mountains).
        
        Returns:
            List of peak hashes, ordered by decreasing height
            
        Complexity: O(log N)
        """
        if not self._nodes:
            return []
        
        peaks = []
        peak_positions = self._get_peak_positions()
        for pos in peak_positions:
            if pos < len(self._nodes):
                peaks.append(self._nodes[pos])
        
        return peaks
    
    def _get_peak_positions(self) -> List[int]:
        """
        Get positions of all peaks in the MMR.
        
        Based on decomposing leaf_count into powers of 2.
        """
        if self._leaf_count == 0:
            return []
        
        positions = []
        pos = 0
        remaining = self._leaf_count
        
        while remaining > 0:
            # Find the highest power of 2 that fits
            # This corresponds to the size of the next tree
            tree_size = 1 << (remaining.bit_length() - 1)
            tree_height = remaining.bit_length() - 1
            
            # Peak is at pos + 2^(height+1) - 2 for a complete binary tree
            peak_pos = pos + (1 << (tree_height + 1)) - 2
            positions.append(peak_pos)
            
            # Move past this tree: tree has (2*tree_size - 1) nodes
            pos += (1 << (tree_height + 1)) - 1
            remaining -= tree_size
        
        return positions
    
    def get_root(self) -> bytes:
        """
        Compute the MMR root by bagging the peaks.
        
        Returns:
            32-byte root hash
            
        Complexity: O(log N)
        """
        peaks = self.get_peaks()
        
        if not peaks:
            return b"\x00" * 32
        
        if len(peaks) == 1:
            return peaks[0]
        
        # Bag peaks right-to-left
        result = peaks[-1]
        for peak in reversed(peaks[:-1]):
            result = _hash_internal(peak, result)
        
        return result
    
    def get_proof(self, leaf_index: int) -> MMRProof:
        """
        Generate membership proof for a leaf.
        
        The proof contains:
        1. Authentication path from leaf to its peak (siblings at each level)
        2. Other peaks for bagging to verify the full root
        
        Args:
            leaf_index: Index of the leaf (0-indexed)
            
        Returns:
            MMRProof containing authentication path
            
        Raises:
            IndexError: If leaf_index is out of bounds
            
        Complexity: O(log N)
        """
        if leaf_index < 0 or leaf_index >= self._leaf_count:
            raise IndexError(f"Leaf index {leaf_index} out of bounds [0, {self._leaf_count})")
        
        # Get leaf hash
        leaf_hash = _hash_leaf(self._leaf_data[leaf_index])
        
        # Find which peak this leaf belongs to
        peak_pos, peak_height = self._find_peak_containing(leaf_index)
        
        # Build authentication path from leaf to peak
        siblings: List[Tuple[bytes, bool]] = []
        
        # Calculate the local index within the subtree
        # First, find the starting leaf index for this peak's tree
        local_leaf_idx = leaf_index
        pos = 0
        remaining = self._leaf_count
        
        while remaining > 0:
            tree_size = 1 << (remaining.bit_length() - 1)
            tree_height = remaining.bit_length() - 1
            current_peak_pos = pos + (1 << (tree_height + 1)) - 2
            
            if current_peak_pos == peak_pos:
                break
            
            local_leaf_idx -= tree_size
            pos += (1 << (tree_height + 1)) - 1
            remaining -= tree_size
        
        # Now traverse from leaf to peak within this subtree
        # Position within the subtree
        tree_base_pos = pos
        current_local_idx = local_leaf_idx
        
        for h in range(peak_height):
            # At each height, determine if we're left or right child
            is_right_child = (current_local_idx >> h) & 1
            
            # Calculate sibling position
            sibling_local_idx = current_local_idx ^ (1 << h)
            
            # Convert local index to node position
            # Leaf positions: tree_base + 2*leaf_idx (within subtree)
            # For height h, offset is more complex
            sibling_node_pos = self._local_to_node_position(tree_base_pos, sibling_local_idx, h)
            
            if sibling_node_pos < len(self._nodes):
                siblings.append((self._nodes[sibling_node_pos], not is_right_child))
            
            # Move to parent level
            current_local_idx >>= 1
        
        # Get peaks for bagging (excluding the one containing this leaf)
        all_peaks = self.get_peaks()
        proved_peak_hash = self._nodes[peak_pos] if peak_pos < len(self._nodes) else leaf_hash
        
        # Find index of this peak and collect others for bagging
        peak_positions = self._get_peak_positions()
        peak_idx = peak_positions.index(peak_pos)
        peaks_bag = [all_peaks[i] for i in range(len(all_peaks)) if i != peak_idx]
        
        return MMRProof(
            leaf_index=leaf_index,
            leaf_hash=leaf_hash,
            siblings=siblings,
            peaks_bag=peaks_bag,
            mmr_size=self._leaf_count,
        )
    
    def _local_to_node_position(self, tree_base: int, local_idx: int, height: int) -> int:
        """
        Convert local index within a subtree to global node position.
        
        Args:
            tree_base: Starting position of the subtree in _nodes
            local_idx: Local index at the given height
            height: Current height (0 = leaf level)
            
        Returns:
            Global position in _nodes array
        """
        # At height h, each node represents 2^h leaves
        # Position within height h: tree_base + offset for heights below + local_idx
        offset = 0
        for h in range(height):
            # At height h, there are 2^(tree_height - h) nodes
            offset += (1 << (height - h))
        return tree_base + 2 * local_idx + offset
    
    def _leaf_to_node_position(self, leaf_index: int) -> int:
        """
        Convert leaf index to node position in the flat array.
        
        Uses O(1) formula: pos = 2*leaf_index - popcount(leaf_index)
        
        Args:
            leaf_index: 0-based index of the leaf
            
        Returns:
            Position of the leaf in the flat node array
        """
        # OPTIMIZED: O(1) via bit manipulation instead of O(N) loop
        return 2 * leaf_index - _popcount(leaf_index)
    
    def _get_height_at_position(self, pos: int) -> int:
        """
        Get the height of a node at a given position.
        
        Height 0 = leaf, height 1 = parent of two leaves, etc.
        """
        # Height is determined by trailing 1-bits of (pos+1)
        return _count_trailing_ones(pos + 1)
    
    def _get_sibling_offset(self, height: int) -> int:
        """Get the offset to the sibling at a given height."""
        # At height h, sibling is 2^(h+1) - 1 positions away
        return (1 << (height + 1)) - 1
    
    def _find_peak_containing(self, leaf_index: int) -> Tuple[int, int]:
        """
        Find which peak contains a given leaf.
        
        Returns:
            (peak_position, peak_height)
        """
        # Walk through peaks to find which one contains this leaf
        pos = 0
        leaf_count = 0
        remaining = self._leaf_count
        
        while remaining > 0:
            tree_size = 1 << (remaining.bit_length() - 1)
            tree_height = remaining.bit_length() - 1
            
            if leaf_index < leaf_count + tree_size:
                # This leaf is in this tree
                peak_pos = pos + (1 << (tree_height + 1)) - 2
                return peak_pos, tree_height
            
            leaf_count += tree_size
            pos += (1 << (tree_height + 1)) - 1
            remaining -= tree_size
        
        return -1, -1  # Should not reach here
    
    def _is_peak(self, pos: int) -> bool:
        """Check if position is a peak."""
        peak_positions = self._get_peak_positions()
        return pos in peak_positions
    
    @staticmethod
    def verify_proof(
        leaf_data: bytes,
        proof: MMRProof,
        expected_root: bytes,
    ) -> bool:
        """
        Verify a membership proof.
        
        Args:
            leaf_data: Raw data that was supposedly added at leaf_index
            proof: The membership proof
            expected_root: Expected MMR root
            
        Returns:
            True if proof is valid
            
        Complexity: O(log N)
        """
        # Compute leaf hash
        computed_hash = _hash_leaf(leaf_data)
        
        if computed_hash != proof.leaf_hash:
            return False
        
        # Traverse authentication path
        current_hash = computed_hash
        for sibling_hash, is_right_sibling in proof.siblings:
            if is_right_sibling:
                current_hash = _hash_internal(current_hash, sibling_hash)
            else:
                current_hash = _hash_internal(sibling_hash, current_hash)
        
        # The result should be one of the peaks; bag with other peaks
        all_peaks = [current_hash] + proof.peaks_bag
        
        if len(all_peaks) == 1:
            final_root = all_peaks[0]
        else:
            # Bag peaks (this is a simplification; real impl needs proper ordering)
            final_root = all_peaks[-1]
            for peak in reversed(all_peaks[:-1]):
                final_root = _hash_internal(peak, final_root)
        
        return final_root == expected_root
    
    def to_dict(self) -> dict:
        """Serialize MMR state for persistence."""
        return {
            "nodes": [n.hex() for n in self._nodes],
            "leaf_data": [d.hex() for d in self._leaf_data],
            "leaf_count": self._leaf_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MerkleMountainRange":
        """Deserialize MMR from stored state."""
        mmr = cls()
        mmr._nodes = [bytes.fromhex(n) for n in data["nodes"]]
        mmr._leaf_data = [bytes.fromhex(d) for d in data["leaf_data"]]
        mmr._leaf_count = data["leaf_count"]
        return mmr
    
    def __len__(self) -> int:
        return self._leaf_count
    
    def __repr__(self) -> str:
        return f"MerkleMountainRange(leaves={self._leaf_count}, nodes={len(self._nodes)}, root={self.get_root().hex()[:16]}...)"
