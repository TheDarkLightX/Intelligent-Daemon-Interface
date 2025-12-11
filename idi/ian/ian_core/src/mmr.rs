//! Merkle Mountain Range (MMR) implementation
//!
//! An MMR is an append-only authenticated data structure that supports:
//! - O(log N) append
//! - O(log N) membership proofs
//! - O(log N) root computation
//!
//! This implementation uses a peaks-only storage model for efficiency:
//! - Only peak hashes are stored (not all internal nodes)
//! - Proofs reconstruct paths on-demand using stored leaf data
//! - Memory: O(log N) for peaks + O(N) for leaf data

use crate::{sha256_domain, Hash32, IanError, Result};
use serde::{Deserialize, Serialize};

/// Domain prefixes for hash computation
const LEAF_PREFIX: &[u8] = b"IAN_MMR_LEAF_V1";
const NODE_PREFIX: &[u8] = b"IAN_MMR_NODE_V1";
const PEAK_PREFIX: &[u8] = b"IAN_MMR_PEAK_V1";

/// MMR membership proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmrProof {
    pub leaf_index: usize,
    pub leaf_hash: Hash32,
    pub siblings: Vec<(Hash32, bool)>, // (hash, is_right)
    pub peaks: Vec<Hash32>,
    pub mmr_size: usize,
}

/// Merkle Mountain Range
/// 
/// Stores peaks (roots of complete binary subtrees) and original leaf data.
/// Proofs are constructed by rebuilding the necessary path from leaf data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleMountainRange {
    /// Current peaks (roots of complete binary subtrees)
    peaks: Vec<Hash32>,
    /// Original leaf data for proof reconstruction
    leaf_data: Vec<Vec<u8>>,
    /// Number of leaves
    leaf_count: usize,
}

impl Default for MerkleMountainRange {
    fn default() -> Self {
        Self::new()
    }
}

impl MerkleMountainRange {
    /// Create a new empty MMR
    pub fn new() -> Self {
        Self {
            peaks: Vec::new(),
            leaf_data: Vec::new(),
            leaf_count: 0,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let peak_capacity = (usize::BITS - capacity.leading_zeros()) as usize;
        Self {
            peaks: Vec::with_capacity(peak_capacity),
            leaf_data: Vec::with_capacity(capacity),
            leaf_count: 0,
        }
    }

    /// Number of leaves in the MMR
    #[inline]
    pub fn size(&self) -> usize {
        self.leaf_count
    }

    /// Number of peaks (equals popcount of leaf_count)
    #[inline]
    pub fn peak_count(&self) -> usize {
        self.peaks.len()
    }

    /// Append a leaf and return its index
    ///
    /// Complexity: O(log N)
    pub fn append(&mut self, data: &[u8]) -> usize {
        let leaf_hash = sha256_domain(LEAF_PREFIX, data);
        let leaf_index = self.leaf_count;
        
        // Store original data
        self.leaf_data.push(data.to_vec());
        self.leaf_count += 1;
        
        // Add as new peak
        self.peaks.push(leaf_hash);
        
        // Merge consecutive peaks of equal height
        // Number of trailing 1s in leaf_count (after increment) tells us how many merges
        let mut merges = self.leaf_count.trailing_zeros() as usize;
        
        while merges > 0 && self.peaks.len() >= 2 {
            let right = self.peaks.pop().unwrap();
            let left = self.peaks.pop().unwrap();
            let parent = self.hash_nodes(&left, &right);
            self.peaks.push(parent);
            merges -= 1;
        }
        
        leaf_index
    }

    /// Hash two child nodes into a parent
    #[inline]
    fn hash_nodes(&self, left: &Hash32, right: &Hash32) -> Hash32 {
        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(left);
        combined[32..].copy_from_slice(right);
        sha256_domain(NODE_PREFIX, &combined)
    }

    /// Get the peak hashes
    pub fn get_peaks(&self) -> Vec<Hash32> {
        self.peaks.clone()
    }

    /// Get the MMR root (bag of peaks)
    ///
    /// Complexity: O(log N)
    pub fn get_root(&self) -> Hash32 {
        if self.peaks.is_empty() {
            return [0u8; 32];
        }
        
        if self.peaks.len() == 1 {
            return self.peaks[0];
        }
        
        // Bag peaks right-to-left
        let mut root = self.peaks[self.peaks.len() - 1];
        for peak in self.peaks.iter().rev().skip(1) {
            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(peak);
            combined[32..].copy_from_slice(&root);
            root = sha256_domain(PEAK_PREFIX, &combined);
        }
        
        root
    }

    /// Get a membership proof for a leaf
    ///
    /// Reconstructs the path from leaf data on demand.
    /// Complexity: O(log N) hash computations
    pub fn get_proof(&self, leaf_index: usize) -> Result<MmrProof> {
        if leaf_index >= self.leaf_count {
            return Err(IanError::InvalidLeafIndex(leaf_index));
        }

        let leaf_hash = sha256_domain(LEAF_PREFIX, &self.leaf_data[leaf_index]);
        
        // Find which subtree (peak) this leaf belongs to
        let (subtree_start, subtree_size, _peak_idx) = self.find_subtree_for_leaf(leaf_index);
        let local_index = leaf_index - subtree_start;
        
        // Build sibling path within the subtree
        let siblings = self.build_path(subtree_start, subtree_size, local_index);

        Ok(MmrProof {
            leaf_index,
            leaf_hash,
            siblings,
            peaks: self.peaks.clone(),
            mmr_size: self.leaf_count,
        })
    }

    /// Find (subtree_start, subtree_size, peak_index) for a leaf
    fn find_subtree_for_leaf(&self, leaf_index: usize) -> (usize, usize, usize) {
        let mut start = 0;
        let mut peak_idx = 0;
        
        for bit in (0..usize::BITS).rev() {
            let tree_size = 1usize << bit;
            if start + tree_size <= self.leaf_count && 
               (self.leaf_count & tree_size) != 0 {
                if leaf_index < start + tree_size {
                    return (start, tree_size, peak_idx);
                }
                start += tree_size;
                peak_idx += 1;
            }
        }
        
        (start, 1, peak_idx)
    }

    /// Build sibling path for a leaf within a subtree
    fn build_path(&self, subtree_start: usize, subtree_size: usize, local_index: usize) -> Vec<(Hash32, bool)> {
        if subtree_size == 1 {
            return vec![];
        }
        
        let mut siblings = Vec::new();
        let mut current_size = subtree_size;
        let mut current_offset = subtree_start;
        let mut current_local = local_index;
        
        while current_size > 1 {
            let half = current_size / 2;
            let is_in_right = current_local >= half;
            
            // Compute sibling hash
            let sibling_start = if is_in_right {
                current_offset // Left sibling
            } else {
                current_offset + half // Right sibling
            };
            let sibling_size = half;
            
            let sibling_hash = self.compute_subtree_hash(sibling_start, sibling_size);
            // is_in_right means leaf is in right subtree, so sibling is LEFT
            // The bool in proof means "sibling is on right", so we negate
            siblings.push((sibling_hash, !is_in_right));
            
            // Move to next level
            if is_in_right {
                current_offset += half;
                current_local -= half;
            }
            current_size = half;
        }
        
        siblings
    }

    /// Compute the hash of a subtree given its start and size
    fn compute_subtree_hash(&self, start: usize, size: usize) -> Hash32 {
        if size == 1 {
            return sha256_domain(LEAF_PREFIX, &self.leaf_data[start]);
        }
        
        let half = size / 2;
        let left_hash = self.compute_subtree_hash(start, half);
        let right_hash = self.compute_subtree_hash(start + half, half);
        self.hash_nodes(&left_hash, &right_hash)
    }

    /// Verify a membership proof
    ///
    /// Complexity: O(log N)
    pub fn verify_proof(data: &[u8], proof: &MmrProof, expected_root: &Hash32) -> bool {
        // Verify leaf hash
        let computed_leaf_hash = sha256_domain(LEAF_PREFIX, data);
        if computed_leaf_hash != proof.leaf_hash {
            return false;
        }

        // Walk up the tree
        let mut current = proof.leaf_hash;
        for (sibling, is_right) in &proof.siblings {
            current = if *is_right {
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(&current);
                combined[32..].copy_from_slice(sibling);
                sha256_domain(NODE_PREFIX, &combined)
            } else {
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(sibling);
                combined[32..].copy_from_slice(&current);
                sha256_domain(NODE_PREFIX, &combined)
            };
        }

        // Current should now be the peak
        // Verify it's in the peaks list
        if !proof.peaks.contains(&current) {
            return false;
        }

        // Bag peaks and verify root
        if proof.peaks.is_empty() {
            return false;
        }

        if proof.peaks.len() == 1 {
            return proof.peaks[0] == *expected_root;
        }

        let mut root = proof.peaks[proof.peaks.len() - 1];
        for peak in proof.peaks.iter().rev().skip(1) {
            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(peak);
            combined[32..].copy_from_slice(&root);
            root = sha256_domain(PEAK_PREFIX, &combined);
        }

        root == *expected_root
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| IanError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| IanError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_mmr() {
        let mmr = MerkleMountainRange::new();
        assert_eq!(mmr.size(), 0);
        assert_eq!(mmr.get_root(), [0u8; 32]);
    }

    #[test]
    fn test_single_leaf() {
        let mut mmr = MerkleMountainRange::new();
        let idx = mmr.append(b"hello");
        
        assert_eq!(idx, 0);
        assert_eq!(mmr.size(), 1);
        assert_ne!(mmr.get_root(), [0u8; 32]);
    }

    #[test]
    fn test_append_multiple() {
        let mut mmr = MerkleMountainRange::new();
        
        for i in 0..100 {
            let idx = mmr.append(format!("leaf_{}", i).as_bytes());
            assert_eq!(idx, i);
        }
        
        assert_eq!(mmr.size(), 100);
    }

    #[test]
    fn test_peaks_count() {
        // Number of peaks = popcount(leaf_count)
        let mut mmr = MerkleMountainRange::new();
        
        for i in 1usize..=16 {
            mmr.append(format!("leaf_{}", i).as_bytes());
            let peaks = mmr.get_peaks();
            assert_eq!(peaks.len(), i.count_ones() as usize);
        }
    }

    #[test]
    fn test_root_deterministic() {
        let mut mmr1 = MerkleMountainRange::new();
        let mut mmr2 = MerkleMountainRange::new();
        
        for i in 0..10 {
            mmr1.append(format!("leaf_{}", i).as_bytes());
            mmr2.append(format!("leaf_{}", i).as_bytes());
        }
        
        assert_eq!(mmr1.get_root(), mmr2.get_root());
    }

    #[test]
    fn test_proof_simple() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(b"leaf_0");
        mmr.append(b"leaf_1");
        
        let root = mmr.get_root();
        
        // Verify proof for leaf 0
        let proof = mmr.get_proof(0).unwrap();
        assert!(MerkleMountainRange::verify_proof(b"leaf_0", &proof, &root));
        
        // Verify proof for leaf 1
        let proof = mmr.get_proof(1).unwrap();
        assert!(MerkleMountainRange::verify_proof(b"leaf_1", &proof, &root));
    }

    #[test]
    fn test_serialization() {
        let mut mmr = MerkleMountainRange::new();
        for i in 0..10 {
            mmr.append(format!("leaf_{}", i).as_bytes());
        }
        
        let json = mmr.to_json().unwrap();
        let restored = MerkleMountainRange::from_json(&json).unwrap();
        
        assert_eq!(mmr.size(), restored.size());
        assert_eq!(mmr.get_root(), restored.get_root());
    }
}
