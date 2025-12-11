//! De-duplication service
//!
//! Two-tier de-duplication combining:
//! 1. Bloom filter for fast probabilistic pre-check
//! 2. Authoritative hash index for definitive lookup
//!
//! This provides O(1) average-case de-dup with bounded false-positive fallback.

use crate::bloom::BloomFilter;
use crate::Hash32;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Authoritative de-duplication index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupIndex {
    /// Map from pack_hash to log_index
    index: HashMap<Hash32, usize>,
}

impl DedupIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: HashMap::with_capacity(capacity),
        }
    }

    /// Add an entry
    #[inline]
    pub fn add(&mut self, pack_hash: Hash32, log_index: usize) {
        self.index.insert(pack_hash, log_index);
    }

    /// Check if a hash exists
    #[inline]
    pub fn contains(&self, pack_hash: &Hash32) -> bool {
        self.index.contains_key(pack_hash)
    }

    /// Get the log index for a hash
    #[inline]
    pub fn get(&self, pack_hash: &Hash32) -> Option<usize> {
        self.index.get(pack_hash).copied()
    }

    /// Number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

impl Default for DedupIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Two-tier de-duplication service
#[derive(Debug, Clone)]
pub struct DedupService {
    /// Fast probabilistic filter
    bloom: BloomFilter,
    /// Authoritative index
    index: DedupIndex,
}

impl DedupService {
    /// Create a new de-dup service
    ///
    /// # Arguments
    /// * `expected_contributions` - Expected number of contributions
    /// * `fp_rate` - Target false positive rate for Bloom filter
    pub fn new(expected_contributions: usize, fp_rate: f64) -> Self {
        Self {
            bloom: BloomFilter::new(expected_contributions, fp_rate),
            index: DedupIndex::with_capacity(expected_contributions),
        }
    }

    /// Check if a pack hash is a duplicate
    ///
    /// Two-tier check:
    /// 1. Bloom filter (fast, may have false positives)
    /// 2. Authoritative index (definitive)
    ///
    /// Complexity: O(1) average
    #[inline]
    pub fn is_duplicate(&self, pack_hash: &Hash32) -> bool {
        // Fast path: if Bloom says no, definitely no
        if !self.bloom.maybe_contains(pack_hash) {
            return false;
        }
        
        // Slow path: authoritative check
        self.index.contains(pack_hash)
    }

    /// Add a new pack hash
    ///
    /// Complexity: O(1)
    pub fn add(&mut self, pack_hash: Hash32, log_index: usize) {
        self.bloom.add(&pack_hash);
        self.index.add(pack_hash, log_index);
    }

    /// Get the log index for a pack hash (if it exists)
    #[inline]
    pub fn get_log_index(&self, pack_hash: &Hash32) -> Option<usize> {
        self.index.get(pack_hash)
    }

    /// Number of registered contributions
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Bloom filter statistics
    pub fn bloom_stats(&self) -> BloomStats {
        BloomStats {
            fill_ratio: self.bloom.fill_ratio(),
            estimated_fp_rate: self.bloom.estimated_fp_rate(),
            bits_set: self.bloom.bits_set(),
        }
    }
}

/// Statistics about the Bloom filter
#[derive(Debug, Clone)]
pub struct BloomStats {
    pub fill_ratio: f64,
    pub estimated_fp_rate: f64,
    pub bits_set: usize,
}

/// Batch de-duplication for bulk operations
pub struct BatchDedupChecker<'a> {
    service: &'a DedupService,
    cache: HashMap<Hash32, bool>,
}

impl<'a> BatchDedupChecker<'a> {
    /// Create a new batch checker
    pub fn new(service: &'a DedupService) -> Self {
        Self {
            service,
            cache: HashMap::new(),
        }
    }

    /// Check if duplicate (with local caching)
    pub fn is_duplicate(&mut self, pack_hash: &Hash32) -> bool {
        if let Some(&cached) = self.cache.get(pack_hash) {
            return cached;
        }

        let is_dup = self.service.is_duplicate(pack_hash);
        self.cache.insert(*pack_hash, is_dup);
        is_dup
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_service_basic() {
        let mut service = DedupService::new(1000, 0.01);
        
        let hash1: Hash32 = [1u8; 32];
        let hash2: Hash32 = [2u8; 32];
        
        assert!(!service.is_duplicate(&hash1));
        
        service.add(hash1, 0);
        
        assert!(service.is_duplicate(&hash1));
        assert!(!service.is_duplicate(&hash2));
    }

    #[test]
    fn test_dedup_no_false_negatives() {
        let mut service = DedupService::new(10000, 0.01);
        
        // Add 10000 items
        for i in 0u32..10000 {
            let mut hash: Hash32 = [0u8; 32];
            hash[..4].copy_from_slice(&i.to_le_bytes());
            service.add(hash, i as usize);
        }
        
        // All added items must be detected as duplicates
        for i in 0u32..10000 {
            let mut hash: Hash32 = [0u8; 32];
            hash[..4].copy_from_slice(&i.to_le_bytes());
            assert!(service.is_duplicate(&hash), "False negative at {}", i);
        }
    }

    #[test]
    fn test_dedup_get_log_index() {
        let mut service = DedupService::new(100, 0.01);
        
        let hash: Hash32 = [42u8; 32];
        service.add(hash, 123);
        
        assert_eq!(service.get_log_index(&hash), Some(123));
    }

    #[test]
    fn test_batch_checker() {
        let mut service = DedupService::new(100, 0.01);
        
        let hash: Hash32 = [1u8; 32];
        service.add(hash, 0);
        
        let mut checker = BatchDedupChecker::new(&service);
        
        // First call goes to service
        assert!(checker.is_duplicate(&hash));
        
        // Second call uses cache
        assert!(checker.is_duplicate(&hash));
    }
}
