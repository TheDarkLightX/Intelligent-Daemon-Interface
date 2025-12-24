//! Bloom Filter implementation
//!
//! A probabilistic data structure for set membership with:
//! - O(k) insertion and lookup where k = number of hash functions
//! - No false negatives (if check returns false, item is definitely not in set)
//! - Configurable false positive rate
//!
//! Optimized for:
//! - Cache-friendly bit access
//! - SIMD-friendly hash computation (using multiple SHA-256 derivations)

use crate::sha256;
use serde::{Deserialize, Serialize};

/// Bloom filter for probabilistic set membership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array stored as bytes
    bits: Vec<u8>,
    /// Number of bits (m)
    num_bits: usize,
    /// Number of hash functions (k)
    num_hashes: usize,
    /// Number of items inserted
    count: usize,
    /// Target false positive rate
    fp_rate: f64,
}

impl BloomFilter {
    /// Create a new Bloom filter with target capacity and false positive rate
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to insert
    /// * `fp_rate` - Target false positive rate (0 < fp_rate < 1)
    ///
    /// # Panics
    /// Panics if expected_items is 0 or fp_rate is not in (0, 1)
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be positive");
        assert!(fp_rate > 0.0 && fp_rate < 1.0, "fp_rate must be in (0, 1)");

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let ln2 = std::f64::consts::LN_2;
        let num_bits = (-(expected_items as f64) * fp_rate.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bits = num_bits.max(64); // Minimum 64 bits

        // Optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes = ((num_bits as f64 / expected_items as f64) * ln2).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 16); // Clamp to [1, 16]

        let num_bytes = (num_bits + 7) / 8;

        Self {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes,
            count: 0,
            fp_rate,
        }
    }

    /// Create with specific parameters (for advanced use)
    pub fn with_params(num_bits: usize, num_hashes: usize) -> Self {
        let num_bytes = (num_bits + 7) / 8;
        Self {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes,
            count: 0,
            fp_rate: 0.01,
        }
    }

    /// Add an item to the filter
    ///
    /// Complexity: O(k) where k = number of hash functions
    #[inline]
    pub fn add(&mut self, item: &[u8]) {
        let positions = self.hash_positions(item);
        for pos in positions {
            self.set_bit(pos);
        }
        self.count += 1;
    }

    /// Check if an item might be in the filter
    ///
    /// Returns:
    /// - false: Item is definitely NOT in the set
    /// - true: Item is PROBABLY in the set (may be false positive)
    ///
    /// Complexity: O(k)
    #[inline]
    pub fn maybe_contains(&self, item: &[u8]) -> bool {
        let positions = self.hash_positions(item);
        positions.iter().all(|&pos| self.get_bit(pos))
    }

    /// Number of items added
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Estimated current false positive rate
    pub fn estimated_fp_rate(&self) -> f64 {
        // p = (1 - e^(-kn/m))^k
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        let m = self.num_bits as f64;
        
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Number of bits set
    pub fn bits_set(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Fill ratio (bits set / total bits)
    pub fn fill_ratio(&self) -> f64 {
        self.bits_set() as f64 / self.num_bits as f64
    }

    /// Clear all bits
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.count = 0;
    }

    /// Compute hash positions for an item
    ///
    /// Uses enhanced double hashing: h(i) = (h1 + i*h2 + i^2) mod m
    #[inline]
    fn hash_positions(&self, item: &[u8]) -> Vec<usize> {
        // Get two independent hashes from SHA-256
        let hash1 = sha256(item);
        
        // Create second hash with domain separation
        let mut prefixed = Vec::with_capacity(item.len() + 1);
        prefixed.push(0x01);
        prefixed.extend_from_slice(item);
        let hash2 = sha256(&prefixed);

        // Extract h1 and h2 as u64
        let h1 = u64::from_le_bytes(hash1[0..8].try_into().unwrap());
        let h2 = u64::from_le_bytes(hash2[0..8].try_into().unwrap());

        let m = self.num_bits as u64;
        
        (0..self.num_hashes)
            .map(|i| {
                let i = i as u64;
                // Enhanced double hashing with quadratic probing
                let pos = h1.wrapping_add(i.wrapping_mul(h2)).wrapping_add(i.wrapping_mul(i));
                (pos % m) as usize
            })
            .collect()
    }

    /// Set a bit at position
    #[inline]
    fn set_bit(&mut self, pos: usize) {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    /// Get a bit at position
    #[inline]
    fn get_bit(&self, pos: usize) -> bool {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        (self.bits[byte_idx] >> bit_idx) & 1 == 1
    }
}

/// Counting Bloom filter (allows removal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountingBloomFilter {
    /// Counter array (4 bits per counter, packed into bytes)
    counters: Vec<u8>,
    /// Number of counters
    num_counters: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Number of items
    count: usize,
}

impl CountingBloomFilter {
    /// Create a new counting Bloom filter
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let num_counters = (-(expected_items as f64) * fp_rate.ln() / (ln2 * ln2)).ceil() as usize;
        let num_counters = num_counters.max(64);
        
        let num_hashes = ((num_counters as f64 / expected_items as f64) * ln2).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 16);
        
        // 4 bits per counter, 2 counters per byte
        let num_bytes = (num_counters + 1) / 2;
        
        Self {
            counters: vec![0u8; num_bytes],
            num_counters,
            num_hashes,
            count: 0,
        }
    }

    /// Add an item
    pub fn add(&mut self, item: &[u8]) {
        let positions = self.hash_positions(item);
        for pos in positions {
            self.increment(pos);
        }
        self.count += 1;
    }

    /// Remove an item (decrements counters)
    pub fn remove(&mut self, item: &[u8]) -> bool {
        if !self.maybe_contains(item) {
            return false;
        }
        
        let positions = self.hash_positions(item);
        for pos in positions {
            self.decrement(pos);
        }
        self.count = self.count.saturating_sub(1);
        true
    }

    /// Check if item might be present
    pub fn maybe_contains(&self, item: &[u8]) -> bool {
        let positions = self.hash_positions(item);
        positions.iter().all(|&pos| self.get_counter(pos) > 0)
    }

    fn hash_positions(&self, item: &[u8]) -> Vec<usize> {
        let hash1 = sha256(item);
        let mut prefixed = vec![0x01];
        prefixed.extend_from_slice(item);
        let hash2 = sha256(&prefixed);

        let h1 = u64::from_le_bytes(hash1[0..8].try_into().unwrap());
        let h2 = u64::from_le_bytes(hash2[0..8].try_into().unwrap());
        let m = self.num_counters as u64;

        (0..self.num_hashes)
            .map(|i| {
                let i = i as u64;
                let pos = h1.wrapping_add(i.wrapping_mul(h2)).wrapping_add(i.wrapping_mul(i));
                (pos % m) as usize
            })
            .collect()
    }

    fn get_counter(&self, pos: usize) -> u8 {
        let byte_idx = pos / 2;
        let nibble = pos % 2;
        if nibble == 0 {
            self.counters[byte_idx] & 0x0F
        } else {
            (self.counters[byte_idx] >> 4) & 0x0F
        }
    }

    fn increment(&mut self, pos: usize) {
        let byte_idx = pos / 2;
        let nibble = pos % 2;
        let current = self.get_counter(pos);
        if current < 15 {
            if nibble == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current + 1);
            } else {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0x0F) | ((current + 1) << 4);
            }
        }
    }

    fn decrement(&mut self, pos: usize) {
        let byte_idx = pos / 2;
        let nibble = pos % 2;
        let current = self.get_counter(pos);
        if current > 0 {
            if nibble == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current - 1);
            } else {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0x0F) | ((current - 1) << 4);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_add_contains() {
        let mut bf = BloomFilter::new(1000, 0.01);
        
        bf.add(b"hello");
        bf.add(b"world");
        
        assert!(bf.maybe_contains(b"hello"));
        assert!(bf.maybe_contains(b"world"));
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut bf = BloomFilter::new(10000, 0.01);
        
        // Add 10000 items
        for i in 0..10000 {
            bf.add(format!("item_{}", i).as_bytes());
        }
        
        // All added items must be found
        for i in 0..10000 {
            assert!(bf.maybe_contains(format!("item_{}", i).as_bytes()));
        }
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let mut bf = BloomFilter::new(10000, 0.01);
        
        // Add 10000 items
        for i in 0..10000 {
            bf.add(format!("item_{}", i).as_bytes());
        }
        
        // Check items NOT added
        let mut fp_count = 0;
        for i in 10000..20000 {
            if bf.maybe_contains(format!("item_{}", i).as_bytes()) {
                fp_count += 1;
            }
        }
        
        let fp_rate = fp_count as f64 / 10000.0;
        // Should be close to 1%
        assert!(fp_rate < 0.02, "FP rate {} exceeds 2%", fp_rate);
    }

    #[test]
    fn test_counting_bloom() {
        let mut cbf = CountingBloomFilter::new(1000, 0.01);
        
        cbf.add(b"hello");
        assert!(cbf.maybe_contains(b"hello"));
        
        cbf.remove(b"hello");
        assert!(!cbf.maybe_contains(b"hello"));
    }
}
