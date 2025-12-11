//! Leaderboard implementation
//!
//! A bounded top-K leaderboard using a min-heap with:
//! - O(log K) insertion
//! - O(K log K) sorted retrieval
//! - O(1) worst score query
//!
//! Optimized for high-frequency updates with stable ordering.

use crate::{sha256, Hash32, IanError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Entry in the leaderboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub pack_hash: Hash32,
    pub score: f64,
    pub timestamp_ms: u64,
    pub log_index: usize,
    pub contributor_id: String,
}

impl LeaderboardEntry {
    /// Create a new entry
    pub fn new(
        pack_hash: Hash32,
        score: f64,
        timestamp_ms: u64,
        log_index: usize,
        contributor_id: String,
    ) -> Self {
        Self {
            pack_hash,
            score,
            timestamp_ms,
            log_index,
            contributor_id,
        }
    }
}

/// Wrapper for min-heap ordering (smallest score at top)
#[derive(Debug, Clone)]
struct MinHeapEntry(LeaderboardEntry);

impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.pack_hash == other.0.pack_hash
    }
}

impl Eq for MinHeapEntry {}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        // Compare by score (smaller is "greater" for eviction)
        // Tie-break by timestamp (newer is "greater" for eviction)
        match other.0.score.partial_cmp(&self.0.score) {
            Some(Ordering::Equal) | None => {
                // Tie-break: earlier timestamp wins (stays longer)
                self.0.timestamp_ms.cmp(&other.0.timestamp_ms)
            }
            Some(ord) => ord,
        }
    }
}

/// Bounded top-K leaderboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    /// Capacity (K)
    capacity: usize,
    /// Entries stored as sorted vector (for serialization)
    /// In-memory, we use a heap for operations
    #[serde(skip)]
    heap: BinaryHeap<MinHeapEntry>,
    /// Serializable entries
    entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    /// Create a new leaderboard with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity + 1),
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Add an entry, returns true if entry was added to leaderboard
    ///
    /// Complexity: O(log K)
    pub fn add(&mut self, entry: LeaderboardEntry) -> bool {
        // Check if already at capacity and score is too low
        if self.heap.len() >= self.capacity {
            if let Some(worst) = self.heap.peek() {
                if entry.score < worst.0.score {
                    return false;
                }
                if entry.score == worst.0.score && entry.timestamp_ms >= worst.0.timestamp_ms {
                    return false;
                }
            }
        }

        // Add the new entry
        self.heap.push(MinHeapEntry(entry));

        // Evict if over capacity
        if self.heap.len() > self.capacity {
            self.heap.pop();
        }

        true
    }

    /// Get the worst (lowest) score currently on leaderboard
    ///
    /// Complexity: O(1)
    pub fn worst_score(&self) -> Option<f64> {
        self.heap.peek().map(|e| e.0.score)
    }

    /// Get all entries sorted by score (descending)
    ///
    /// Complexity: O(K log K)
    pub fn top_k(&self) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self.heap.iter().map(|e| e.0.clone()).collect();
        entries.sort_by(|a, b| {
            match b.score.partial_cmp(&a.score) {
                Some(Ordering::Equal) | None => a.timestamp_ms.cmp(&b.timestamp_ms),
                Some(ord) => ord,
            }
        });
        entries
    }

    /// Get the top entry (highest score)
    pub fn top(&self) -> Option<LeaderboardEntry> {
        self.top_k().into_iter().next()
    }

    /// Check if a pack hash is on the leaderboard
    pub fn contains(&self, pack_hash: &Hash32) -> bool {
        self.heap.iter().any(|e| &e.0.pack_hash == pack_hash)
    }

    /// Compute a deterministic hash of the leaderboard state
    pub fn root_hash(&self) -> Hash32 {
        let entries = self.top_k();
        if entries.is_empty() {
            return [0u8; 32];
        }

        let mut hasher_data = Vec::new();
        for entry in entries {
            hasher_data.extend_from_slice(&entry.pack_hash);
            hasher_data.extend_from_slice(&entry.score.to_le_bytes());
            hasher_data.extend_from_slice(&entry.timestamp_ms.to_le_bytes());
        }
        sha256(&hasher_data)
    }

    /// Prepare for serialization
    pub fn prepare_serialize(&mut self) {
        self.entries = self.top_k();
    }

    /// Restore from serialization
    pub fn restore_from_entries(&mut self) {
        self.heap.clear();
        for entry in self.entries.drain(..) {
            self.heap.push(MinHeapEntry(entry));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(score: f64, ts: u64) -> LeaderboardEntry {
        LeaderboardEntry::new(
            [0u8; 32],
            score,
            ts,
            0,
            "test".to_string(),
        )
    }

    #[test]
    fn test_leaderboard_add() {
        let mut lb = Leaderboard::new(3);
        
        assert!(lb.add(make_entry(0.5, 1000)));
        assert!(lb.add(make_entry(0.7, 2000)));
        assert!(lb.add(make_entry(0.3, 3000)));
        
        assert_eq!(lb.len(), 3);
    }

    #[test]
    fn test_leaderboard_eviction() {
        let mut lb = Leaderboard::new(2);
        
        lb.add(make_entry(0.5, 1000));
        lb.add(make_entry(0.7, 2000));
        
        // This should evict 0.5
        lb.add(make_entry(0.6, 3000));
        
        assert_eq!(lb.len(), 2);
        assert_eq!(lb.worst_score(), Some(0.6));
    }

    #[test]
    fn test_leaderboard_top_k_sorted() {
        let mut lb = Leaderboard::new(5);
        
        lb.add(make_entry(0.3, 1000));
        lb.add(make_entry(0.9, 2000));
        lb.add(make_entry(0.5, 3000));
        lb.add(make_entry(0.7, 4000));
        
        let top = lb.top_k();
        assert_eq!(top.len(), 4);
        
        // Should be descending
        assert!(top[0].score >= top[1].score);
        assert!(top[1].score >= top[2].score);
        assert!(top[2].score >= top[3].score);
    }

    #[test]
    fn test_leaderboard_tie_breaking() {
        let mut lb = Leaderboard::new(2);
        
        lb.add(make_entry(0.5, 1000)); // Earlier
        lb.add(make_entry(0.5, 2000)); // Later
        lb.add(make_entry(0.5, 3000)); // Even later
        
        // Should keep the two earlier ones
        let top = lb.top_k();
        assert_eq!(top.len(), 2);
        assert!(top[0].timestamp_ms <= top[1].timestamp_ms);
    }

    #[test]
    fn test_leaderboard_reject_low_score() {
        let mut lb = Leaderboard::new(2);
        
        lb.add(make_entry(0.8, 1000));
        lb.add(make_entry(0.7, 2000));
        
        // This should be rejected (score too low)
        assert!(!lb.add(make_entry(0.5, 3000)));
        assert_eq!(lb.len(), 2);
    }
}
