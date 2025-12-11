//! IAN Core - High-performance Rust implementation
//!
//! This crate provides optimized implementations of:
//! - Merkle Mountain Range (MMR) for append-only logs
//! - Bloom Filter for probabilistic de-duplication
//! - Leaderboard for top-K tracking
//!
//! Performance targets:
//! - MMR append: O(log N), < 1μs for N < 1M
//! - Bloom check: O(k), < 100ns
//! - Leaderboard insert: O(log K), < 500ns

pub mod mmr;
pub mod bloom;
pub mod leaderboard;
pub mod dedup;

#[cfg(feature = "python")]
mod python;

pub use mmr::{MerkleMountainRange, MmrProof};
pub use bloom::BloomFilter;
pub use leaderboard::Leaderboard;
pub use dedup::DedupService;

/// Error types for IAN core operations
#[derive(Debug, thiserror::Error)]
pub enum IanError {
    #[error("Invalid leaf index: {0}")]
    InvalidLeafIndex(usize),
    
    #[error("Proof verification failed")]
    ProofVerificationFailed,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Capacity exceeded: {current} > {max}")]
    CapacityExceeded { current: usize, max: usize },
}

pub type Result<T> = std::result::Result<T, IanError>;

/// 32-byte hash type
pub type Hash32 = [u8; 32];

/// Hash helper using SHA-256
pub fn sha256(data: &[u8]) -> Hash32 {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Hash with domain separation
pub fn sha256_domain(domain: &[u8], data: &[u8]) -> Hash32 {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let hash = sha256(b"hello");
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_domain() {
        let hash1 = sha256_domain(b"leaf", b"data");
        let hash2 = sha256_domain(b"node", b"data");
        assert_ne!(hash1, hash2);
    }
}
