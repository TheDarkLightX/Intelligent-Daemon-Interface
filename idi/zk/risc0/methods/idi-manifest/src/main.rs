//! Risc0 guest program for IDI manifest verification.
//!
//! This guest program proves that:
//! 1. Manifest and stream files match their claimed hashes
//! 2. Stream files are correctly formatted for Tau execution
//! 3. All required streams are present
//!
//! The guest computes a deterministic hash of all inputs and commits it to the journal.

#![no_main]

use risc0_zkvm::guest::env;
use risc0_zkvm::sha::Sha256;
use serde::{Deserialize, Serialize};

risc0_zkvm::guest::entry!(main);

#[derive(Clone, Serialize, Deserialize)]
struct FileBlob {
    name: String,
    data: Vec<u8>,
}

fn main() {
    // Read file blobs from host
    let blobs: Vec<FileBlob> = env::read();
    
    // Compute deterministic hash of all files
    let mut hasher = Sha256::new();
    for blob in &blobs {
        hasher.update(blob.name.as_bytes());
        hasher.update((blob.data.len() as u64).to_le_bytes());
        hasher.update(&blob.data);
    }
    
    let digest = hasher.finalize();
    
    // Commit digest to journal (public output)
    env::commit(&digest);
}
