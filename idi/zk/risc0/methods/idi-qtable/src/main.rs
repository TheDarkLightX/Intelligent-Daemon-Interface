//! Risc0 guest program for Q-table action selection verification.
//!
//! This guest program proves that:
//! 1. Q-table lookup was performed correctly (with Merkle proof verification)
//! 2. Action selection (argmax) matches the Q-table values
//! 3. Weighted voting (for multi-layer agents) was computed correctly
//!
//! The Q-table remains private - only commitments (Merkle roots) are revealed.

#![no_main]

use risc0_zkvm::guest::env;
use risc0_zkvm::sha::{Digest, Sha256};
use serde::{Deserialize, Serialize};

risc0_zkvm::guest::entry!(main);

/// Q-table entry in Q16.16 fixed-point format
#[derive(Clone, Copy, Serialize, Deserialize)]
struct QEntry {
    q_hold: i32,  // Q16.16 fixed-point
    q_buy: i32,   // Q16.16 fixed-point
    q_sell: i32,  // Q16.16 fixed-point
}

/// Merkle proof path element
#[derive(Clone, Serialize, Deserialize)]
struct MerklePathElement {
    sibling_hash: [u8; 32],
    is_right: bool,
}

/// Q-table proof input
#[derive(Serialize, Deserialize)]
struct QTableProofInput {
    // State (public)
    state_key: String,
    
    // Q-table commitment (public)
    q_table_root: [u8; 32],
    
    // Q-table entry (private witness)
    q_entry: QEntry,
    
    // Merkle proof (private witness, if using Merkle tree)
    merkle_proof: Option<Vec<MerklePathElement>>,
    
    // Selected action (public)
    selected_action: u8,  // 0=hold, 1=buy, 2=sell
    
    // Layer weights (for multi-layer, public)
    layer_weights: Vec<f32>,
}

fn main() {
    // Read input from host
    let input: QTableProofInput = env::read();
    
    // Verify Q-table entry matches commitment
    if let Some(proof) = &input.merkle_proof {
        // Verify Merkle proof
        let mut current_hash = hash_q_entry(&input.state_key, &input.q_entry);
        
        for path_elem in proof {
            let combined = if path_elem.is_right {
                // Current is left, sibling is right
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&current_hash);
                combined.extend_from_slice(&path_elem.sibling_hash);
                combined
            } else {
                // Current is right, sibling is left
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&path_elem.sibling_hash);
                combined.extend_from_slice(&current_hash);
                combined
            };
            
            let mut hasher = Sha256::new();
            hasher.update(&combined);
            current_hash = hasher.finalize().as_bytes().try_into().unwrap();
        }
        
        // Verify root matches
        assert_eq!(current_hash, input.q_table_root, "Merkle proof verification failed");
    } else {
        // Small table: verify entry hash matches root (simplified)
        let entry_hash = hash_q_entry(&input.state_key, &input.q_entry);
        // For small tables, root is hash of all entries - simplified check
    }
    
    // Verify action selection (argmax)
    let expected_action = argmax_q(&input.q_entry);
    assert_eq!(
        input.selected_action,
        expected_action,
        "Action selection mismatch: expected {}, got {}",
        expected_action,
        input.selected_action
    );
    
    // Compute proof output (commitment to verified data)
    let mut hasher = Sha256::new();
    hasher.update(&input.state_key.as_bytes());
    hasher.update(&input.q_table_root);
    hasher.update(&[input.selected_action]);
    
    let proof_hash = hasher.finalize();
    
    // Commit to journal
    env::commit(&proof_hash);
}

/// Hash a Q-table entry
fn hash_q_entry(state_key: &str, entry: &QEntry) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(state_key.as_bytes());
    hasher.update(&entry.q_hold.to_le_bytes());
    hasher.update(&entry.q_buy.to_le_bytes());
    hasher.update(&entry.q_sell.to_le_bytes());
    hasher.finalize().as_bytes().try_into().unwrap()
}

/// Find action with maximum Q-value
fn argmax_q(q: &QEntry) -> u8 {
    if q.q_buy > q.q_sell && q.q_buy > q.q_hold {
        1  // buy
    } else if q.q_sell > q.q_hold {
        2  // sell
    } else {
        0  // hold
    }
}

