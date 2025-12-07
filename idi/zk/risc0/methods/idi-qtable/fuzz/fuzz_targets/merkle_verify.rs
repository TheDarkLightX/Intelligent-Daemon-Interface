#![no_main]

use libfuzzer_sys::fuzz_target;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct MerkleProofInput {
    state_key: String,
    q_entry: QEntry,
    merkle_proof: Option<Vec<MerklePathElement>>,
    q_table_root: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct QEntry {
    q_hold: i32,
    q_buy: i32,
    q_sell: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MerklePathElement {
    sibling_hash: Vec<u8>,
    is_right: bool,
}

fuzz_target!(|data: &[u8]| {
    // Try to deserialize as Merkle proof input
    if let Ok(input) = serde_json::from_slice::<MerkleProofInput>(data) {
        // Verify root hash is 32 bytes
        if input.q_table_root.len() != 32 {
            return;
        }
        
        // Verify all proof path elements have 32-byte hashes
        if let Some(proof) = &input.merkle_proof {
            for elem in proof {
                if elem.sibling_hash.len() != 32 {
                    return;
                }
            }
        }
        
        // Simulate Merkle proof verification (simplified)
        // In real fuzzing, would call actual verification function
        let mut current_hash = hash_q_entry(&input.state_key, &input.q_entry);
        
        if let Some(proof) = &input.merkle_proof {
            for path_elem in proof {
                let combined = if path_elem.is_right {
                    [&current_hash[..], &path_elem.sibling_hash[..]].concat()
                } else {
                    [&path_elem.sibling_hash[..], &current_hash[..]].concat()
                };
                current_hash = sha256(&combined);
            }
        }
        
        // Verify root matches (should not crash on malformed input)
        let _matches = current_hash == input.q_table_root;
    }
});

fn hash_q_entry(state_key: &str, entry: &QEntry) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    state_key.hash(&mut hasher);
    entry.q_hold.hash(&mut hasher);
    entry.q_buy.hash(&mut hasher);
    entry.q_sell.hash(&mut hasher);
    hasher.finish().to_le_bytes().to_vec()
}

fn sha256(data: &[u8]) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

