#![no_std]
#![no_main]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use risc0_zkvm::guest::env;
use serde::Deserialize;
use sha2::{Digest, Sha256};

risc0_zkvm::guest::entry!(main);

#[derive(Deserialize)]
struct FileBlob {
    name: String,
    data: Vec<u8>,
}

pub fn main() {
    let blobs: Vec<FileBlob> = env::read();
    let mut hasher = Sha256::new();
    for blob in blobs {
        hasher.update(blob.name.as_bytes());
        hasher.update((blob.data.len() as u64).to_le_bytes());
        hasher.update(&blob.data);
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    env::commit(&out);
}

