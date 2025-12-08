use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use idi_risc0_methods::{IDI_RISC0_METHODS_GUEST_ELF, IDI_RISC0_METHODS_GUEST_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(version, about = "Risc0 prover/verifier for IDI manifests")]
enum Args {
    /// Generate a proof from manifest and streams
    Prove {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        streams: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Verify a proof binary
    Verify {
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        streams: PathBuf,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct FileBlob {
    name: String,
    data: Vec<u8>,
}

fn gather_blobs(manifest: &Path, streams: &Path) -> Result<Vec<FileBlob>> {
    let mut blobs = Vec::new();
    let manifest_bytes = fs::read(manifest).with_context(|| format!("read manifest {}", manifest.display()))?;
    blobs.push(FileBlob {
        name: "manifest".to_string(),
        data: manifest_bytes,
    });

    if streams.exists() {
        let mut entries: Vec<_> = WalkDir::new(streams)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .collect();
        entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_lowercase());
        for entry in entries {
            let rel = entry
                .path()
                .strip_prefix(streams)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();
            let data = fs::read(entry.path()).with_context(|| format!("read stream {}", entry.path().display()))?;
            blobs.push(FileBlob {
                name: format!("streams/{}", rel),
                data,
            });
        }
    }
    Ok(blobs)
}

fn deterministic_hash(blobs: &[FileBlob]) -> String {
    let mut hasher = Sha256::new();
    for blob in blobs {
        hasher.update(blob.name.as_bytes());
        hasher.update((blob.data.len() as u64).to_le_bytes());
        hasher.update(&blob.data);
    }
    hex::encode(hasher.finalize())
}

fn write_json(path: &Path, value: serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &value)?;
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse() {
        Args::Prove {
            manifest,
            streams,
            proof,
            receipt,
        } => {
            let blobs = gather_blobs(&manifest, &streams)?;
            let env = ExecutorEnv::builder().write(&blobs)?.build()?;
            let prover = default_prover();
            let prove_info = prover.prove(env, IDI_RISC0_METHODS_GUEST_ELF)?;
            prove_info.receipt.verify(IDI_RISC0_METHODS_GUEST_ID)?;

            if let Some(parent) = proof.parent() {
                fs::create_dir_all(parent)?;
            }
            let proof_bytes = bincode::serialize(&prove_info.receipt)?;
            fs::write(&proof, proof_bytes)?;

            let digest: [u8; 32] = prove_info.receipt.journal.decode()?;
            let digest_hex = hex::encode(digest);
            let host_digest = deterministic_hash(&blobs);
            if host_digest != digest_hex {
                anyhow::bail!(
                    "guest digest {} does not match host digest {}",
                    digest_hex,
                    host_digest
                );
            }

            let receipt_json = json!({
                "prover": "risc0",
                "manifest": manifest,
                "streams": streams,
                "digest_hex": digest_hex,
                "method_id": format!("{IDI_RISC0_METHODS_GUEST_ID:?}"),
                "proof": proof,
            });
            write_json(&receipt, receipt_json)?;
            Ok(())
        }
        Args::Verify { proof, manifest, streams } => {
            // Read proof binary
            let proof_bytes = fs::read(&proof)
                .with_context(|| format!("read proof {}", proof.display()))?;
            
            // Deserialize receipt
            let receipt: risc0_zkvm::Receipt = bincode::deserialize(&proof_bytes)
                .context("deserialize proof binary")?;
            
            // Verify receipt against method ID
            receipt.verify(IDI_RISC0_METHODS_GUEST_ID)
                .context("verify receipt against method ID")?;
            
            // Decode journal to get digest
            let digest: [u8; 32] = receipt.journal.decode()
                .context("decode receipt journal")?;
            let digest_hex = hex::encode(digest);

            // Bind digest to manifest + streams deterministically
            let blobs = gather_blobs(&manifest, &streams)?;
            let host_digest = deterministic_hash(&blobs);
            if host_digest != digest_hex {
                bail!(
                    "guest digest {} does not match host digest {}",
                    digest_hex,
                    host_digest
                );
            }
            
            eprintln!("✓ Proof verified successfully");
            eprintln!("  Method ID: {IDI_RISC0_METHODS_GUEST_ID:?}");
            eprintln!("  Digest: {digest_hex}");
            
            Ok(())
        }
    }
}
