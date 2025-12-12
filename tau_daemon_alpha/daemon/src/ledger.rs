use anyhow::Result;
use serde_json;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tau_core::model::*;
use tracing::{debug, info, warn};

pub struct Ledger {
    ledger_path: std::path::PathBuf,
    last_hash: Option<String>,
}

impl Ledger {
    pub fn new(ledger_dir: &Path) -> Result<Self> {
        fs::create_dir_all(ledger_dir)?;

        let ledger_path = ledger_dir.join("ticks.log");

        // Find the last hash from existing ledger
        let last_hash = Self::find_last_hash(&ledger_path)?;

        info!(
            "Initialized ledger at {:?} with last hash: {:?}",
            ledger_path, last_hash
        );

        Ok(Self {
            ledger_path,
            last_hash,
        })
    }

    /// Find the last hash from the ledger file
    fn find_last_hash(ledger_path: &Path) -> Result<Option<String>> {
        if !ledger_path.exists() {
            return Ok(None);
        }

        let file = fs::File::open(ledger_path)?;
        let reader = BufReader::new(file);
        let mut last_hash = None;

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(record) = serde_json::from_str::<LedgerRecord>(&line) {
                    last_hash = record.hash;
                }
            }
        }

        Ok(last_hash)
    }

    /// Append a ledger record
    pub async fn append(
        &mut self,
        tick: u64,
        oracle_snapshot: &OracleSnapshot,
        kernel_outputs: &KernelOutputs,
        monitor_outputs: &MonitorOutputs,
        guards: &TickGuards,
        health: &Health,
        economics: Option<&Economics>,
        actions: Vec<Action>,
    ) -> Result<()> {
        debug!("Appending ledger record for tick {}", tick);

        let record = LedgerRecord {
            tick,
            ts: chrono::Utc::now().to_rfc3339(),
            guards: guards.clone(),
            kernel: kernel_outputs.clone(),
            monitors: monitor_outputs.clone(),
            health_ok: health.health_ok,
            oracle_snapshot: oracle_snapshot.clone(),
            economics: economics.cloned(),
            actions,
            prev_hash: self.last_hash.clone(),
            hash: None, // Will be computed below
        };

        // Compute hash
        let mut hasher = Sha256::new();
        let record_json = serde_json::to_string(&record)?;
        hasher.update(record_json.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Create final record with hash
        let mut final_record = record;
        final_record.hash = Some(hash.clone());

        // Append to file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.ledger_path)?;

        let record_json = serde_json::to_string(&final_record)?;
        writeln!(file, "{}", record_json)?;
        file.flush()?;

        // Update last hash
        self.last_hash = Some(hash.clone());

        debug!(
            "Appended ledger record for tick {} with hash {}",
            tick, hash
        );
        Ok(())
    }

    /// Create a periodic snapshot
    pub async fn create_snapshot(&self, tick: u64) -> Result<()> {
        let snapshot_dir = self.ledger_path.parent().unwrap().join("snapshots");
        fs::create_dir_all(&snapshot_dir)?;

        let snapshot_path = snapshot_dir.join(format!("tick_{:08}.json", tick));

        // Copy current ledger to snapshot
        if self.ledger_path.exists() {
            fs::copy(&self.ledger_path, &snapshot_path)?;
            info!("Created snapshot at {:?}", snapshot_path);
        }

        Ok(())
    }

    /// Recover state from ledger
    pub async fn recover_state(&self) -> Result<Option<RecoveryState>> {
        if !self.ledger_path.exists() {
            return Ok(None);
        }

        let file = fs::File::open(&self.ledger_path)?;
        let reader = BufReader::new(file);
        let mut last_record = None;

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(record) = serde_json::from_str::<LedgerRecord>(&line) {
                    last_record = Some(record);
                }
            }
        }

        if let Some(record) = last_record {
            let recovery_state = RecoveryState {
                last_tick: record.tick,
                last_hash: record.hash,
                kernel_state: record.kernel,
                health: record.health_ok,
            };
            Ok(Some(recovery_state))
        } else {
            Ok(None)
        }
    }

    /// Validate ledger integrity
    pub async fn validate_integrity(&self) -> Result<bool> {
        if !self.ledger_path.exists() {
            return Ok(true);
        }

        let file = fs::File::open(&self.ledger_path)?;
        let reader = BufReader::new(file);
        let mut prev_hash = None;
        let mut line_number = 0;

        for line in reader.lines() {
            line_number += 1;
            if let Ok(line) = line {
                if let Ok(record) = serde_json::from_str::<LedgerRecord>(&line) {
                    // Check prev_hash matches
                    if let Some(expected_prev) = prev_hash {
                        if record.prev_hash != Some(expected_prev) {
                            warn!("Hash chain broken at line {}", line_number);
                            return Ok(false);
                        }
                    }

                    // Check current hash
                    if let Some(hash) = &record.hash {
                        let mut hasher = Sha256::new();
                        let mut record_for_hash = record.clone();
                        record_for_hash.hash = None;
                        let record_json = serde_json::to_string(&record_for_hash)?;
                        hasher.update(record_json.as_bytes());
                        let computed_hash = format!("{:x}", hasher.finalize());

                        if hash != &computed_hash {
                            warn!("Hash mismatch at line {}", line_number);
                            return Ok(false);
                        }
                    }

                    prev_hash = record.hash;
                }
            }
        }

        info!("Ledger integrity validation passed");
        Ok(true)
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryState {
    pub last_tick: u64,
    pub last_hash: Option<String>,
    pub kernel_state: KernelOutputs,
    pub health: bool,
}
