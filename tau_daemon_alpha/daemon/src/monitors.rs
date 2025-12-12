use anyhow::Result;
use std::path::Path;
use tau_core::{model::*, Config};
use tracing::{debug, warn};

use crate::fsio::FileIO;

pub struct MonitorManager {
    config: Config,
}

impl MonitorManager {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Read monitor outputs from files
    pub async fn read_outputs(&self, tick: u64) -> Result<MonitorOutputs> {
        debug!("Reading monitor outputs for tick {}", tick);

        self.validate_files().await?;
        let outputs = self.load_outputs(&self.config.paths.kernel_outputs)?;

        debug!("Monitor outputs: {:?}", outputs);
        Ok(outputs)
    }

    /// Aggregate health from monitor outputs
    pub fn aggregate_health(&self, outputs: &MonitorOutputs) -> Health {
        let mut health = Health::new();

        // Check each invariant
        if !outputs.inv_action_excl {
            health.add_failure("inv_action_excl");
        }
        if !outputs.inv_fresh {
            health.add_failure("inv_fresh");
        }
        if !outputs.inv_burn_profit {
            health.add_failure("inv_burn_profit");
        }
        if !outputs.inv_nonce {
            health.add_failure("inv_nonce");
        }
        if !outputs.inv_timeout_exit {
            health.add_failure("inv_timeout_exit");
        }
        if !outputs.inv_fresh_drop_exit {
            health.add_failure("inv_fresh_drop_exit");
        }

        // Check liveness
        if outputs.liveness_stalled {
            health.add_failure("liveness_stalled");
        }

        debug!("Health aggregation: {:?}", health);
        health
    }

    /// Validate that all required monitor files exist and are readable
    pub async fn validate_files(&self) -> Result<()> {
        debug!("Validating monitor files...");

        let outputs_dir = &self.config.paths.kernel_outputs;
        let required_files = [
            "inv_action_excl.out",
            "inv_fresh.out",
            "inv_burn_profit.out",
            "inv_nonce.out",
            "inv_timeout_exit.out",
            "inv_fresh_drop_exit.out",
            "liveness_progress.out",
            "liveness_stalled.out",
        ];

        for file in required_files.iter() {
            let path = outputs_dir.join(file);
            if !path.exists() {
                return Err(anyhow::anyhow!("Missing monitor output file: {:?}", path));
            }
        }

        debug!("Monitor files validation passed");
        Ok(())
    }

    /// Check if monitors are stalled
    pub async fn check_stalled(&self, outputs: &MonitorOutputs) -> Result<bool> {
        let stalled = outputs.liveness_stalled;

        if stalled {
            warn!("Monitors detected stall condition");
        }

        Ok(stalled)
    }

    fn load_outputs(&self, outputs_dir: &Path) -> Result<MonitorOutputs> {
        let outputs = MonitorOutputs {
            inv_action_excl: FileIO::read_last_bool(&outputs_dir.join("inv_action_excl.out"))?,
            inv_fresh: FileIO::read_last_bool(&outputs_dir.join("inv_fresh.out"))?,
            inv_burn_profit: FileIO::read_last_bool(&outputs_dir.join("inv_burn_profit.out"))?,
            inv_nonce: FileIO::read_last_bool(&outputs_dir.join("inv_nonce.out"))?,
            inv_timeout_exit: FileIO::read_last_bool(&outputs_dir.join("inv_timeout_exit.out"))?,
            inv_fresh_drop_exit: FileIO::read_last_bool(
                &outputs_dir.join("inv_fresh_drop_exit.out"),
            )?,
            liveness_progress: FileIO::read_last_bool(&outputs_dir.join("liveness_progress.out"))?,
            liveness_stalled: FileIO::read_last_bool(&outputs_dir.join("liveness_stalled.out"))?,
            health_ok: true,
            alarm_latched: false,
        };

        Ok(outputs)
    }
}
