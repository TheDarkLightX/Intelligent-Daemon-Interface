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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;
    use tempfile::tempdir;

    fn test_config(tempdir: &Path) -> Config {
        use std::path::PathBuf;

        Config {
            daemon: tau_core::config::DaemonConfig {
                tick_period_ms: 1000,
                quarantine_clear_ticks: 3,
            },
            tick: tau_core::config::TickConfig {
                interval_ms: 100,
                fail_hold_ticks: 2,
            },
            paths: tau_core::config::PathsConfig {
                tau_bin: PathBuf::from("/usr/bin/true"),
                kernel_spec: tempdir.join("kernel.tau"),
                kernel_inputs: tempdir.join("inputs"),
                kernel_outputs: tempdir.join("outputs"),
                specs_root: tempdir.join("specs"),
                data_dir: tempdir.join("data"),
                ledger_dir: tempdir.join("ledger"),
            },
            oracle: tau_core::config::OracleConfig {
                max_age_ms: 1000,
                min_quorum: 1,
                tolerance_bps: 5,
            },
            economics: tau_core::config::EconomicsConfig {
                scale: 1,
                fee_bps_buy: 0,
                fee_bps_sell: 0,
                slippage_bps_limit: 0,
                gas_unit_cost: 0,
                trade_qty: 1,
            },
            cooldown: tau_core::config::CooldownConfig {
                enabled: true,
                ticks: 1,
            },
            risk: tau_core::config::RiskConfig {
                max_position_ticks: 1,
                max_drawdown_bps: 1,
            },
            exchanges: HashMap::new(),
            thresholds: tau_core::config::ThresholdsConfig {
                vol_high_percentile: 0.0,
                slip_bps: 0,
                fee_bps: 0,
                min_profit_usd: 0.0,
                gas_usd_p99: 0.0,
                burn_alpha: 0.0,
            },
            solver: tau_core::config::SolverConfig {
                engine: "mock".to_string(),
                max_time_sec: 1.0,
                num_workers: 1,
                log: false,
                scaling_factor: 1,
            },
            wallet: tau_core::config::WalletConfig {
                chain: "devnet".to_string(),
                rpc: "http://localhost".to_string(),
                keyfile: PathBuf::from("/tmp/keyfile"),
            },
        }
    }

    #[tokio::test]
    async fn validate_files_reports_missing_outputs() {
        let dir = tempdir().unwrap();
        let cfg = test_config(dir.path());
        let manager = MonitorManager::new(cfg);

        let err = manager.validate_files().await.expect_err("should fail");
        assert!(err.to_string().contains("Missing monitor output file"));
    }

    #[tokio::test]
    async fn aggregate_health_tracks_invariant_failures() {
        let dir = tempdir().unwrap();
        let cfg = test_config(dir.path());
        let manager = MonitorManager::new(cfg.clone());

        // Seed output files so validation passes
        std::fs::create_dir_all(&cfg.paths.kernel_outputs).unwrap();
        let outputs_dir = &cfg.paths.kernel_outputs;
        for (name, value) in [
            ("inv_action_excl.out", true),
            ("inv_fresh.out", false),
            ("inv_burn_profit.out", true),
            ("inv_nonce.out", false),
            ("inv_timeout_exit.out", true),
            ("inv_fresh_drop_exit.out", true),
            ("liveness_progress.out", true),
            ("liveness_stalled.out", false),
        ] {
            FileIO::write_bool_atomic(&outputs_dir.join(name), value).unwrap();
        }

        manager.validate_files().await.expect("files should exist");
        let outputs = manager.read_outputs(1).await.expect("outputs should parse");
        let health = manager.aggregate_health(&outputs);

        assert!(!health.health_ok);
        assert!(health.failed_bits.contains(&"inv_fresh".to_string()));
        assert!(health.failed_bits.contains(&"inv_nonce".to_string()));
    }
}
