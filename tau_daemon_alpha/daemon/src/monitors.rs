use anyhow::Result;
use std::path::PathBuf;
use std::sync::Mutex;
use tau_core::{Config, model::*};
use tracing::{debug, warn, error};

use crate::fsio::FileIO;

pub struct MonitorManager {
    config: Config,
    outputs_dir: PathBuf,
    last_progress_tick: Mutex<Option<u64>>,
    alarm_latched: Mutex<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_timeout_exit_requires_sell_only_when_timed_out_in_state() {
        assert!(MonitorManager::derive_timeout_exit(false, true, true, false));
        assert!(MonitorManager::derive_timeout_exit(true, false, true, false));
        assert!(MonitorManager::derive_timeout_exit(true, true, false, false));
        assert!(!MonitorManager::derive_timeout_exit(true, true, true, false));
        assert!(MonitorManager::derive_timeout_exit(true, true, true, true));
    }

    #[test]
    fn derive_fresh_drop_exit_requires_sell_only_when_in_state_and_not_fresh() {
        assert!(MonitorManager::derive_fresh_drop_exit(false, false, false));
        assert!(MonitorManager::derive_fresh_drop_exit(true, true, false));
        assert!(!MonitorManager::derive_fresh_drop_exit(true, false, false));
        assert!(MonitorManager::derive_fresh_drop_exit(true, false, true));
    }
}

impl MonitorManager {
    const STALL_TICKS: u64 = 8;

    pub fn new(config: Config, outputs_dir: PathBuf) -> Self {
        Self {
            config,
            outputs_dir,
            last_progress_tick: Mutex::new(None),
            alarm_latched: Mutex::new(false),
        }
    }

    fn read_kernel_bit(&self, filename: &str) -> Result<bool> {
        FileIO::read_last_bool(&self.outputs_dir.join(filename))
    }

    fn derive_timeout_exit(state: bool, timer_b0: bool, timer_b1: bool, sell: bool) -> bool {
        let timed_out = timer_b0 && timer_b1;
        if !(state && timed_out) {
            return true;
        }

        sell
    }

    fn derive_fresh_drop_exit(state: bool, oracle_fresh: bool, sell: bool) -> bool {
        if !(state && !oracle_fresh) {
            return true;
        }

        sell
    }

    /// Read monitor outputs from files
    pub async fn read_outputs(&self, tick: u64) -> Result<MonitorOutputs> {
        debug!("Reading monitor outputs for tick {}", tick);

        let state = self.read_kernel_bit("state.out")?;
        let sell = self.read_kernel_bit("sell_signal.out")?;
        let oracle_fresh = self.read_kernel_bit("oracle_fresh.out")?;
        let timer_b0 = self.read_kernel_bit("timer_b0.out")?;
        let timer_b1 = self.read_kernel_bit("timer_b1.out")?;
        let progress = self.read_kernel_bit("progress_flag.out")?;

        let inv_action_excl = self.read_kernel_bit("obs_action_excl.out")?;
        let inv_fresh = self.read_kernel_bit("obs_fresh_exec.out")?;
        let inv_burn_profit = self.read_kernel_bit("obs_burn_profit.out")?;
        let inv_nonce = self.read_kernel_bit("obs_nonce_effect.out")?;

        let inv_timeout_exit = Self::derive_timeout_exit(state, timer_b0, timer_b1, sell);
        let inv_fresh_drop_exit = Self::derive_fresh_drop_exit(state, oracle_fresh, sell);

        let liveness_progress = progress;

        if liveness_progress {
            if let Ok(mut last) = self.last_progress_tick.lock() {
                *last = Some(tick);
            }
        }

        let last_progress_tick = self.last_progress_tick.lock().ok().and_then(|l| *l);
        let liveness_stalled = match last_progress_tick {
            Some(last) => tick.saturating_sub(last) > Self::STALL_TICKS,
            None => false,
        };

        let health_ok = inv_action_excl
            && inv_fresh
            && inv_burn_profit
            && inv_nonce
            && inv_timeout_exit
            && inv_fresh_drop_exit
            && !liveness_stalled;

        if !health_ok {
            if let Ok(mut latched) = self.alarm_latched.lock() {
                *latched = true;
            }
        }

        let alarm_latched = self.alarm_latched.lock().map(|v| *v).unwrap_or(true);

        let outputs = MonitorOutputs {
            inv_action_excl,
            inv_fresh,
            inv_burn_profit,
            inv_nonce,
            inv_timeout_exit,
            inv_fresh_drop_exit,
            liveness_progress,
            liveness_stalled,
            health_ok,
            alarm_latched,
        };

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

        let required = [
            "state.out",
            "sell_signal.out",
            "oracle_fresh.out",
            "timer_b0.out",
            "timer_b1.out",
            "progress_flag.out",
            "obs_action_excl.out",
            "obs_fresh_exec.out",
            "obs_burn_profit.out",
            "obs_nonce_effect.out",
        ];

        for filename in required {
            let _ = self.read_kernel_bit(filename)?;
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
} 