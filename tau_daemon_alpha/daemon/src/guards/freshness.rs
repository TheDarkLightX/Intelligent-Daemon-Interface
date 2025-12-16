use anyhow::Result;
use async_trait::async_trait;
use std::sync::Mutex;
use tau_core::{Config, model::*};
use tracing::{debug, warn};

pub struct FreshnessWitnessImpl {
    last_diagnostics: Mutex<(bool, bool, Vec<String>)>,
}

impl FreshnessWitnessImpl {
    pub fn new() -> Self {
        Self {
            last_diagnostics: Mutex::new((true, true, Vec::new())),
        }
    }

    /// Check if oracle data is fresh enough
    fn check_age(&self, oracle_snapshot: &OracleSnapshot, config: &Config) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let age_ms = now.saturating_sub(oracle_snapshot.timestamp);
        let age_ok = age_ms <= config.oracle.max_age_ms;
        
        if !age_ok {
            warn!("Oracle data too old: {}ms > {}ms", age_ms, config.oracle.max_age_ms);
        }
        
        age_ok
    }

    /// Check if enough oracle sources agree within tolerance
    fn check_quorum(&self, oracle_snapshot: &OracleSnapshot, config: &Config) -> (bool, Vec<String>) {
        let median = oracle_snapshot.median;
        let tolerance = median.checked_mul(Fx::new(config.oracle.tolerance_bps as u128))
            .and_then(|t| t.checked_div(Fx::new(10_000)))
            .unwrap_or(Fx::new(0));

        let mut within_tolerance = 0;
        let mut violating_sources = Vec::new();

        for (i, source) in oracle_snapshot.sources.iter().enumerate() {
            let diff = if source.price > median {
                source.price.checked_sub(median)
            } else {
                median.checked_sub(source.price)
            };

            if let Some(diff) = diff {
                if diff <= tolerance {
                    within_tolerance += 1;
                } else {
                    violating_sources.push(format!("oracle_{}", i));
                }
            } else {
                violating_sources.push(format!("oracle_{}_overflow", i));
            }
        }

        let quorum_ok = within_tolerance >= config.oracle.min_quorum;
        
        if !quorum_ok {
            warn!(
                "Oracle quorum failed: {}/{} within tolerance (need {})",
                within_tolerance,
                oracle_snapshot.sources.len(),
                config.oracle.min_quorum
            );
        }

        (quorum_ok, violating_sources)
    }
}

#[async_trait]
impl super::FreshnessWitness for FreshnessWitnessImpl {
    async fn check(&self, oracle_data: &OracleSnapshot, config: &Config) -> Result<bool> {
        debug!("Checking freshness witness...");
        
        let age_ok = self.check_age(oracle_data, config);
        let (quorum_ok, violating_sources) = self.check_quorum(oracle_data, config);
        let fresh_witness = age_ok && quorum_ok;

        if let Ok(mut diag) = self.last_diagnostics.lock() {
            *diag = (age_ok, quorum_ok, violating_sources);
        }
        
        debug!(
            "Freshness witness: age_ok={}, quorum_ok={}, fresh_witness={}",
            age_ok, quorum_ok, fresh_witness
        );
        
        Ok(fresh_witness)
    }

    async fn get_diagnostics(&self) -> Result<(bool, bool, Vec<String>)> {
        let diag = self
            .last_diagnostics
            .lock()
            .map(|d| d.clone())
            .unwrap_or((true, true, Vec::new()));
        Ok(diag)
    }
}