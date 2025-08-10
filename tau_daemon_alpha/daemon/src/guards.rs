use anyhow::Result;
use tau_core::{Config, model::*};
use async_trait::async_trait;

pub mod freshness;
pub mod profit;
pub mod failure;
pub mod cooldown;
pub mod risk;

/// Trait for freshness witness implementation
#[async_trait]
pub trait FreshnessWitness: Send + Sync {
    async fn check(&self, oracle_data: &OracleSnapshot, config: &Config) -> Result<bool>;
    async fn get_diagnostics(&self) -> Result<(bool, bool, Vec<String>)>;
}

/// Trait for profit guard implementation
#[async_trait]
pub trait ProfitGuard: Send + Sync {
    async fn compute(&mut self, venue_state: &VenueState, config: &Config, tick: u64) -> Result<bool>;
    async fn get_economics(&self) -> Result<Option<Economics>>;
}

/// Trait for failure echo implementation
#[async_trait]
pub trait FailureEcho: Send + Sync {
    async fn compute_and_latch(&mut self, config: &Config, health: &Health, tick: u64) -> Result<bool>;
    async fn get_hold_counter(&self) -> u32;
    async fn clear(&mut self) -> Result<()>;
}

/// Trait for cooldown implementation
#[async_trait]
pub trait CooldownGuard: Send + Sync {
    async fn compute(&self, last_trade_tick: Option<u64>, current_tick: u64, config: &Config) -> Result<bool>;
    async fn get_ticks_since_trade(&self) -> Option<u32>;
}

/// Trait for risk management
#[async_trait]
pub trait RiskGuard: Send + Sync {
    async fn compute(&mut self, venue_state: &VenueState, config: &Config) -> Result<bool>;
    async fn get_risk_metrics(&self) -> Result<RiskMetrics>;
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub position_ticks: u32,
    pub drawdown_bps: u32,
    pub max_drawdown_bps: u32,
}

/// Coordinator for all guards
pub struct GuardCoordinator {
    pub freshness: Box<dyn FreshnessWitness>,
    pub profit: Box<dyn ProfitGuard>,
    pub failure: Box<dyn FailureEcho>,
    pub cooldown: Box<dyn CooldownGuard>,
    pub risk: Box<dyn RiskGuard>,
}

impl GuardCoordinator {
    pub fn new(
        freshness: Box<dyn FreshnessWitness>,
        profit: Box<dyn ProfitGuard>,
        failure: Box<dyn FailureEcho>,
        cooldown: Box<dyn CooldownGuard>,
        risk: Box<dyn RiskGuard>,
    ) -> Self {
        Self {
            freshness,
            profit,
            failure,
            cooldown,
            risk,
        }
    }

    /// Compute all guards for a tick
    pub async fn compute_guards(
        &mut self,
        oracle_snapshot: &OracleSnapshot,
        venue_state: &VenueState,
        health: &Health,
        config: &Config,
        tick: u64,
    ) -> Result<TickGuards> {
        // Compute freshness witness
        let fresh_witness = self.freshness.check(oracle_snapshot, config).await?;

        // Compute profit guard
        let profit_guard = self.profit.compute(venue_state, config, tick).await?;

        // Compute failure echo (with latching)
        let failure_echo = self.failure.compute_and_latch(config, health, tick).await?;

        // Compute cooldown (if enabled)
        let cooldown_ok = if config.cooldown.enabled {
            self.cooldown.compute(venue_state.last_trade_tick, tick, config).await?
        } else {
            true
        };

        // Compute risk guard
        let risk_ok = self.risk.compute(venue_state, config).await?;

        Ok(TickGuards {
            fresh_witness,
            profit_guard,
            failure_echo,
            cooldown_ok,
            risk_ok,
        })
    }

    /// Get diagnostics for logging
    pub async fn get_diagnostics(&self) -> Result<GuardDiagnostics> {
        let (age_ok, quorum_ok, violating_oracles) = self.freshness.get_diagnostics().await?;
        let economics = self.profit.get_economics().await?;
        let hold_counter = self.failure.get_hold_counter().await;
        let ticks_since_trade = self.cooldown.get_ticks_since_trade().await;
        let risk_metrics = self.risk.get_risk_metrics().await?;

        Ok(GuardDiagnostics {
            age_ok,
            quorum_ok,
            violating_oracles,
            economics,
            hold_counter,
            ticks_since_trade,
            risk_metrics,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GuardDiagnostics {
    pub age_ok: bool,
    pub quorum_ok: bool,
    pub violating_oracles: Vec<String>,
    pub economics: Option<Economics>,
    pub hold_counter: u32,
    pub ticks_since_trade: Option<u32>,
    pub risk_metrics: RiskMetrics,
} 