use anyhow::Result;
use async_trait::async_trait;
use tau_core::{Config, model::*};
use tracing::{debug, warn};

pub struct RiskGuardImpl {
    last_metrics: Option<super::RiskMetrics>,
}

impl RiskGuardImpl {
    pub fn new() -> Self {
        Self {
            last_metrics: None,
        }
    }

    /// Compute position duration in ticks
    fn compute_position_ticks(&self, venue_state: &VenueState, current_tick: u64) -> u32 {
        match venue_state.last_trade_tick {
            Some(last_tick) => current_tick.saturating_sub(last_tick) as u32,
            None => 0,
        }
    }

    /// Compute drawdown in basis points
    fn compute_drawdown_bps(&self, venue_state: &VenueState) -> u32 {
        // This is a simplified drawdown calculation
        // In a real implementation, this would track peak value and current value
        if venue_state.balance.raw() == 0 {
            return 0;
        }

        // For now, assume no drawdown
        // This would be computed as: (peak_value - current_value) / peak_value * 10000
        0
    }
}

#[async_trait]
impl super::RiskGuard for RiskGuardImpl {
    async fn compute(&mut self, venue_state: &VenueState, config: &Config) -> Result<bool> {
        debug!("Computing risk guard...");

        // Use UNIX time seconds as a coarse stand-in for tick index if last_trade_tick is recorded in ticks; 
        // In production, pass the actual tick counter from the looper.
        let current_tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let position_ticks = self.compute_position_ticks(venue_state, current_tick);
        let drawdown_bps = self.compute_drawdown_bps(venue_state);

        let position_ok = position_ticks <= config.risk.max_position_ticks;
        let drawdown_ok = drawdown_bps <= config.risk.max_drawdown_bps;

        let risk_ok = position_ok && drawdown_ok;

        let metrics = super::RiskMetrics {
            position_ticks,
            drawdown_bps,
            max_drawdown_bps: config.risk.max_drawdown_bps,
        };

        self.last_metrics.replace(metrics.clone());

        debug!(
            "Risk guard: position_ticks={}, drawdown_bps={}, position_ok={}, drawdown_ok={}, risk_ok={}",
            position_ticks, drawdown_bps, position_ok, drawdown_ok, risk_ok
        );

        if !risk_ok {
            warn!(
                "Risk guard failed: position_ticks={}, drawdown_bps={}, max_position_ticks={}, max_drawdown_bps={}",
                position_ticks, drawdown_bps, config.risk.max_position_ticks, config.risk.max_drawdown_bps
            );
        }

        Ok(risk_ok)
    }

    async fn get_risk_metrics(&self) -> Result<super::RiskMetrics> {
        Ok(self.last_metrics.clone().unwrap_or(super::RiskMetrics {
            position_ticks: 0,
            drawdown_bps: 0,
            max_drawdown_bps: 0,
        }))
    }
} 