use anyhow::Result;
use tau_core::{
    config::Config,
    model::{KernelInputs, MarketSnapshot, TickGuards},
};
use tracing::info;

/// Manages interaction with the Tau kernel specification.
#[derive(Debug)]
pub struct KernelManager;

impl KernelManager {
    /// Creates a new `KernelManager`.
    pub fn new() -> Self {
        info!("Initializing kernel manager.");
        Self
    }

    /// Prepares the kernel inputs based on the current market state and configuration.
    pub fn prepare_inputs(
        &self,
        snapshot: &MarketSnapshot,
        previous_mid: Option<f64>,
        guards: &TickGuards,
        config: &Config,
    ) -> Result<KernelInputs> {
        info!("Preparing kernel inputs...");

        let mid_price = (snapshot.bid_price + snapshot.ask_price) / 2.0;
        let price_eps = 0.000_000_1f64; // negligible epsilon to avoid flapping
        let price_bit = previous_mid
            .map(|prev| mid_price > (prev * (1.0 + price_eps)))
            .unwrap_or(true);
        let trend_bit = previous_mid.map(|prev| mid_price > prev).unwrap_or(false);

        let volume_bit = snapshot
            .quote_volume
            .or(snapshot.base_volume)
            .map(|vol| vol > 0.0)
            .unwrap_or(false);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        let data_age_ms = now_ms.saturating_sub(snapshot.timestamp.saturating_mul(1000));
        let oracle_fresh = data_age_ms <= config.oracle.max_age_ms;

        let inputs = KernelInputs {
            price_bit,
            volume_bit,
            trend_bit,
            profit_guard: guards.profit_guard,
            failure_echo: guards.failure_echo,
            oracle_fresh,
        };

        info!(?inputs, "Prepared kernel inputs.");
        Ok(inputs)
    }
}
