use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level configuration for the daemon, loaded from config.toml.
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub daemon: DaemonConfig,
    pub tick: TickConfig,
    pub paths: PathsConfig,
    pub oracle: OracleConfig,
    pub economics: EconomicsConfig,
    pub cooldown: CooldownConfig,
    pub risk: RiskConfig,
    pub exchanges: HashMap<String, ExchangeConfig>,
    pub thresholds: ThresholdsConfig,
    pub solver: SolverConfig,
    pub wallet: WalletConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DaemonConfig {
    /// The tick period in milliseconds.
    pub tick_period_ms: u64,
    /// Number of consecutive clean ticks to exit Quarantine.
    pub quarantine_clear_ticks: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TickConfig {
    /// Tick interval in milliseconds
    pub interval_ms: u64,
    /// Number of ticks to hold failure echo
    pub fail_hold_ticks: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PathsConfig {
    /// Path to Tau binary
    pub tau_bin: PathBuf,
    /// Path to kernel specification
    pub kernel_spec: PathBuf,
    /// Path to kernel inputs directory
    pub kernel_inputs: PathBuf,
    /// Path to kernel outputs directory
    pub kernel_outputs: PathBuf,
    /// Path to supporting specs root
    pub specs_root: PathBuf,
    /// Path to data directory
    pub data_dir: PathBuf,
    /// Path to ledger directory
    pub ledger_dir: PathBuf,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OracleConfig {
    /// Maximum age of oracle data in milliseconds
    pub max_age_ms: u64,
    /// Minimum quorum size
    pub min_quorum: u32,
    /// Tolerance in basis points
    pub tolerance_bps: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct EconomicsConfig {
    /// Fixed-point scale factor
    pub scale: u64,
    /// Buy fee in basis points
    pub fee_bps_buy: u32,
    /// Sell fee in basis points
    pub fee_bps_sell: u32,
    /// Slippage limit in basis points
    pub slippage_bps_limit: u32,
    /// Gas unit cost
    pub gas_unit_cost: u64,
    /// Trade quantity in scaled units
    pub trade_qty: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CooldownConfig {
    /// Whether cooldown is enabled
    pub enabled: bool,
    /// Number of ticks for cooldown
    pub ticks: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RiskConfig {
    /// Maximum position ticks
    pub max_position_ticks: u32,
    /// Maximum drawdown in basis points
    pub max_drawdown_bps: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ExchangeConfig {
    pub rest_url: String,
    pub pair: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ThresholdsConfig {
    pub vol_high_percentile: f64,
    pub slip_bps: u32,
    pub fee_bps: u32,
    pub min_profit_usd: f64,
    pub gas_usd_p99: f64,
    pub burn_alpha: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SolverConfig {
    pub engine: String,
    pub max_time_sec: f64,
    pub num_workers: u32,
    pub log: bool,
    /// The scaling factor for converting floats to integers for the solver.
    pub scaling_factor: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct WalletConfig {
    pub chain: String,
    pub rpc: String,
    pub keyfile: PathBuf,
}

impl Config {
    /// Loads configuration from a TOML file.
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }
}

