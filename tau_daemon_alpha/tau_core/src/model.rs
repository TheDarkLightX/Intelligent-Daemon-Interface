use serde::{Deserialize, Serialize};
use std::fmt;

/// Fixed-point scaled integer for monetary calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Fx(u128);

impl Fx {
    pub const SCALE: u128 = 1_000_000_000; // 1e9

    pub fn new(value: u128) -> Self {
        Self(value)
    }

    pub fn from_float(f: f64) -> Self {
        Self((f * Self::SCALE as f64) as u128)
    }

    pub fn to_float(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn raw(self) -> u128 {
        self.0
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }

    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let result = self.0.checked_mul(other.0)?;
        Some(Self(result / Self::SCALE))
    }

    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.0 == 0 {
            return None;
        }
        let result = self.0.checked_mul(Self::SCALE)?;
        Some(Self(result / other.0))
    }
}

impl fmt::Display for Fx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.9}", self.to_float())
    }
}

/// Tick guards for the kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickGuards {
    pub fresh_witness: bool,
    pub profit_guard: bool,
    pub failure_echo: bool,
    pub cooldown_ok: bool,
    pub risk_ok: bool,
}

/// Kernel outputs from V35
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelOutputs {
    pub state: bool,
    pub holding: bool,
    pub buy: bool,
    pub sell: bool,
    pub oracle_fresh: bool,
    pub timer_b0: bool,
    pub timer_b1: bool,
    pub nonce: bool,
    pub entry_price: bool,
    pub profit: bool,
    pub burn: bool,
    pub has_burned: bool,
    pub obs_action_excl: bool,
    pub obs_fresh_exec: bool,
    pub obs_burn_profit: bool,
    pub obs_nonce_effect: bool,
    pub progress: bool,
}

/// Monitor outputs from supporting specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorOutputs {
    pub inv_action_excl: bool,
    pub inv_fresh: bool,
    pub inv_burn_profit: bool,
    pub inv_nonce: bool,
    pub inv_timeout_exit: bool,
    pub inv_fresh_drop_exit: bool,
    pub liveness_progress: bool,
    pub liveness_stalled: bool,
    pub health_ok: bool,
    pub alarm_latched: bool,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub health_ok: bool,
    pub failed_bits: Vec<String>,
}

impl Health {
    pub fn new() -> Self {
        Self {
            health_ok: true,
            failed_bits: Vec::new(),
        }
    }

    pub fn add_failure(&mut self, bit: &str) {
        self.health_ok = false;
        self.failed_bits.push(bit.to_string());
    }
}

/// Oracle snapshot with diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSnapshot {
    pub median: Fx,
    pub age_ok: bool,
    pub quorum_ok: bool,
    pub sources: Vec<OracleSource>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSource {
    pub price: Fx,
    pub age_ms: u64,
    pub within_tol: bool,
}

/// Economic calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Economics {
    pub buy_price: Fx,
    pub sell_price: Fx,
    pub qty: Fx,
    pub fee_buy: Fx,
    pub fee_sell: Fx,
    pub gas: Fx,
    pub slippage_bps: u32,
    pub pnl: Fx,
}

/// Venue state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueState {
    pub position: Fx,
    pub balance: Fx,
    pub last_trade_tick: Option<u64>,
    pub pending_orders: Vec<Order>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub side: OrderSide,
    pub qty: Fx,
    pub price: Fx,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Failed,
}

/// Ledger record for each tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerRecord {
    pub tick: u64,
    pub ts: String,
    pub guards: TickGuards,
    pub kernel: KernelOutputs,
    pub monitors: MonitorOutputs,
    pub health_ok: bool,
    pub oracle_snapshot: OracleSnapshot,
    pub economics: Option<Economics>,
    pub actions: Vec<Action>,
    pub prev_hash: Option<String>,
    pub hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: String,
    pub id: String,
    pub status: String,
}

/// Legacy types for backward compatibility
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub bid_price: f64,
    pub ask_price: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct KernelInputs {
    pub i0_price_bit: bool,
    pub i1_oracle_fresh: bool,
    pub i2_trend_bit: bool,
    pub i3_profit_guard: bool,
    pub i4_failure_echo: bool,
}
