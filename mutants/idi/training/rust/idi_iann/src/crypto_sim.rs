use rand::prelude::*;

use crate::action::Action;
use crate::regime::Regime;

#[derive(Clone, Debug)]
pub struct MarketParams {
    pub drift_bull: f64,
    pub drift_bear: f64,
    pub drift_chop: f64,
    pub vol_base: f64,
    pub vol_panic: f64,
    pub shock_prob: f64,
    pub shock_scale: f64,
    pub fee_bps: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            drift_bull: 0.002,
            drift_bear: -0.002,
            drift_chop: 0.0,
            vol_base: 0.01,
            vol_panic: 0.04,
            shock_prob: 0.02,
            shock_scale: 0.05,
            fee_bps: 5.0,
        }
    }
}

impl MarketParams {
    /// Validate that all parameters are in valid ranges.
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err` with description if invalid
    pub fn validate(&self) -> Result<(), String> {
        if self.vol_base <= 0.0 {
            return Err("vol_base must be > 0".to_string());
        }
        if self.vol_panic <= 0.0 {
            return Err("vol_panic must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.shock_prob) {
            return Err("shock_prob must be in [0, 1]".to_string());
        }
        if self.shock_scale <= 0.0 {
            return Err("shock_scale must be > 0".to_string());
        }
        if self.fee_bps < 0.0 {
            return Err("fee_bps must be >= 0".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MarketState {
    pub price: f64,
    pub regime: Regime,
    pub t: usize,
    pub position: i32,
    pub pnl: f64,
}

pub struct CryptoSim {
    pub params: MarketParams,
    rng: StdRng,
    pub state: MarketState,
}

impl CryptoSim {
    /// Create a new crypto market simulator.
    ///
    /// # Arguments
    /// * `seed` - Random seed for reproducibility
    /// * `params` - Market parameters (drift, volatility, fees)
    ///
    /// # Returns
    /// Simulator instance with initial state
    ///
    /// # Panics
    /// Panics if `params.validate()` fails. Call `validate()` before constructing.
    pub fn new(seed: u64, params: MarketParams) -> Self {
        // Validate parameters to ensure expect() calls in compute_return() are safe
        params.validate().expect("MarketParams validation failed");
        Self {
            params,
            rng: StdRng::seed_from_u64(seed),
            state: MarketState {
                price: 1000.0,
                regime: Regime::Chop,
                t: 0,
                position: 0,
                pnl: 0.0,
            },
        }
    }

    /// Reset simulator to initial state.
    ///
    /// # Returns
    /// Reference to reset market state
    pub fn reset(&mut self) -> &MarketState {
        self.state = MarketState {
            price: 1000.0,
            regime: Regime::Chop,
            t: 0,
            position: 0,
            pnl: 0.0,
        };
        &self.state
    }

    /// Get drift rate for current regime.
    ///
    /// # Returns
    /// Drift rate (expected return per step)
    fn regime_drift(&self) -> f64 {
        match self.state.regime {
            Regime::Bull => self.params.drift_bull,
            Regime::Bear => self.params.drift_bear,
            Regime::Panic => -self.params.drift_bear.abs() * 1.5,
            Regime::Chop => self.params.drift_chop,
        }
    }

    /// Get volatility for current regime.
    ///
    /// # Returns
    /// Volatility (std dev of returns)
    fn regime_vol(&self) -> f64 {
        match self.state.regime {
            Regime::Panic => self.params.vol_panic,
            _ => self.params.vol_base,
        }
    }

    /// Possibly switch market regime (5% chance per step).
    fn maybe_switch_regime(&mut self) {
        if self.rng.gen::<f64>() < 0.05 {
            let regimes = [Regime::Bull, Regime::Bear, Regime::Chop, Regime::Panic];
            self.state.regime = regimes[self.rng.gen_range(0..regimes.len())];
        }
    }

    /// Execute trading action and return next state and PnL.
    ///
    /// # Arguments
    /// * `action` - Trading action (Hold/Buy/Sell)
    ///
    /// # Returns
    /// Tuple of (state_reference, pnl)
    ///
    /// # Invariants
    /// * Price evolves via geometric Brownian motion with regime-dependent drift/vol
    /// * Fees applied only on position changes
    /// * PnL = (price_change * position) - fees
    pub fn step(&mut self, action: Action) -> (&MarketState, f64) {
        self.state.t += 1;
        self.maybe_switch_regime();
        let old_price = self.state.price;
        let return_rate = self.compute_return();
        self.state.price *= return_rate.exp();
        let fee = self.apply_fee(action);
        let pnl = self.update_pnl(old_price, fee);
        (&self.state, pnl)
    }

    /// Compute price return from drift, volatility, and shocks.
    ///
    /// # Returns
    /// Log return (drift + noise + optional shock)
    /// Compute price return from drift, volatility, and shocks.
    ///
    /// # Returns
    /// Log return (drift + noise + optional shock)
    ///
    /// # Panics
    /// Panics if MarketParams were not validated (vol_base/vol_panic/shock_scale must be > 0)
    fn compute_return(&mut self) -> f64 {
        let drift = self.regime_drift();
        let vol = self.regime_vol();
        // Safety: vol is guaranteed > 0 if MarketParams.validate() was called
        // Using expect with clear message for debugging
        let mut noise = self.rng.sample::<f64, _>(
            rand_distr::Normal::new(0.0, vol)
                .expect("volatility must be positive (call MarketParams::validate())")
        );
        if self.rng.gen::<f64>() < self.params.shock_prob {
            noise += self.rng.sample::<f64, _>(
                rand_distr::Normal::new(0.0, self.params.shock_scale)
                    .expect("shock_scale must be positive (call MarketParams::validate())")
            );
        }
        drift + noise
    }

    /// Apply trading action and return fee.
    ///
    /// # Arguments
    /// * `action` - Trading action
    ///
    /// # Returns
    /// Fee paid (0.0 if no position change)
    fn apply_fee(&mut self, action: Action) -> f64 {
        match action {
            Action::Buy if self.state.position <= 0 => {
                self.state.position = 1;
                self.state.price * (self.params.fee_bps / 1e4)
            }
            Action::Sell if self.state.position >= 0 => {
                self.state.position = -1;
                self.state.price * (self.params.fee_bps / 1e4)
            }
            _ => 0.0,
        }
    }

    /// Update PnL and return current step PnL.
    ///
    /// # Arguments
    /// * `old_price` - Price before step
    /// * `fee` - Fee paid
    ///
    /// # Returns
    /// PnL for this step
    fn update_pnl(&mut self, old_price: f64, fee: f64) -> f64 {
        let pnl = (self.state.price - old_price) * (self.state.position as f64) - fee;
        self.state.pnl += pnl;
        pnl
    }
}

