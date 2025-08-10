use anyhow::Result;
use async_trait::async_trait;
use tau_core::{Config, model::*};
use tracing::{debug, warn};

pub struct ProfitGuardImpl {
    last_economics: Option<Economics>,
}

impl ProfitGuardImpl {
    pub fn new() -> Self {
        Self {
            last_economics: None,
        }
    }

    /// Compute fees for a trade
    fn compute_fees(&self, qty: Fx, price: Fx, fee_bps: u32) -> Result<Fx> {
        let fee_amount = qty
            .checked_mul(price)
            .ok_or_else(|| anyhow::anyhow!("Fee calculation overflow"))?
            .checked_mul(Fx::new(fee_bps as u128))
            .ok_or_else(|| anyhow::anyhow!("Fee calculation overflow"))?
            .checked_div(Fx::new(10_000))
            .ok_or_else(|| anyhow::anyhow!("Fee calculation overflow"))?;
        
        Ok(fee_amount)
    }

    /// Compute slippage in basis points
    fn compute_slippage_bps(&self, expected_price: Fx, actual_price: Fx) -> Result<u32> {
        if expected_price.raw() == 0 {
            return Ok(0);
        }

        let slippage = if actual_price > expected_price {
            actual_price.checked_sub(expected_price)
        } else {
            expected_price.checked_sub(actual_price)
        };

        let slippage_bps = slippage
            .ok_or_else(|| anyhow::anyhow!("Slippage calculation overflow"))?
            .checked_mul(Fx::new(10_000))
            .ok_or_else(|| anyhow::anyhow!("Slippage calculation overflow"))?
            .checked_div(expected_price)
            .ok_or_else(|| anyhow::anyhow!("Slippage calculation overflow"))?;

        Ok(slippage_bps.raw() as u32)
    }

    /// Compute realized PnL
    fn compute_pnl(&self, buy_price: Fx, sell_price: Fx, qty: Fx, config: &Config) -> Result<Economics> {
        let gross_buy = qty
            .checked_mul(buy_price)
            .ok_or_else(|| anyhow::anyhow!("Gross buy calculation overflow"))?;
        
        let gross_sell = qty
            .checked_mul(sell_price)
            .ok_or_else(|| anyhow::anyhow!("Gross sell calculation overflow"))?;

        let fee_buy = self.compute_fees(qty, buy_price, config.economics.fee_bps_buy)?;
        let fee_sell = self.compute_fees(qty, sell_price, config.economics.fee_bps_sell)?;
        
        let gas_cost = Fx::new(config.economics.gas_unit_cost.into());
        
        let slippage_bps = self.compute_slippage_bps(sell_price, sell_price)?; // Assuming no slippage for now
        
        let total_costs = fee_buy
            .checked_add(fee_sell)
            .ok_or_else(|| anyhow::anyhow!("Total costs calculation overflow"))?
            .checked_add(gas_cost)
            .ok_or_else(|| anyhow::anyhow!("Total costs calculation overflow"))?;

        let pnl = gross_sell
            .checked_sub(gross_buy)
            .ok_or_else(|| anyhow::anyhow!("PnL calculation overflow"))?
            .checked_sub(total_costs)
            .ok_or_else(|| anyhow::anyhow!("PnL calculation overflow"))?;

        let economics = Economics {
            buy_price,
            sell_price,
            qty,
            fee_buy,
            fee_sell,
            gas: gas_cost,
            slippage_bps,
            pnl,
        };

        Ok(economics)
    }
}

#[async_trait]
impl super::ProfitGuard for ProfitGuardImpl {
    async fn compute(&mut self, _venue_state: &VenueState, config: &Config, _tick: u64) -> Result<bool> {
        debug!("Computing profit guard...");

        // For now, use simple profit calculation
        // In a real implementation, this would use actual market data
        let buy_price = Fx::from_float(100.0);
        let sell_price = Fx::from_float(101.0);
        let qty = Fx::new(config.economics.trade_qty.into());

        let economics = self.compute_pnl(buy_price, sell_price, qty, config)?;
        
        // Store economics for diagnostics
        self.last_economics.replace(economics.clone());

        // Check if PnL is positive
        let profit_guard = economics.pnl.raw() > 0;

        // Check if slippage is within limits
        let slippage_ok = economics.slippage_bps <= config.economics.slippage_bps_limit;

        // Check if fees are settled (assume true for now)
        let fees_settled = true;

        // Check if execution price is acceptable (assume true for now)
        let execution_price_ok = true;

        let profit_guard = profit_guard && slippage_ok && fees_settled && execution_price_ok;

        debug!(
            "Profit guard: pnl={}, slippage_bps={}, profit_guard={}",
            economics.pnl, economics.slippage_bps, profit_guard
        );

        if !profit_guard {
            warn!(
                "Profit guard failed: pnl={}, slippage_bps={}, fees_settled={}, execution_price_ok={}",
                economics.pnl, economics.slippage_bps, fees_settled, execution_price_ok
            );
        }

        Ok(profit_guard)
    }

    async fn get_economics(&self) -> Result<Option<Economics>> {
        Ok(self.last_economics.clone())
    }
} 