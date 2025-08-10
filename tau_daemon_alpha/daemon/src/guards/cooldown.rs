use anyhow::Result;
use async_trait::async_trait;
use tau_core::{Config, model::*};
use tracing::debug;

pub struct CooldownGuardImpl;

impl CooldownGuardImpl {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl super::CooldownGuard for CooldownGuardImpl {
    async fn compute(&self, last_trade_tick: Option<u64>, current_tick: u64, config: &Config) -> Result<bool> {
        debug!("Computing cooldown guard...");

        if !config.cooldown.enabled {
            debug!("Cooldown disabled, allowing trade");
            return Ok(true);
        }

        let cooldown_ok = match last_trade_tick {
            Some(last_tick) => {
                let ticks_since_trade = current_tick.saturating_sub(last_tick);
                let ok = ticks_since_trade >= config.cooldown.ticks.into();
                
                debug!(
                    "Cooldown check: last_trade_tick={}, current_tick={}, ticks_since_trade={}, cooldown_ticks={}, cooldown_ok={}",
                    last_tick, current_tick, ticks_since_trade, config.cooldown.ticks, ok
                );
                
                ok
            }
            None => {
                debug!("No previous trade found, cooldown ok");
                true
            }
        };

        Ok(cooldown_ok)
    }

    async fn get_ticks_since_trade(&self) -> Option<u32> {
        // This would typically return cached value from last compute
        // For now, return None
        None
    }
} 