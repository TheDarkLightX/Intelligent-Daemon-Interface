use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct TraceTick {
    pub q_buy: u8,
    pub q_sell: u8,
    pub risk_budget_ok: u8,
    pub q_emote_positive: u8,
    pub q_emote_alert: u8,
    pub q_regime: u8,
}

impl TraceTick {
    pub fn new(q_buy: u8, q_sell: u8, q_regime: u8) -> Self {
        Self {
            q_buy,
            q_sell,
            risk_budget_ok: 1,
            q_emote_positive: q_buy,
            q_emote_alert: q_sell,
            q_regime,
        }
    }
}
