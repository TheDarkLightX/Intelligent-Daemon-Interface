use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::trace::TraceTick;

pub type StateKey = (u8, u8, u8, u8, u8);

#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyEntry {
    pub q_values: HashMap<String, f32>,
}

impl PolicyEntry {
    pub fn best_action(&self) -> String {
        self.q_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(action, _)| action.clone())
            .unwrap_or_else(|| "hold".to_string())
    }
}

#[derive(Default)]
pub struct LookupPolicy {
    table: HashMap<StateKey, PolicyEntry>,
}

impl LookupPolicy {
    pub fn q_value(&mut self, state: StateKey, action: &'static str) -> f32 {
        self.table
            .entry(state)
            .or_insert_with(Self::default_entry)
            .q_values
            .get(action)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn update(&mut self, state: StateKey, action: &'static str, delta: f32) {
        let entry = self.table.entry(state).or_insert_with(Self::default_entry);
        let value = entry.q_values.entry(action.to_string()).or_insert(0.0);
        *value += delta;
    }

    pub fn best_action(&mut self, state: StateKey) -> &'static str {
        let action = self
            .table
            .entry(state)
            .or_insert_with(Self::default_entry)
            .best_action();
        match action.as_str() {
            "buy" => "buy",
            "sell" => "sell",
            _ => "hold",
        }
    }

    fn default_entry() -> PolicyEntry {
        let mut map = HashMap::new();
        for action in ["hold", "buy", "sell"] {
            map.insert(action.to_string(), 0.0);
        }
        PolicyEntry { q_values: map }
    }

    pub fn export_trace(&self, trace: &[TraceTick]) -> Vec<String> {
        let mut output = Vec::new();
        for tick in trace {
            output.push(format!(
                "{},{},{},{},{},{}",
                tick.q_buy,
                tick.q_sell,
                tick.risk_budget_ok,
                tick.q_emote_positive,
                tick.q_emote_alert,
                tick.q_regime
            ));
        }
        output
    }
}
