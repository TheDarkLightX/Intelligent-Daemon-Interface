use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::trace::TraceTick;
use crate::traits::Policy;

pub type StateKey = (u8, u8, u8, u8, u8);

#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyEntry {
    pub q_values: HashMap<Action, f32>,
}

impl PolicyEntry {
    pub fn best_action(&self) -> Action {
        self.q_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(action, _)| *action)
            .unwrap_or_default()
    }
}

#[derive(Default)]
pub struct LookupPolicy {
    table: HashMap<StateKey, PolicyEntry>,
}

impl Policy for LookupPolicy {
    fn q_value(&mut self, state: StateKey, action: Action) -> f32 {
        self.table
            .entry(state)
            .or_insert_with(Self::default_entry)
            .q_values
            .get(&action)
            .copied()
            .unwrap_or(0.0)
    }

    fn update(&mut self, state: StateKey, action: Action, delta: f32) {
        let entry = self.table.entry(state).or_insert_with(Self::default_entry);
        let value = entry.q_values.entry(action).or_insert(0.0);
        *value += delta;
    }

    fn best_action(&mut self, state: StateKey) -> Action {
        self.table
            .entry(state)
            .or_insert_with(Self::default_entry)
            .best_action()
    }
}

impl LookupPolicy {
    fn default_entry() -> PolicyEntry {
        let mut map = HashMap::new();
        map.insert(Action::Hold, 0.0);
        map.insert(Action::Buy, 0.0);
        map.insert(Action::Sell, 0.0);
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
