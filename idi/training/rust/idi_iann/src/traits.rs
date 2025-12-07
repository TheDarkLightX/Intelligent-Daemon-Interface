use crate::action::Action;

pub trait Observation {
    fn as_state(&self) -> (u8, u8, u8, u8, u8);
}

pub trait Environment {
    type Obs: Observation;

    fn reset(&mut self) -> Self::Obs;
    fn step(&mut self, action: Action) -> (Self::Obs, f32);
}

pub trait Policy {
    fn q_value(&mut self, state: (u8, u8, u8, u8, u8), action: Action) -> f32;
    fn update(&mut self, state: (u8, u8, u8, u8, u8), action: Action, delta: f32);
    fn best_action(&mut self, state: (u8, u8, u8, u8, u8)) -> Action;
}

pub trait RewardCalculator {
    fn calculate(
        &self,
        pnl: f32,
        scarcity_drift: f32,
        position: i8,
        rewards: &crate::config::RewardWeights,
    ) -> f32;
}

