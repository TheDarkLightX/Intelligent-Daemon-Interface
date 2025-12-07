use rand::{Rng, SeedableRng};

use crate::action::Action;
use crate::config::{QuantizerConfig, RewardWeights};
use crate::traits::{Environment, Observation};

#[derive(Clone, Copy)]
pub struct EnvObservation {
    pub price: u8,
    pub volume: u8,
    pub trend: u8,
    pub scarcity: u8,
    pub mood: u8,
}

impl Observation for EnvObservation {
    fn as_state(&self) -> (u8, u8, u8, u8, u8) {
        (
            self.price,
            self.volume,
            self.trend,
            self.scarcity,
            self.mood,
        )
    }
}

#[derive(Clone)]
pub struct SyntheticMarketEnv {
    quant: QuantizerConfig,
    rewards: RewardWeights,
    rng: rand::rngs::StdRng,
    position: i8,
}

impl SyntheticMarketEnv {
    pub const ACTIONS: [Action; 3] = [Action::Hold, Action::Buy, Action::Sell];

    pub fn new(quant: QuantizerConfig, rewards: RewardWeights, seed: u64) -> Self {
        Self {
            quant,
            rewards,
            rng: SeedableRng::seed_from_u64(seed),
            position: 0,
        }
    }

    fn observe(&mut self) -> EnvObservation {
        EnvObservation {
            price: self.rng.gen_range(0..self.quant.price_buckets),
            volume: self.rng.gen_range(0..self.quant.volume_buckets),
            trend: self.rng.gen_range(0..self.quant.trend_buckets),
            scarcity: self.rng.gen_range(0..self.quant.scarcity_buckets),
            mood: self.rng.gen_range(0..self.quant.mood_buckets),
        }
    }

    fn update_position(&mut self, action: Action) {
        match action {
            Action::Buy => self.position = (self.position + 1).clamp(-1, 1),
            Action::Sell => self.position = (self.position - 1).clamp(-1, 1),
            Action::Hold => {}
        }
    }

    fn calculate_reward(&self, price_drift: f32, scarcity_drift: f32) -> f32 {
        let pnl = price_drift * self.position as f32 * self.rewards.pnl;
        let scarcity_bonus = scarcity_drift * self.rewards.scarcity_alignment;
        let ethics_bonus = if self.position >= 0 {
            self.rewards.ethics_bonus
        } else {
            -self.rewards.ethics_bonus
        };
        pnl + scarcity_bonus + ethics_bonus
    }
}

impl Environment for SyntheticMarketEnv {
    type Obs = EnvObservation;

    /// Reset environment to initial state.
    ///
    /// # Returns
    /// Initial observation with position = 0
    fn reset(&mut self) -> Self::Obs {
        self.position = 0;
        self.observe()
    }

    /// Execute action and return next observation and reward.
    ///
    /// # Arguments
    /// * `action` - Trading action (Hold/Buy/Sell)
    ///
    /// # Returns
    /// Tuple of (next_observation, reward)
    ///
    /// # Invariants
    /// * Position is clamped to [-1, 1]
    /// * Reward components: PnL + scarcity + ethics
    fn step(&mut self, action: Action) -> (Self::Obs, f32) {
        self.update_position(action);
        let price_drift: f32 = self.rng.gen_range(-1.0..1.0);
        let scarcity_drift: f32 = self.rng.gen_range(-0.5..0.5);
        let reward = self.calculate_reward(price_drift, scarcity_drift);
        (self.observe(), reward)
    }
}
