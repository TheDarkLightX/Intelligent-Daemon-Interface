use rand::{Rng, SeedableRng};

use crate::config::{QuantizerConfig, RewardWeights};

#[derive(Clone, Copy)]
pub struct Observation {
    pub price: u8,
    pub volume: u8,
    pub trend: u8,
    pub scarcity: u8,
    pub mood: u8,
}

impl Observation {
    pub fn as_state(&self) -> (u8, u8, u8, u8, u8) {
        (
            self.price,
            self.volume,
            self.trend,
            self.scarcity,
            self.mood,
        )
    }
}

pub struct SyntheticMarketEnv {
    quant: QuantizerConfig,
    rewards: RewardWeights,
    rng: rand::rngs::StdRng,
    position: i8,
}

impl SyntheticMarketEnv {
    pub const ACTIONS: [&'static str; 3] = ["hold", "buy", "sell"];

    pub fn new(quant: QuantizerConfig, rewards: RewardWeights, seed: u64) -> Self {
        Self {
            quant,
            rewards,
            rng: SeedableRng::seed_from_u64(seed),
            position: 0,
        }
    }

    pub fn reset(&mut self) -> Observation {
        self.position = 0;
        self.observe()
    }

    pub fn step(&mut self, action: &str) -> (Observation, f32) {
        match action {
            "buy" => self.position = (self.position + 1).clamp(-1, 1),
            "sell" => self.position = (self.position - 1).clamp(-1, 1),
            _ => {}
        }
        let price_drift: f32 = self.rng.gen_range(-1.0..1.0);
        let scarcity_drift: f32 = self.rng.gen_range(-0.5..0.5);
        let pnl = price_drift * self.position as f32 * self.rewards.pnl;
        let scarcity_bonus = scarcity_drift * self.rewards.scarcity_alignment;
        let ethics_bonus = if self.position >= 0 {
            self.rewards.ethics_bonus
        } else {
            -self.rewards.ethics_bonus
        };
        let reward = pnl + scarcity_bonus + ethics_bonus;
        (self.observe(), reward)
    }

    fn observe(&mut self) -> Observation {
        Observation {
            price: self.rng.gen_range(0..self.quant.price_buckets),
            volume: self.rng.gen_range(0..self.quant.volume_buckets),
            trend: self.rng.gen_range(0..self.quant.trend_buckets),
            scarcity: self.rng.gen_range(0..self.quant.scarcity_buckets),
            mood: self.rng.gen_range(0..self.quant.mood_buckets),
        }
    }
}
