use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizerConfig {
    pub price_buckets: u8,
    pub volume_buckets: u8,
    pub trend_buckets: u8,
    pub scarcity_buckets: u8,
    pub mood_buckets: u8,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            price_buckets: 4,
            volume_buckets: 4,
            trend_buckets: 4,
            scarcity_buckets: 8,
            mood_buckets: 4,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewardWeights {
    pub pnl: f32,
    pub scarcity_alignment: f32,
    pub ethics_bonus: f32,
    pub communication_clarity: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            pnl: 1.0,
            scarcity_alignment: 0.5,
            ethics_bonus: 0.75,
            communication_clarity: 0.2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub episodes: usize,
    pub episode_length: usize,
    pub discount: f32,
    pub learning_rate: f32,
    pub exploration_decay: f32,
    pub quantizer: QuantizerConfig,
    pub rewards: RewardWeights,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            episodes: 64,
            episode_length: 32,
            discount: 0.92,
            learning_rate: 0.2,
            exploration_decay: 0.995,
            quantizer: QuantizerConfig::default(),
            rewards: RewardWeights::default(),
        }
    }
}
