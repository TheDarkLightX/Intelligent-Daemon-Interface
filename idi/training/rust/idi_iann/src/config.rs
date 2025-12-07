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
pub struct EmoteConfig {
    #[serde(default = "default_palette")]
    pub palette: std::collections::HashMap<u8, String>,
    #[serde(default = "default_linger_ticks")]
    pub linger_ticks: u8,
}

fn default_palette() -> std::collections::HashMap<u8, String> {
    let mut map = std::collections::HashMap::new();
    map.insert(0, "🙂 steady".to_string());
    map.insert(1, "🚀 optimistic".to_string());
    map.insert(2, "😐 cautious".to_string());
    map.insert(3, "⚠️ alert".to_string());
    map
}

fn default_linger_ticks() -> u8 {
    2
}

impl Default for EmoteConfig {
    fn default() -> Self {
        Self {
            palette: default_palette(),
            linger_ticks: default_linger_ticks(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileCoderConfig {
    pub num_tilings: u8,
    #[serde(default = "default_tile_sizes")]
    pub tile_sizes: Vec<u8>,
    #[serde(default = "default_offsets")]
    pub offsets: Vec<u8>,
}

fn default_tile_sizes() -> Vec<u8> {
    vec![4, 4, 4, 4, 4]
}

fn default_offsets() -> Vec<u8> {
    vec![0, 1, 2, 3, 4]
}

impl Default for TileCoderConfig {
    fn default() -> Self {
        Self {
            num_tilings: 3,
            tile_sizes: default_tile_sizes(),
            offsets: default_offsets(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommunicationConfig {
    #[serde(default = "default_comm_actions")]
    pub actions: Vec<String>,
}

fn default_comm_actions() -> Vec<String> {
    vec!["silent".to_string(), "positive".to_string(), "alert".to_string(), "persist".to_string()]
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            actions: default_comm_actions(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    #[serde(default = "default_true")]
    pub emit_weight_streams: bool,
    #[serde(default = "default_momentum_threshold")]
    pub momentum_threshold: f32,
    #[serde(default = "default_contrarian_threshold")]
    pub contrarian_threshold: f32,
    #[serde(default = "default_true")]
    pub trend_favors_even: bool,
}

fn default_true() -> bool {
    true
}

fn default_momentum_threshold() -> f32 {
    0.6
}

fn default_contrarian_threshold() -> f32 {
    0.3
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            emit_weight_streams: true,
            momentum_threshold: 0.6,
            contrarian_threshold: 0.3,
            trend_favors_even: true,
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
    #[serde(default)]
    pub emote: EmoteConfig,
    #[serde(default)]
    pub layers: LayerConfig,
    #[serde(default)]
    pub tile_coder: Option<TileCoderConfig>,
    #[serde(default)]
    pub communication: CommunicationConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            episodes: 128,
            episode_length: 64,
            discount: 0.92,
            learning_rate: 0.2,
            exploration_decay: 0.995,
            quantizer: QuantizerConfig::default(),
            rewards: RewardWeights::default(),
            emote: EmoteConfig::default(),
            layers: LayerConfig::default(),
            tile_coder: None,
            communication: CommunicationConfig::default(),
        }
    }
}
