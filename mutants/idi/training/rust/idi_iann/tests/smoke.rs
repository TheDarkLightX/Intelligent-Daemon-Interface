use idi_iann::{action::Action, config::TrainingConfig, trainer::QTrainer, traits::Environment};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[test]
fn trainer_runs() {
    let mut trainer = QTrainer::new(TrainingConfig::default());
    let trace = trainer.run();
    assert_eq!(trace.len(), trainer.config.episode_length);
}

#[test]
fn env_step_produces_reward() {
    let trainer = QTrainer::new(TrainingConfig::default());
    let mut env = trainer.env_clone();
    let (_, reward) = env.step(Action::Hold);
    // reward can be negative but should be finite
    assert!(reward.is_finite());
}

#[derive(Deserialize)]
struct Schema {
    episodes: usize,
    episode_length: usize,
    discount: f32,
    learning_rate: f32,
    exploration_decay: f32,
    quantizer: QuantizerSchema,
    rewards: RewardsSchema,
}

#[derive(Deserialize)]
struct QuantizerSchema {
    price_buckets: u8,
    volume_buckets: u8,
    trend_buckets: u8,
    scarcity_buckets: u8,
    mood_buckets: u8,
}

#[derive(Deserialize)]
struct RewardsSchema {
    pnl: f32,
    scarcity_alignment: f32,
    ethics_bonus: f32,
    communication_clarity: f32,
}

#[test]
fn schema_matches_defaults() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../..");
    // Try config_defaults.json first, fallback to config_schema.json
    let defaults_path = path.join("config_defaults.json");
    let schema_path = path.join("config_schema.json");
    
    let data = if defaults_path.exists() {
        fs::read_to_string(defaults_path).expect("defaults file missing")
    } else {
        fs::read_to_string(schema_path).expect("schema file missing")
    };
    let schema: Schema = serde_json::from_str(&data).expect("invalid config json");
    let cfg = TrainingConfig::default();
    assert_eq!(cfg.episodes, schema.episodes);
    assert_eq!(cfg.episode_length, schema.episode_length);
    assert!((cfg.discount - schema.discount).abs() < f32::EPSILON);
    assert!((cfg.learning_rate - schema.learning_rate).abs() < f32::EPSILON);
    assert!((cfg.exploration_decay - schema.exploration_decay).abs() < f32::EPSILON);
    assert_eq!(cfg.quantizer.price_buckets, schema.quantizer.price_buckets);
    assert_eq!(cfg.quantizer.volume_buckets, schema.quantizer.volume_buckets);
    assert_eq!(cfg.quantizer.trend_buckets, schema.quantizer.trend_buckets);
    assert_eq!(cfg.quantizer.scarcity_buckets, schema.quantizer.scarcity_buckets);
    assert_eq!(cfg.quantizer.mood_buckets, schema.quantizer.mood_buckets);
    assert!((cfg.rewards.pnl - schema.rewards.pnl).abs() < f32::EPSILON);
    assert!((cfg.rewards.scarcity_alignment - schema.rewards.scarcity_alignment).abs() < f32::EPSILON);
    assert!((cfg.rewards.ethics_bonus - schema.rewards.ethics_bonus).abs() < f32::EPSILON);
    assert!((cfg.rewards.communication_clarity - schema.rewards.communication_clarity).abs() < f32::EPSILON);
}
