use idi_iann::{
    action::Action,
    config::TrainingConfig,
    env::SyntheticMarketEnv,
    policy::LookupPolicy,
    traits::{Environment, Observation, Policy},
    trainer::QTrainer,
};

#[test]
fn trainer_produces_trace() {
    let mut trainer = QTrainer::new(TrainingConfig::default());
    let trace = trainer.run();
    assert!(!trace.is_empty());
    assert_eq!(trace.len(), trainer.config.episode_length);
}

#[test]
fn env_reset_works() {
    let mut env = SyntheticMarketEnv::new(
        TrainingConfig::default().quantizer,
        TrainingConfig::default().rewards,
        42,
    );
    let obs = env.reset();
    let state = obs.as_state();
    // State should be valid (all values within expected ranges)
    assert!(state.0 < 4); // price_buckets
    assert!(state.1 < 4); // volume_buckets
    assert!(state.2 < 4); // trend_buckets
    assert!(state.3 < 8); // scarcity_buckets
    assert!(state.4 < 4); // mood_buckets
}

#[test]
fn policy_updates_q_values() {
    let mut policy = LookupPolicy::default();
    let state = (1, 2, 3, 4, 5);
    let action = Action::Buy;

    let initial_q = policy.q_value(state, action);
    policy.update(state, action, 0.5);
    let updated_q = policy.q_value(state, action);

    assert!((updated_q - initial_q - 0.5).abs() < f32::EPSILON);
}

#[test]
fn policy_best_action_returns_valid_action() {
    let mut policy = LookupPolicy::default();
    let state = (1, 2, 3, 4, 5);
    let best = policy.best_action(state);
    // Should return one of the valid actions
    assert!(matches!(best, Action::Hold | Action::Buy | Action::Sell));
}

