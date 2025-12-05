use idi_iann::{config::TrainingConfig, trainer::QTrainer};

#[test]
fn trainer_runs() {
    let mut trainer = QTrainer::new(TrainingConfig::default());
    let trace = trainer.run();
    assert_eq!(trace.len(), trainer.config.episode_length);
}
