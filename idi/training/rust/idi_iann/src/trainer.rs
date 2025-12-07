use crate::action::Action;
use crate::{
    config::TrainingConfig,
    emote::EmotionEngine,
    env::SyntheticMarketEnv,
    policy::LookupPolicy,
    traits::{Environment, Observation, Policy},
    trace::TraceTick,
};

/// Q-learning trainer for tabular RL with communication Q-table.
///
/// Trains a lookup-table policy using TD learning and generates
/// Tau-ready traces for agent specifications.
pub struct QTrainer {
    /// Training configuration (hyperparameters, quantizer, rewards)
    pub config: TrainingConfig,
    /// Environment simulator
    env: SyntheticMarketEnv,
    /// Trading action Q-table
    policy: LookupPolicy,
    /// Emotion engine for expressive outputs
    emotion: EmotionEngine,
}

impl QTrainer {
    /// Create a new trainer with default environment (synthetic market).
    ///
    /// # Arguments
    /// * `config` - Training configuration
    ///
    /// # Returns
    /// Configured trainer instance
    pub fn new(config: TrainingConfig) -> Self {
        let env = SyntheticMarketEnv::new(config.quantizer.clone(), config.rewards.clone(), 0);
        Self {
            emotion: EmotionEngine::new(2),
            config,
            env,
            policy: LookupPolicy::default(),
        }
    }

    pub fn env_clone(&self) -> SyntheticMarketEnv {
        self.env.clone()
    }

    pub fn run(&mut self) -> Vec<TraceTick> {
        (0..self.config.episodes).for_each(|_| {
            self.train_episode();
        });
        self.rollout()
    }

    fn train_episode(&mut self) {
        let mut obs = self.env.reset();
        (0..self.config.episode_length).for_each(|_| {
            let state = obs.as_state();
            let action = self.policy.best_action(state);
            let (next_obs, reward) = self.env.step(action);
            let next_state = next_obs.as_state();
            self.update_policy(state, action, reward, next_state);
            obs = next_obs;
        });
    }

    fn compute_td_target(
        &mut self,
        reward: f32,
        next_state: (u8, u8, u8, u8, u8),
    ) -> f32 {
        let best_next = self.policy.best_action(next_state);
        reward + self.config.discount * self.policy.q_value(next_state, best_next)
    }

    fn update_policy(
        &mut self,
        state: (u8, u8, u8, u8, u8),
        action: Action,
        reward: f32,
        next_state: (u8, u8, u8, u8, u8),
    ) {
        let td_target = self.compute_td_target(reward, next_state);
        let current = self.policy.q_value(state, action);
        let td_error = td_target - current;
        self.policy
            .update(state, action, self.config.learning_rate * td_error);
    }

    /// Generate greedy rollout trace for Tau spec inputs.
    ///
    /// Uses learned policy (no exploration) to generate deterministic
    /// action sequences suitable for Tau spec execution.
    ///
    /// # Returns
    /// Vector of trace ticks with Q-actions and emotive cues
    fn rollout(&mut self) -> Vec<TraceTick> {
        let mut trace = Vec::with_capacity(self.config.episode_length);
        let mut obs = self.env.reset();
        for _ in 0..self.config.episode_length {
            let state = obs.as_state();
            let action = self.policy.best_action(state);
            let emote_bits = self.emotion.render(state.4);
            trace.push(TraceTick {
                q_buy: u8::from(action == Action::Buy),
                q_sell: u8::from(action == Action::Sell),
                risk_budget_ok: 1,
                q_emote_positive: emote_bits.positive,
                q_emote_alert: emote_bits.alert,
                q_regime: state.3,
            });
            let (next_obs, _) = self.env.step(action);
            obs = next_obs;
        }
        trace
    }
}
