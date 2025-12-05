use crate::{
    config::TrainingConfig,
    emote::EmotionEngine,
    env::SyntheticMarketEnv,
    policy::{LookupPolicy, StateKey},
    trace::TraceTick,
};

pub struct QTrainer {
    pub config: TrainingConfig,
    env: SyntheticMarketEnv,
    policy: LookupPolicy,
    emotion: EmotionEngine,
}

impl QTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        let env = SyntheticMarketEnv::new(config.quantizer.clone(), config.rewards.clone(), 0);
        Self {
            emotion: EmotionEngine::new(2),
            config,
            env,
            policy: LookupPolicy::default(),
        }
    }

    pub fn run(&mut self) -> Vec<TraceTick> {
        for _ in 0..self.config.episodes {
            let mut obs = self.env.reset();
            for _ in 0..self.config.episode_length {
                let state = obs.as_state();
                let action = self.policy.best_action(state);
                let (next_obs, reward) = self.env.step(action);
                self.update_policy(state, action, reward, next_obs.as_state());
                obs = next_obs;
            }
        }
        self.rollout()
    }

    fn update_policy(
        &mut self,
        state: StateKey,
        action: &'static str,
        reward: f32,
        next_state: StateKey,
    ) {
        let best_next = self.policy.best_action(next_state);
        let td_target = reward + self.config.discount * self.policy.q_value(next_state, best_next);
        let current = self.policy.q_value(state, action);
        let td_error = td_target - current;
        self.policy
            .update(state, action, self.config.learning_rate * td_error);
    }

    fn rollout(&mut self) -> Vec<TraceTick> {
        let mut trace = Vec::new();
        let mut obs = self.env.reset();
        for _ in 0..self.config.episode_length {
            let state = obs.as_state();
            let action = self.policy.best_action(state);
            let emote_bits = self.emotion.render(state.4);
            trace.push(TraceTick {
                q_buy: u8::from(action == "buy"),
                q_sell: u8::from(action == "sell"),
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
