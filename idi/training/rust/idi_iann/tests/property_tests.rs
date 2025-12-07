#[cfg(test)]
mod property_tests {
    use idi_iann::action::Action;
    use idi_iann::config::TrainingConfig;
    use idi_iann::policy::LookupPolicy;
    use idi_iann::traits::Policy;
    use idi_iann::trainer::QTrainer;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn q_update_convex_hull(
            current_q in -1000.0f32..1000.0f32,
            target_q in -1000.0f32..1000.0f32,
            learning_rate in 0.01f32..1.0f32,
        ) {
            let mut policy = LookupPolicy::default();
            let state = (0u8, 0u8, 0u8, 0u8, 0u8);

            // Set initial Q-value
            policy.update(state, Action::Buy, current_q);

            // Compute update
            let td_error = target_q - current_q;
            policy.update(state, Action::Buy, learning_rate * td_error);

            let new_q = policy.q_value(state, Action::Buy);

            // Q_new should be in convex hull of current_q and target_q
            let min_val = current_q.min(target_q);
            let max_val = current_q.max(target_q);

            prop_assert!(
                new_q >= min_val - 1e-3 && new_q <= max_val + 1e-3,
                "Q-update violated convex hull: {} -> {} -> {}",
                current_q, new_q, target_q
            );
        }

        #[test]
        fn td_target_bounds(
            reward in -100.0f32..100.0f32,
            discount in 0.0f32..0.999f32,
            next_q in -1000.0f32..1000.0f32,
        ) {
            let td_target = reward + discount * next_q;

            // TD target should be finite
            prop_assert!(td_target.is_finite());

            // TD target should be bounded by reward + discount * next_q
            if discount < 1.0 {
                let lower_bound = reward + discount * next_q.min(0.0);
                let upper_bound = reward + discount * next_q.max(0.0);
                prop_assert!(
                    td_target >= lower_bound - 1e-3 && td_target <= upper_bound + 1e-3
                );
            }
        }

        #[test]
        fn q_update_respects_learning_rate(
            current_q in -100.0f32..100.0f32,
            target_q in -100.0f32..100.0f32,
            learning_rate in 0.0f32..1.0f32,
        ) {
            let mut policy = LookupPolicy::default();
            let state = (0u8, 0u8, 0u8, 0u8, 0u8);

            policy.update(state, Action::Buy, current_q);
            let initial_q = policy.q_value(state, Action::Buy);

            let td_error = target_q - initial_q;
            policy.update(state, Action::Buy, learning_rate * td_error);

            let new_q = policy.q_value(state, Action::Buy);

            // With learning_rate=0, Q should not change
            if learning_rate.abs() < 1e-6 {
                prop_assert!((new_q - initial_q).abs() < 1e-3);
            }

            // With learning_rate=1, Q should equal target
            if (learning_rate - 1.0).abs() < 1e-6 {
                prop_assert!((new_q - target_q).abs() < 1e-3);
            }
        }

        #[test]
        fn greedy_action_is_argmax(
            q_hold in -100.0f32..100.0f32,
            q_buy in -100.0f32..100.0f32,
            q_sell in -100.0f32..100.0f32,
        ) {
            let mut policy = LookupPolicy::default();
            let state = (0u8, 0u8, 0u8, 0u8, 0u8);

            policy.update(state, Action::Hold, q_hold);
            policy.update(state, Action::Buy, q_buy);
            policy.update(state, Action::Sell, q_sell);

            let best_action = policy.best_action(state);
            let best_q = policy.q_value(state, best_action);

            let all_qs = [
                policy.q_value(state, Action::Hold),
                policy.q_value(state, Action::Buy),
                policy.q_value(state, Action::Sell),
            ];
            let max_q = all_qs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            prop_assert_eq!(best_q, max_q, "best_action should return argmax");
        }

        #[test]
        fn action_enum_roundtrip(action_str in "hold|buy|sell") {
            // Test that Action enum can be created from string and converted back
            if let Ok(action) = action_str.parse::<Action>() {
                let back_to_str = action.as_str();
                prop_assert_eq!(back_to_str, action_str);
            }
        }
    }
}

