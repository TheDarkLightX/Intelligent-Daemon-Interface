use tau_core::model::{Action, KernelOutputs};
use tracing::{debug, info};

/// Handles the execution of actions based on kernel outputs.
#[derive(Debug)]
pub struct Actuator;

impl Actuator {
    /// Creates a new `Actuator`.
    pub fn new() -> Self {
        info!("Initializing actuator.");
        Self
    }

    /// Handles kernel outputs and executes appropriate actions.
    pub fn handle_outputs(&self, outputs: &KernelOutputs) -> Vec<Action> {
        debug!("Handling kernel outputs: {:?}", outputs);

        let mut actions = Vec::new();

        // Handle buy signal
        if outputs.buy {
            info!("Executing buy action");
            actions.push(Action {
                action_type: "buy".to_string(),
                id: format!("{}", uuid::Uuid::new_v4()),
                status: "submitted".to_string(),
            });
        }

        // Handle sell signal
        if outputs.sell {
            info!("Executing sell action");
            actions.push(Action {
                action_type: "sell".to_string(),
                id: format!("{}", uuid::Uuid::new_v4()),
                status: "submitted".to_string(),
            });
        }

        // Handle no action
        if !outputs.buy && !outputs.sell {
            debug!("No action required");
        }

        actions
    }
}
