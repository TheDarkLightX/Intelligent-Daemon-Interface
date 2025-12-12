use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    #[error("Invalid state: {0:?}")]
    InvalidState((u8, u8, u8, u8, u8)),

    #[error("Policy error: {0}")]
    PolicyError(String),

    #[error("Environment error: {0}")]
    EnvironmentError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

