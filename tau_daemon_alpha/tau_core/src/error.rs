use thiserror::Error;

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network request error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Solver error: {0}")]
    Solver(String),

    #[error("Kernel invariant violated: {0}")]
    Invariant(String),

    #[error("Action execution error: {0}")]
    Action(String),
}
