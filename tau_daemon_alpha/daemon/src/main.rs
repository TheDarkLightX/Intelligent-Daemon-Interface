use anyhow::Result;
use std::path::PathBuf;
use tau_core::Config;
use tracing::{error, info};

mod actuator;
mod execution;
mod fsio;
mod guards;
mod kernel;
mod ledger;
mod looper;
mod monitors;
mod oracle;
mod state;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing subscriber for structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Initializing Enhanced Tau Daemon with Supporting Specifications...");

    // Define the path to the configuration file, assuming it's in the project root.
    let config_path = PathBuf::from("tau_daemon.toml");

    // Load configuration from the file.
    let config = match Config::from_file(&config_path) {
        Ok(cfg) => {
            info!(
                "Configuration loaded successfully from {:?}",
                config_path.canonicalize()?
            );
            cfg
        }
        Err(e) => {
            error!(
                "Failed to load configuration from {:?}: {}",
                config_path.display(),
                e
            );
            return Err(e.into());
        }
    };

    info!(config = ?config, "Enhanced daemon configured. Starting main loop...");

    // Start the main application loop.
    if let Err(e) = looper::run(config).await {
        error!("The enhanced daemon encountered a fatal error: {}", e);
    }

    Ok(())
}
