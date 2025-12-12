use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use tau_core::model::MarketSnapshot;
use tracing::{debug, warn};

/// Manages market data collection from various sources.
#[derive(Debug)]
pub struct Oracle {
    snapshot_path: PathBuf,
}

impl Oracle {
    /// Creates a new `Oracle`.
    pub fn new(data_dir: &Path) -> Result<Self> {
        debug!("Initializing oracle.");
        fs::create_dir_all(data_dir)?;
        Ok(Self {
            snapshot_path: data_dir.join("market_snapshot.json"),
        })
    }

    /// Fetches a snapshot of current market data.
    pub async fn get_market_snapshot(&self) -> Result<MarketSnapshot> {
        debug!("Fetching market snapshot...");

        let contents = fs::read_to_string(&self.snapshot_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read snapshot file {:?}: {}",
                self.snapshot_path,
                e
            )
        })?;

        let snapshot: MarketSnapshot = serde_json::from_str(&contents).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse snapshot from {:?}: {}",
                self.snapshot_path,
                e
            )
        })?;

        if snapshot.bid_price <= 0.0 || snapshot.ask_price <= 0.0 {
            warn!(
                "Snapshot prices are non-positive: bid={} ask={}",
                snapshot.bid_price, snapshot.ask_price
            );
        }

        debug!(?snapshot, "Market snapshot fetched.");
        Ok(snapshot)
    }
}
