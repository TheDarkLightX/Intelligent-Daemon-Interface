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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn reads_valid_snapshot_from_disk() {
        let dir = tempdir().unwrap();
        let snapshot_path = dir.path().join("market_snapshot.json");
        let payload = serde_json::json!({
            "bid_price": 100.5,
            "ask_price": 101.5,
            "timestamp": 1_700_000_000u64,
            "base_volume": 10.0,
            "quote_volume": 1015.0,
            "sources": [
                {"price": 100.5, "age_ms": 10u64, "within_tol": true},
                {"price": 101.5, "age_ms": 12u64, "within_tol": true}
            ]
        });
        fs::write(&snapshot_path, serde_json::to_string(&payload).unwrap()).unwrap();

        let oracle = Oracle::new(dir.path()).expect("oracle should initialize");
        let snapshot = oracle
            .get_market_snapshot()
            .await
            .expect("snapshot should parse");

        assert_eq!(snapshot.bid_price, 100.5);
        assert_eq!(snapshot.ask_price, 101.5);
        assert_eq!(snapshot.timestamp, 1_700_000_000u64);
        assert_eq!(snapshot.sources.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn surfaces_parse_errors() {
        let dir = tempdir().unwrap();
        let snapshot_path = dir.path().join("market_snapshot.json");
        fs::write(&snapshot_path, "not-json").unwrap();

        let oracle = Oracle::new(dir.path()).expect("oracle should initialize");
        let err = oracle
            .get_market_snapshot()
            .await
            .expect_err("invalid json should error");

        assert!(err.to_string().contains("Failed to parse snapshot"));
    }
}
