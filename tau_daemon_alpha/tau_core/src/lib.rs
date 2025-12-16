pub mod config;
pub mod error;
pub mod model;

// Re-export key structs for easier access across the workspace.
pub use config::Config;
pub use error::DaemonError;
pub use model::*;

#[cfg(test)]
mod tests {
    use super::model::Fx;

    #[test]
    fn fx_checked_div_by_zero_returns_none() {
        let x = Fx::new(1);
        let z = Fx::new(0);
        assert!(x.checked_div(z).is_none());
    }

    #[test]
    fn fx_from_float_roundtrips_reasonably() {
        let x = Fx::from_float(1.25);
        let y = x.to_float();
        assert!((y - 1.25).abs() < 1e-6);
    }
}

