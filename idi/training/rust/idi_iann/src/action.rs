use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    #[default]
    Hold,
    Buy,
    Sell,
}

impl Action {
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Hold => "hold",
            Action::Buy => "buy",
            Action::Sell => "sell",
        }
    }
}

impl FromStr for Action {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hold" => Ok(Action::Hold),
            "buy" => Ok(Action::Buy),
            "sell" => Ok(Action::Sell),
            _ => Err(format!("Invalid action: {}", s)),
        }
    }
}

