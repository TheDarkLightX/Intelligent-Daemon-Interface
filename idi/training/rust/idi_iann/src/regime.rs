use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    #[default]
    Chop,
    Panic,
}

impl Regime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Bull => "bull",
            Regime::Bear => "bear",
            Regime::Chop => "chop",
            Regime::Panic => "panic",
        }
    }
}

impl FromStr for Regime {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bull" => Ok(Regime::Bull),
            "bear" => Ok(Regime::Bear),
            "chop" => Ok(Regime::Chop),
            "panic" => Ok(Regime::Panic),
            _ => Err(format!("Invalid regime: {}", s)),
        }
    }
}

