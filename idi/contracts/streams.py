"""Stream contract registry for training, spec generation, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StreamDef:
    name: str
    tau_type: str
    filename: str
    role: str  # input/output/mirror
    description: str = ""


_CONTRACTS: Dict[str, List[StreamDef]] = {
    "v38": [
        StreamDef("price", "sbf", "inputs/price.in", "input", "Normalized price bucket"),
        StreamDef("volume", "sbf", "inputs/volume.in", "input", "Volume bucket"),
        StreamDef("trend", "sbf", "inputs/trend.in", "input", "Trend bucket"),
        StreamDef("profit_guard", "sbf", "inputs/profit_guard.in", "input", "Profit guard"),
        StreamDef("failure_echo", "sbf", "inputs/failure_echo.in", "input", "Failure echo"),
        StreamDef("q_buy", "sbf", "inputs/q_buy.in", "input", "Buy signal"),
        StreamDef("q_sell", "sbf", "inputs/q_sell.in", "input", "Sell signal"),
        StreamDef("risk_budget_ok", "sbf", "inputs/risk_budget_ok.in", "input", "Risk budget ok"),
        StreamDef("q_regime", "bv[5]", "inputs/q_regime.in", "input", "Regime bits"),
        StreamDef("q_emote_positive", "sbf", "inputs/q_emote_positive.in", "input", "Emote positive"),
        StreamDef("q_emote_alert", "sbf", "inputs/q_emote_alert.in", "input", "Emote alert"),
    ],
}


def get_contract(name: str = "v38") -> List[StreamDef]:
    if name not in _CONTRACTS:
        raise ValueError(f"Unknown contract: {name}")
    return list(_CONTRACTS[name])
