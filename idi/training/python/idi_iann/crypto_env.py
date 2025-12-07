"""Crypto-style market simulator for Q-table training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class StressScenario:
    """Predefined stress test scenario."""

    name: str
    drift_override: Optional[float] = None
    vol_multiplier: float = 1.0
    shock_prob_override: Optional[float] = None
    fee_multiplier: float = 1.0
    forced_regime: Optional[str] = None


# Predefined stress scenarios
STRESS_SCENARIOS = {
    "normal": StressScenario(name="normal"),
    "panic": StressScenario(
        name="panic",
        drift_override=-0.01,
        vol_multiplier=4.0,
        shock_prob_override=0.2,
        forced_regime="panic",
    ),
    "extreme_shock": StressScenario(
        name="extreme_shock",
        vol_multiplier=2.0,
        shock_prob_override=0.5,
    ),
    "fee_spike": StressScenario(
        name="fee_spike",
        fee_multiplier=10.0,
    ),
    "bear_market": StressScenario(
        name="bear_market",
        drift_override=-0.005,
        vol_multiplier=1.5,
        forced_regime="bear",
    ),
    "flash_crash": StressScenario(
        name="flash_crash",
        drift_override=-0.02,
        vol_multiplier=5.0,
        shock_prob_override=0.8,
        forced_regime="panic",
    ),
}


@dataclass
class MarketParams:
    """Configurable market dynamics parameters."""

    regimes: Tuple[str, ...] = ("bull", "bear", "chop", "panic")
    drift_bull: float = 0.002
    drift_bear: float = -0.002
    drift_chop: float = 0.0
    vol_base: float = 0.01
    vol_panic: float = 0.04
    shock_prob: float = 0.02
    shock_scale: float = 0.05
    fee_bps: float = 5.0
    seed: int = 7
    stress_scenario: str = "normal"  # normal, panic, extreme_shock, fee_spike, etc.


@dataclass
class MarketState:
    """Current state of the simulated market."""

    price: float = 1000.0
    regime: str = "chop"
    t: int = 0
    position: int = 0  # -1, 0, 1
    inventory_value: float = 0.0
    pnl: float = 0.0
    last_return: float = 0.0
    risk_event: bool = False


class CryptoMarket:
    """Simple stylized crypto market with regime switches and fat-tail shocks.

    Supports stress test scenarios for robustness testing.
    """

    ACTIONS = ("hold", "buy", "sell")

    def __init__(self, params: MarketParams):
        """Initialize market with parameters.

        Args:
            params: Market parameters including optional stress scenario
        """
        self.p = params
        self.rng = random.Random(params.seed)
        self.state = MarketState()
        self.stress = STRESS_SCENARIOS.get(params.stress_scenario, STRESS_SCENARIOS["normal"])

    def reset(self) -> MarketState:
        """Reset market to initial state."""
        self.state = MarketState()
        return self.state

    def _regime_drift(self) -> float:
        """Get drift for current regime, with stress override."""
        if self.stress.drift_override is not None:
            return self.stress.drift_override

        r = self.state.regime
        if r == "bull":
            return self.p.drift_bull
        if r == "bear":
            return self.p.drift_bear
        if r == "panic":
            return -abs(self.p.drift_bear) * 1.5
        return self.p.drift_chop

    def _regime_vol(self) -> float:
        """Get volatility for current regime, with stress multiplier."""
        base_vol = self.p.vol_panic if self.state.regime == "panic" else self.p.vol_base
        return base_vol * self.stress.vol_multiplier

    def _shock_prob(self) -> float:
        """Get shock probability, with stress override."""
        if self.stress.shock_prob_override is not None:
            return self.stress.shock_prob_override
        return self.p.shock_prob

    def _fee_bps(self) -> float:
        """Get fee in basis points, with stress multiplier."""
        return self.p.fee_bps * self.stress.fee_multiplier

    def _maybe_switch_regime(self) -> None:
        """Potentially switch regime, respecting stress forced regime."""
        if self.stress.forced_regime is not None:
            self.state.regime = self.stress.forced_regime
            return

        # Simple Markov switch with bias toward staying
        if self.rng.random() < 0.05:
            self.state.regime = self.rng.choice(self.p.regimes)

    def step(self, action: str) -> Tuple[MarketState, float, Dict[str, float]]:
        """Execute action and return next state, reward, and info."""
        self.state.t += 1
        self._maybe_switch_regime()
        ret = self._compute_return()
        old_price = self.state.price
        self.state.price *= math.exp(ret)
        fee = self._apply_action(action)
        pnl = self._update_pnl(old_price, fee)
        info = self._build_info(ret, fee)
        return self.state, pnl, info

    def _compute_return(self) -> float:
        """Compute price return from drift, volatility, and shocks."""
        drift = self._regime_drift()
        vol = self._regime_vol()
        noise = self.rng.gauss(0, vol)
        if self.rng.random() < self._shock_prob():
            noise += self.rng.gauss(0, self.p.shock_scale * self.stress.vol_multiplier)
        return drift + noise

    def _apply_action(self, action: str) -> float:
        """Apply trading action and return fee."""
        fee_bps = self._fee_bps()
        if action == "buy" and self.state.position <= 0:
            fee = self.state.price * (fee_bps / 1e4)
            self.state.position = 1
            return fee
        if action == "sell" and self.state.position >= 0:
            fee = self.state.price * (fee_bps / 1e4)
            self.state.position = -1
            return fee
        return 0.0

    def _update_pnl(self, old_price: float, fee: float) -> float:
        """Update PnL and return current step PnL."""
        pnl = (self.state.price - old_price) * self.state.position - fee
        self.state.pnl += pnl
        self.state.inventory_value = self.state.position * self.state.price
        vol = self._regime_vol()
        self.state.last_return = (self.state.price - old_price) / old_price if old_price > 0 else 0.0
        self.state.risk_event = bool(abs(self.state.last_return) > vol * 2.5 or self.state.regime == "panic")
        return pnl

    def _build_info(self, ret: float, fee: float) -> Dict[str, float]:
        """Build info dictionary for step output."""
        vol = self._regime_vol()
        noise = ret - self._regime_drift()
        return {
            "ret": ret,
            "drift": self._regime_drift(),
            "vol": vol,
            "shock": float(abs(noise) > vol * 3),
            "regime": self.state.regime,
            "fee": fee,
            "risk_event": float(self.state.risk_event),
        }
