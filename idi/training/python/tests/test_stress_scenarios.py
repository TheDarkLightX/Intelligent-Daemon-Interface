"""Tests for stress test scenarios in crypto_env."""

import pytest

from idi_iann.crypto_env import CryptoMarket, MarketParams, STRESS_SCENARIOS


def test_normal_scenario():
    """Test normal scenario has no overrides."""
    params = MarketParams(stress_scenario="normal", seed=42)
    env = CryptoMarket(params)

    assert env.stress.name == "normal"
    assert env.stress.vol_multiplier == 1.0
    assert env.stress.forced_regime is None


def test_panic_scenario():
    """Test panic scenario configuration."""
    params = MarketParams(stress_scenario="panic", seed=42)
    env = CryptoMarket(params)

    assert env.stress.name == "panic"
    assert env.stress.vol_multiplier == 4.0
    assert env.stress.forced_regime == "panic"
    assert env.stress.drift_override == -0.01


def test_extreme_shock_scenario():
    """Test extreme shock scenario."""
    params = MarketParams(stress_scenario="extreme_shock", seed=42)
    env = CryptoMarket(params)

    assert env.stress.shock_prob_override == 0.5
    assert env.stress.vol_multiplier == 2.0


def test_fee_spike_scenario():
    """Test fee spike scenario."""
    params = MarketParams(stress_scenario="fee_spike", seed=42)
    env = CryptoMarket(params)

    # Fee should be multiplied
    normal_fee = params.fee_bps
    assert env._fee_bps() == normal_fee * 10.0


def test_panic_regime_is_forced():
    """Test that panic scenario forces panic regime."""
    params = MarketParams(stress_scenario="panic", seed=42)
    env = CryptoMarket(params)

    # Run several steps
    for _ in range(20):
        env.step("hold")

    # Regime should remain panic
    assert env.state.regime == "panic"


def test_normal_regime_can_change():
    """Test that normal scenario allows regime changes."""
    params = MarketParams(stress_scenario="normal", seed=42)
    env = CryptoMarket(params)

    regimes_seen = set()
    for _ in range(500):
        env.step("hold")
        regimes_seen.add(env.state.regime)

    # Should see multiple regimes in normal mode
    assert len(regimes_seen) >= 2


def test_stress_volatility_affects_returns():
    """Test that stress volatility multiplier affects return distribution."""
    normal_params = MarketParams(stress_scenario="normal", seed=42)
    stress_params = MarketParams(stress_scenario="panic", seed=42)

    normal_env = CryptoMarket(normal_params)
    stress_env = CryptoMarket(stress_params)

    normal_returns = []
    stress_returns = []

    for _ in range(100):
        normal_env.step("hold")
        stress_env.step("hold")
        normal_returns.append(normal_env.state.last_return)
        stress_returns.append(stress_env.state.last_return)

    # Stress returns should have higher variance
    normal_var = sum((r - sum(normal_returns)/len(normal_returns))**2 for r in normal_returns) / len(normal_returns)
    stress_var = sum((r - sum(stress_returns)/len(stress_returns))**2 for r in stress_returns) / len(stress_returns)

    assert stress_var > normal_var


def test_flash_crash_scenario():
    """Test flash crash scenario produces severe conditions."""
    params = MarketParams(stress_scenario="flash_crash", seed=42)
    env = CryptoMarket(params)

    assert env.stress.drift_override == -0.02
    assert env.stress.vol_multiplier == 5.0
    assert env.stress.shock_prob_override == 0.8


def test_bear_market_scenario():
    """Test bear market scenario."""
    params = MarketParams(stress_scenario="bear_market", seed=42)
    env = CryptoMarket(params)

    assert env.stress.drift_override == -0.005
    assert env.stress.forced_regime == "bear"


def test_all_scenarios_exist():
    """Test all expected scenarios are defined."""
    expected = ["normal", "panic", "extreme_shock", "fee_spike", "bear_market", "flash_crash"]

    for scenario in expected:
        assert scenario in STRESS_SCENARIOS


def test_scenario_guardrails():
    """Test that policies hit guardrails under stress."""
    params = MarketParams(stress_scenario="flash_crash", seed=42)
    env = CryptoMarket(params)

    # Track drawdowns
    max_price = env.state.price
    max_drawdown = 0.0

    for _ in range(100):
        env.step("hold")
        max_price = max(max_price, env.state.price)
        drawdown = (max_price - env.state.price) / max_price
        max_drawdown = max(max_drawdown, drawdown)

    # Flash crash should produce significant drawdown
    assert max_drawdown > 0.05  # At least 5% drawdown

