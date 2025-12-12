from idi_iann.crypto_env import CryptoMarket, MarketParams


def test_crypto_market_risk_event_emerges() -> None:
    market = CryptoMarket(MarketParams(seed=1, shock_prob=0.5, shock_scale=0.1))
    obs = market.reset()
    assert obs.regime == "chop"
    triggered = False
    for _ in range(20):
        obs, reward, info = market.step("hold")
        assert reward is not None
        if info.get("risk_event"):
            triggered = True
            break
    assert triggered

