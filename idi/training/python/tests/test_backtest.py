import json
import pandas as pd
from pathlib import Path

from idi_iann.config import TrainingConfig
from idi_iann.policy import LookupPolicy
from backtest import backtest, _compute_kpis, _calibrate_stats


def test_backtest_runs_with_minimal_data(tmp_path: Path) -> None:
    cfg = TrainingConfig(episodes=1, episode_length=2)
    policy = LookupPolicy()
    df = pd.DataFrame({"price": [100.0, 101.0, 102.0]})
    kpis, actions = backtest(cfg, policy, df, price_col="price", volume_col=None, fee_bps=5.0)
    assert "mean" in kpis
    assert len(actions) == len(df)


def test_kpis_nonempty() -> None:
    kpis = _compute_kpis([0.1, -0.2, 0.3])
    assert kpis["mean"] != 0.0


def test_calibration() -> None:
    df = pd.DataFrame({"price": [100, 101, 99, 102]})
    stats = _calibrate_stats(df["price"])
    assert "drift" in stats and "vol" in stats


