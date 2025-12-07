import json
from pathlib import Path

from idi_iann.config import TrainingConfig


def test_defaults_match_schema() -> None:
    """Test that Python defaults match the default config JSON."""
    defaults_path = Path(__file__).resolve().parents[2] / "config_defaults.json"
    if not defaults_path.exists():
        pytest.skip("config_defaults.json not found")
    data = json.loads(defaults_path.read_text())
    cfg = TrainingConfig()

    assert cfg.episodes == data["episodes"]
    assert cfg.episode_length == data["episode_length"]
    assert cfg.discount == data["discount"]
    assert cfg.learning_rate == data["learning_rate"]
    assert cfg.exploration_decay == data["exploration_decay"]

    q = data["quantizer"]
    assert cfg.quantizer.price_buckets == q["price_buckets"]
    assert cfg.quantizer.volume_buckets == q["volume_buckets"]
    assert cfg.quantizer.trend_buckets == q["trend_buckets"]
    assert cfg.quantizer.scarcity_buckets == q["scarcity_buckets"]
    assert cfg.quantizer.mood_buckets == q["mood_buckets"]

    r = data["rewards"]
    assert cfg.rewards.pnl == r["pnl"]
    assert cfg.rewards.scarcity_alignment == r["scarcity_alignment"]
    assert cfg.rewards.ethics_bonus == r["ethics_bonus"]
    assert cfg.rewards.communication_clarity == r["communication_clarity"]

