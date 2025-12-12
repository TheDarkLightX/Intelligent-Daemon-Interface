from pathlib import Path

from idi_iann.config import TrainingConfig, TileCoderConfig, CommunicationConfig
from idi_iann.trainer import QTrainer


def test_trainer_produces_traces(tmp_path: Path) -> None:
    trainer = QTrainer(TrainingConfig(episodes=4, episode_length=8))
    policy, trace = trainer.run()
    assert policy is not None
    assert len(trace.ticks) == 8
    trace.export(tmp_path / "inputs")
    for name in (
        "q_buy.in",
        "q_sell.in",
        "risk_budget_ok.in",
        "q_emote_positive.in",
        "q_emote_alert.in",
        "q_emote_persistence.in",
        "q_regime.in",
        "price_up.in",
        "price_down.in",
        "weight_momentum.in",
        "weight_contra.in",
        "weight_trend.in",
    ):
        assert (tmp_path / "inputs" / name).exists()


def test_trainer_with_tilecoder_and_comm(tmp_path: Path) -> None:
    cfg = TrainingConfig(
        episodes=2,
        episode_length=4,
        tile_coder=TileCoderConfig(num_tilings=2, tile_sizes=(2, 2, 2, 2, 2), offsets=(0, 1, 0, 1, 0)),
        communication=CommunicationConfig(actions=("silent", "alert")),
    )
    trainer = QTrainer(cfg)
    _, trace = trainer.run()
    trace.export(tmp_path / "inputs_tc")
    assert (tmp_path / "inputs_tc" / "q_emote_alert.in").exists()

