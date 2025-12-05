from pathlib import Path

from idi_iann.config import TrainingConfig
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
    ):
        assert (tmp_path / "inputs" / name).exists()

