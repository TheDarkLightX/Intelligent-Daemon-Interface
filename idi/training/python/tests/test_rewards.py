from idi_iann.rewards import CommunicationRewardShaper


def test_alert_reward_shaping_with_risk_signal() -> None:
    shaper = CommunicationRewardShaper(alert_bonus=0.2, alert_mismatch_penalty=-0.3)
    shaped = shaper.shape(
        base_reward=0.0,
        comm_action="alert",
        next_state=(0, 0, 1),
        features={"risk_signal": 1.0},
    )
    assert shaped > 0


def test_alert_reward_penalty_without_risk_signal() -> None:
    shaper = CommunicationRewardShaper(alert_bonus=0.2, alert_mismatch_penalty=-0.3)
    shaped = shaper.shape(
        base_reward=0.0,
        comm_action="alert",
        next_state=(0, 0, 0),
        features={"risk_signal": 0.0},
    )
    assert shaped < 0

