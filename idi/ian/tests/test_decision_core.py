"""Boundary and effect-refusal tests for the bridge's pure decision core."""

from dataclasses import replace
from itertools import product
import random
from idi.ian.decision_core import U64_MAX, deposit, half_slash, log_commit, upgrade
from idi.ian.tau_bridge import TauBridge, TauBridgeConfig, MockTauSender
from .test_phase4 import sample_goal_spec, sample_contribution_meta


def test_conservation_and_no_wrap():
    rng = random.Random(20261104)
    for bond in (
        list(range(1025))
        + [U64_MAX - 1, U64_MAX]
        + [rng.randrange(1 << 64) for _ in range(1000)]
    ):
        parts = half_slash(bond)
        assert sum(parts) == bond and all(0 <= p <= bond for p in parts)
    assert deposit(U64_MAX, 1, U64_MAX) == U64_MAX
    assert deposit(U64_MAX - 1, 1, U64_MAX) == U64_MAX
    assert deposit(8, 3, 10) == 8
    assert deposit(8, 2, 10) == 10


def test_commit_boundary_rejections():
    good = (True, b"a" * 32, 10, 9, b"b" * 32, 4)
    assert log_commit(*good).accepted
    for index, value in [
        (0, False),
        (1, b"a"),
        (2, 8),
        (2, -1),
        (2, True),
        (2, U64_MAX + 1),
        (5, 11),
    ]:
        bad = list(good)
        bad[index] = value
        assert not log_commit(*bad).accepted


def test_upgrade_governance_and_clock_product():
    for required, approved, registered, backwards in product([False, True], repeat=4):
        d = upgrade(
            registered,
            b"a" * 32,
            b"b" * 32,
            0,
            1.0,
            99 if backwards else 110,
            100,
            10,
            0,
            required,
            approved,
        )
        assert d.accepted == (
            registered and not backwards and (not required or approved)
        )
    assert not upgrade(
        True, b"a" * 32, b"b" * 32, 0, float("nan"), 110, 100, 10, 0, False, False
    ).accepted
    assert not upgrade(
        True, b"a" * 32, b"b" * 32, 0, 1, 110, 100, 10, U64_MAX, False, False
    ).accepted
    assert not upgrade(
        True, b"a" * 32, b"b" * 32, 0, 10**1000, 110, 100, 10, 0, False, False
    ).accepted


def test_refused_commit_has_no_send_or_state_change(sample_goal_spec):
    sender = MockTauSender()
    bridge = TauBridge(sender=sender)
    bridge.register_goal(sample_goal_spec)
    before = bridge.to_dict()
    sent = len(sender.sent_txs)
    assert not bridge.commit_log(sample_goal_spec.goal_id, b"a", 2, b"b" * 32, 1)[0]
    assert bridge.to_dict() == before and len(sender.sent_txs) == sent


def test_required_governance_needs_verification_not_signature_presence(
    sample_goal_spec, sample_contribution_meta
):
    sender = MockTauSender()
    config = TauBridgeConfig(
        require_governance_for_upgrade=True, upgrade_cooldown_seconds=0
    )
    bridge = TauBridge(sender=sender, config=config)
    bridge.register_goal(sample_goal_spec)
    sent = len(sender.sent_txs)
    args = (
        sample_goal_spec.goal_id,
        sample_contribution_meta,
        b"a" * 32,
        [b"not-a-valid-signature"],
    )
    assert not bridge.upgrade_policy(*args)[0] and len(sender.sent_txs) == sent
    bridge.governance_verifier = lambda tx: False
    assert not bridge.upgrade_policy(*args)[0]
    bridge.governance_verifier = lambda tx: True
    assert bridge.upgrade_policy(*args)[0] and len(sender.sent_txs) == sent + 1


def test_log_size_roundtrip_and_monotonicity(sample_goal_spec):
    bridge = TauBridge()
    bridge.register_goal(sample_goal_spec)
    assert bridge.commit_log(sample_goal_spec.goal_id, b"a" * 32, 10, b"b" * 32, 3)[0]
    restored = TauBridge.from_dict(bridge.to_dict())
    assert restored.get_state(sample_goal_spec.goal_id).log_size == 10
    assert not restored.commit_log(
        sample_goal_spec.goal_id, b"a" * 32, 9, b"b" * 32, 3
    )[0]
