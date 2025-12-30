from __future__ import annotations

from idi.ian.network.consensus import ConsensusCoordinator, ConsensusConfig, ConsensusState, PeerStateSnapshot


class _MockGoal:
    goal_id = "test_goal"


class _MockHeader:
    hash = b"mock_head_hash"


class _MockHead:
    def __init__(self) -> None:
        self.header = _MockHeader()


class _MockChain:
    def __init__(self) -> None:
        self.head = _MockHead()


class _MockCoordinator:
    def __init__(self) -> None:
        self.chain = _MockChain()
        self.goal_spec = _MockGoal()


def _peer(id_suffix: str, *, matching: bool) -> PeerStateSnapshot:
    return PeerStateSnapshot(
        node_id=f"peer_{id_suffix}",
        goal_id="test_goal",
        log_root=b"mock_head_hash" if matching else b"bad_hash",
        log_size=10,
        leaderboard_root=b"",
        active_policy_hash=b"",
        timestamp_ms=0,
    )


def test_consensus_kernel_wiring_reconciles_counts_and_state() -> None:
    config = ConsensusConfig(min_peers_for_consensus=2, quorum_threshold=0.66)
    coord = ConsensusCoordinator(_MockCoordinator(), "me", config)

    # Init
    assert coord.consensus_state == ConsensusState.ISOLATED
    assert coord._kstate.state == "ISOLATED"

    # 2 matching peers => synchronized
    coord._peer_states = {"p1": _peer("1", matching=True), "p2": _peer("2", matching=True)}
    coord._update_kernel_state()
    assert coord._kstate.matching_peers == 2
    assert coord.consensus_state == ConsensusState.SYNCHRONIZED

    # Add a 3rd matching peer
    coord._peer_states["p3"] = _peer("3", matching=True)
    coord._update_kernel_state()
    assert coord._kstate.matching_peers == 3
    assert coord.consensus_state == ConsensusState.SYNCHRONIZED

    # Remove a matching peer (still >= min, still quorum)
    del coord._peer_states["p3"]
    coord._update_kernel_state()
    assert coord._kstate.matching_peers == 2
    assert coord.consensus_state == ConsensusState.SYNCHRONIZED

    # Drop below min => isolated
    del coord._peer_states["p2"]
    coord._update_kernel_state()
    assert coord.consensus_state == ConsensusState.ISOLATED
    assert coord._kstate.state == "ISOLATED"


def test_consensus_kernel_envelope_fail_closed_when_over_capacity() -> None:
    config = ConsensusConfig(min_peers_for_consensus=2, quorum_threshold=0.66)
    coord = ConsensusCoordinator(_MockCoordinator(), "me", config)

    # Exceed kernel peer_count domain (max 10) => must fail closed.
    coord._peer_states = {f"p{i}": _peer(str(i), matching=True) for i in range(11)}
    coord._update_kernel_state()

    assert coord.can_commit_to_tau() is False
    assert coord._verified_envelope_ok is False

