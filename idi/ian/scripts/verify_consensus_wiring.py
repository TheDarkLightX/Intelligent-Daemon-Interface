
import asyncio
import logging
import sys
from idi.ian.network.consensus import ConsensusCoordinator, ConsensusConfig, ConsensusState, PeerStateSnapshot

# Mock Classes
class MockCoordinator:
    def __init__(self):
        self.chain = MockChain()
        self.goal_spec = MockGoal()

class MockGoal:
    goal_id = "test_goal"

class MockChain:
    def __init__(self):
        self.head = MockHead()

class MockHead:
    def __init__(self):
        self.header = MockHeader()

class MockHeader:
    hash = b"mock_head_hash"

def create_peer(id_suffix: str, matching: bool = True) -> PeerStateSnapshot:
    return PeerStateSnapshot(
        node_id=f"peer_{id_suffix}",
        goal_id="test_goal",
        log_root=b"mock_head_hash" if matching else b"bad_hash",
        log_size=10,
        leaderboard_root=b"",
        active_policy_hash=b"",
        timestamp_ms=0
    )

def test_consensus_wiring():
    print("Test: Consensus Wiring (Patched Kernel)")
    
    # 1. Setup
    config = ConsensusConfig(
        min_peers_for_consensus=2,
        quorum_threshold=0.66
    )
    coord = ConsensusCoordinator(MockCoordinator(), "me", config)
    
    # Verify Init
    assert coord._kstate.state == "ISOLATED"
    print("  Init State: ISOLATED - OK")
    
    # 2. Add 2 Matching Peers (Should Sync)
    coord._peer_states = {
        "p1": create_peer("1", True),
        "p2": create_peer("2", True)
    }
    
    # Drive Kernel
    coord._update_kernel_state()
    
    # Check Logic:
    # ISOLATED -> peers_join_become_syncing (1 peer)
    # SYNCING -> peers_join_become_syncing (2 peers)
    # SYNCING -> sync_complete_become_synchronized (2 >= 2*0.66 -> True)
    
    print(f"  State after 2 peers: {coord.status.name}")
    print(f"  Kernel State: {coord._kstate.state}")
    print(f"  Kernel Matching: {coord._kstate.matching_peers}")
    
    assert coord._kstate.matching_peers == 2
    assert coord.status == ConsensusState.SYNCHRONIZED
    print("  Transition to SYNCHRONIZED - OK")
    
    # 3. Test Patch: Peer Leaves but Quorum Maintained
    # Add 3rd peer first
    coord._peer_states["p3"] = create_peer("3", True)
    coord._update_kernel_state()
    assert coord._kstate.matching_peers == 3
    print("  Added 3rd peer - OK")
    
    # Remove 3rd peer (Back to 2)
    # This requires 'peers_leave_matching'
    del coord._peer_states["p3"]
    coord._update_kernel_state()
    
    print(f"  State after removal: {coord.status.name}")
    print(f"  Kernel Matching: {coord._kstate.matching_peers}")
    
    assert coord._kstate.matching_peers == 2
    assert coord.status == ConsensusState.SYNCHRONIZED
    print("  PATCH VERIFIED: Remained SYNCHRONIZED logic handled correctly - OK")
    
    # 4. Remove another peer (Drop below min=2)
    # This uses standard 'peers_leave_become_isolated'
    del coord._peer_states["p2"]
    coord._update_kernel_state()
    
    print(f"  State after dropping below min: {coord.status.name}")
    
    assert coord.status == ConsensusState.ISOLATED
    assert coord._kstate.state == "ISOLATED"
    print("  Transition to ISOLATED - OK")

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_consensus_wiring()
