
import sys
import os
import time
from idi.ian.network.slotbatcher import Slot, SlotState, SlotContribution, MAX_CONTRIBUTIONS_PER_SLOT

def create_mock_contrib(id_suffix: str) -> SlotContribution:
    return SlotContribution(
        contribution_id=f"c{id_suffix}",
        contributor_id=f"u{id_suffix}",
        pack_hash=b"0"*32,
        commitment_hash=b"1"*32,
        quality_weight=1.0,
        submitted_at_ms=int(time.time()*1000)
    )

def test_slot_lifecycle():
    print("Test 1: Normal Lifecycle")
    slot = Slot(
        slot_id="test_slot_1",
        start_time_ms=0,
        end_time_ms=1000,
        _state=SlotState.COLLECTING
    )
    # Post init sets kstate
    assert slot._kstate.state == "COLLECTING"
    print("  Created (COLLECTING) - OK")

    # Add Contribution
    assert slot.add_contribution("c1", create_mock_contrib("1"))
    assert slot._kstate.contribution_count == 1
    print("  Added Contribution - OK")

    # Start Ordering
    slot.transition_to(SlotState.ORDERING)
    assert slot._kstate.state == "ORDERING"
    assert slot.state == SlotState.ORDERING
    print("  Transition to ORDERING - OK")

    # Try Add Contribution (Should Fail)
    print("Test 2: Reject Contribution in ORDERING")
    if not slot.add_contribution("c2", create_mock_contrib("2")):
        print("  Correctly rejected contribution in ORDERING - OK")
    else:
        print("  ERROR: Allowed contribution in ORDERING!")
        sys.exit(1)

    # Commit
    print("Test 3: Commit")
    slot.vrf_output = "mock_vrf" # Shell sets VRF
    # Note: Kernel checks 'has_vrf' AFTER transition? 
    # Or 'compute_vrf_and_commit' SETS has_vrf=True?
    # Command logic: "effects: has_vrf=True (implied by command)".
    # Kernel State Update: "has_vrf=True".
    slot.transition_to(SlotState.COMMITTED)
    assert slot._kstate.state == "COMMITTED"
    assert slot._kstate.has_vrf == True # Kernel sets this on transition!
    print("  Transition to COMMITTED - OK")

    # Execute
    print("Test 4: Execute")
    slot.ordered_ids = ["c1"] # Shell sets ordered ids
    slot.transition_to(SlotState.EXECUTED)
    assert slot._kstate.state == "EXECUTED"
    assert slot._kstate.has_ordered_ids == True
    print("  Transition to EXECUTED - OK")

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_slot_lifecycle()
