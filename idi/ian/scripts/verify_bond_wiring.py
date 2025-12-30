#!/usr/bin/env python3
"""Verify CommitterBond kernel wiring."""

import sys
sys.path.insert(0, '.')

from idi.ian.network.economics import CommitterBond, BondStatus

def main() -> None:
    print("Test: CommitterBond Kernel Wiring")
    
    # Create bond with min_bond=10 (kernel scales to 1-50)
    bond = CommitterBond(
        committer_id="node1",
        goal_id="goal1",
        amount=100,  # Kernel max is 100
        min_bond=10,
    )
    
    # Test 1: Initial state
    assert bond.status == BondStatus.ACTIVE, f"Expected ACTIVE, got {bond.status}"
    assert bond.total_slashed == 0, f"Expected 0 slashed, got {bond.total_slashed}"
    assert bond.can_commit() is True
    print(f"  Init State: {bond.status.name} - OK")
    
    # Test 2: Lock bond
    result = bond.lock(duration_ms=1000)
    assert result is True, "Lock should succeed"
    assert bond.status == BondStatus.LOCKED, f"Expected LOCKED, got {bond.status}"
    assert bond.can_commit() is False
    print(f"  Lock: {bond.status.name} - OK")
    
    # Test 3: Unlock bond
    result = bond.unlock()
    assert result is True, "Unlock should succeed"
    assert bond.status == BondStatus.ACTIVE, f"Expected ACTIVE, got {bond.status}"
    print(f"  Unlock: {bond.status.name} - OK")
    
    # Test 4: Slash partial (stays ACTIVE)
    result = bond.slash(slash_amount=20)
    assert result is True, "Slash should succeed"
    assert bond.status == BondStatus.ACTIVE, f"Expected ACTIVE, got {bond.status}"
    assert bond.total_slashed == 20, f"Expected 20 slashed, got {bond.total_slashed}"
    assert bond.effective_bond() == 80
    print(f"  Slash Partial: status={bond.status.name}, slashed={bond.total_slashed}, effective={bond.effective_bond()} - OK")
    
    # Test 5: Slash below min -> transition to SLASHED
    # Current: amount=100, slashed=20, effective=80, min_bond=10
    # Need to slash enough to drop below min_bond: 80-10=70 more
    result = bond.slash(slash_amount=75)  # This should trigger slash_below_min
    assert result is True, f"Slash should succeed, status={bond.status}"
    assert bond.status == BondStatus.SLASHED, f"Expected SLASHED, got {bond.status}"
    print(f"  Slash Below Min: {bond.status.name} - OK")
    
    # Test 6: Withdraw from SLASHED
    result = bond.withdraw()
    assert result is True, "Withdraw should succeed from SLASHED"
    assert bond.status == BondStatus.WITHDRAWN, f"Expected WITHDRAWN, got {bond.status}"
    assert bond.effective_bond() == 0, f"Withdrawn should have 0 effective bond"
    print(f"  Withdraw: {bond.status.name} - OK")
    
    # Test 7: Cannot lock withdrawn bond
    result = bond.lock(duration_ms=1000)
    assert result is False, "Should not be able to lock withdrawn bond"
    print(f"  Lock Withdrawn (rejected): {bond.status.name} - OK")
    
    # Test 8: Topup (new Foundry feature)
    bond2 = CommitterBond(
        committer_id="node2",
        goal_id="goal2",
        amount=1000,
        min_bond=100,
    )
    result = bond2.topup(topup_amount=500)
    assert result is True, f"Topup should succeed, got {result}"
    assert bond2.amount == 1500, f"Expected amount 1500, got {bond2.amount}"
    print(f"  Topup: amount={bond2.amount} - OK")
    
    # Test 9: Kernel Invariant check
    bond._check_invariants()  # Should not raise
    print(f"  Kernel Invariants: PASS")
    
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
