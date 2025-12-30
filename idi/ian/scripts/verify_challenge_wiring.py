#!/usr/bin/env python3
"""Verify ChallengeBond kernel wiring."""

import sys
sys.path.insert(0, '.')

from idi.ian.network.economics import ChallengeBond, BondStatus

def main() -> None:
    print("Test: ChallengeBond Kernel Wiring")
    
    # Test 1: Initial state
    bond = ChallengeBond(
        challenger_id="challenger1",
        goal_id="goal1",
        challenged_commit_hash=b"hash1",
        amount=500_000,
    )
    
    assert bond.status == BondStatus.ACTIVE, f"Expected ACTIVE, got {bond.status}"
    print(f"  Init State: {bond.status.name} - OK")
    
    # Test 2: Lock for challenge
    result = bond.lock()
    assert result is True, "Lock should succeed"
    assert bond.status == BondStatus.LOCKED, f"Expected LOCKED, got {bond.status}"
    print(f"  Lock: {bond.status.name} - OK")
    
    # Test 3: Resolve Valid (Unlock)
    result = bond.resolve(valid=True)
    assert result is True, "Resolve Valid should succeed"
    assert bond.status == BondStatus.ACTIVE, f"Expected ACTIVE, got {bond.status}"
    assert bond.challenge_valid is True
    print(f"  Resolve Valid: {bond.status.name} - OK")
    
    # Test 4: Withdraw (Full amount)
    result = bond.withdraw()
    assert result is True, "Withdraw should succeed"
    assert bond.status == BondStatus.WITHDRAWN, f"Expected WITHDRAWN, got {bond.status}"
    assert bond.amount == 0, f"Expected amount 0, got {bond.amount}"
    print(f"  Withdraw: {bond.status.name} - OK")
    
    # Test 5: Scenario B (Invalid Challenge)
    bond2 = ChallengeBond(
        challenger_id="challenger2",
        goal_id="goal1",
        challenged_commit_hash=b"hash2",
        amount=200_000_000,
    )
    bond2.lock()
    print(f"  [B] Locked: {bond2.status.name}")
    
    # Resolve Invalid (Slashing)
    result = bond2.resolve(valid=False)
    # Note: ChallengeBond resolves invalid -> SLASHED.
    # But does it slash fully? 
    # Our wiring: resolve(valid=False) -> 'resolve_invalid' with args={'slash_amount': self.amount}
    assert result is True, "Resolve Invalid should succeed"
    assert bond2.status == BondStatus.SLASHED, f"Expected SLASHED, got {bond2.status}"
    assert bond2.amount == 0, f"Expected amount 0 (full slash), got {bond2.amount}"
    print(f"  [B] Resolve Invalid: {bond2.status.name}, Amount: {bond2.amount} - OK")
    
    # Withdraw remainder (should be 0)
    result = bond2.withdraw()
    assert result is True, "Withdraw remainder should succeed"
    assert bond2.status == BondStatus.WITHDRAWN
    print(f"  [B] Withdraw: {bond2.status.name} - OK")

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
