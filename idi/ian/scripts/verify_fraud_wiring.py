#!/usr/bin/env python3
"""Verify FraudProof kernel wiring."""

import sys
sys.path.insert(0, '.')

from idi.ian.network.fraud import FraudProof, FraudType, FraudStatus, RejectionReason

def main() -> None:
    print("Test: FraudProof Kernel Wiring")
    
    # Test 1: Initial state
    proof = FraudProof(
        fraud_type=FraudType.INVALID_COMMIT_SIGNATURE,
        goal_id="goal1",
        challenged_commit_hash=b"hash1",
    )
    
    assert proof.status == FraudStatus.CREATED, f"Expected CREATED, got {proof.status}"
    assert proof.rejection_reason == RejectionReason.NONE, f"Expected NONE, got {proof.rejection_reason}"
    print(f"  Init State: {proof.status.name} - OK")
    
    # Test 2: Submit
    result = proof.submit()
    assert result is True, "Submit should succeed"
    assert proof.status == FraudStatus.SUBMITTED, f"Expected SUBMITTED, got {proof.status}"
    print(f"  Submit: {proof.status.name} - OK")
    
    # Test 3: Confirm Fraud (Verify Valid)
    result = proof.confirm_fraud()
    assert result is True, "Confirm Fraud should succeed"
    assert proof.status == FraudStatus.VERIFIED, f"Expected VERIFIED, got {proof.status}"
    assert proof.rejection_reason == RejectionReason.NONE
    print(f"  Confirm Fraud: {proof.status.name} - OK")
    
    # Test 4: Scenario B (Reject Fraud)
    proof2 = FraudProof(
        fraud_type=FraudType.WRONG_ORDERING,
        goal_id="goal2",
        challenged_commit_hash=b"hash2",
    )
    proof2.submit()
    print(f"  [B] Submitted: {proof2.status.name}")
    
    # Reject because signature invalid
    result = proof2.reject_fraud(RejectionReason.INVALID_SIG)
    assert result is True, "Reject should succeed"
    assert proof2.status == FraudStatus.REJECTED, f"Expected REJECTED, got {proof2.status}"
    assert proof2.rejection_reason == RejectionReason.INVALID_SIG, f"Expected INVALID_SIG, got {proof2.rejection_reason}"
    print(f"  [B] Reject: {proof2.status.name}, Reason: {proof2.rejection_reason.name} - OK")
    
    # Test 5: Invariant check
    proof2._check_invariants()
    print("  Invariants: PASS")

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
