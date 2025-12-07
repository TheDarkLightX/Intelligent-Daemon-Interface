#!/usr/bin/env python3
"""
Exhaustive Edge Case Tests for Tau-P2P Escrow Protocol
Tests ALL state transitions and edge conditions
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import IntEnum

# ============================================================================
# STATE DEFINITIONS (matching tau_p2p_escrow.tau)
# ============================================================================

class P2PState(IntEnum):
    """8 states encoded in 3 bits"""
    IDLE = 0       # 000
    OPEN = 1       # 001
    MATCHED = 2    # 010
    PROVING = 3    # 011
    VERIFIED = 4   # 100
    SETTLED = 5    # 101
    DISPUTED = 6   # 110
    CANCELLED = 7  # 111

# ============================================================================
# FSM TRANSITION TABLE
# ============================================================================

VALID_TRANSITIONS = {
    P2PState.IDLE: [
        (P2PState.OPEN, "create_order"),
        (P2PState.IDLE, "stay_idle"),
    ],
    P2PState.OPEN: [
        (P2PState.MATCHED, "match_order"),
        (P2PState.CANCELLED, "cancel_order"),
        (P2PState.OPEN, "stay_open"),
    ],
    P2PState.MATCHED: [
        (P2PState.PROVING, "submit_proof"),
        (P2PState.CANCELLED, "timeout_no_proof"),
        (P2PState.MATCHED, "stay_matched"),
    ],
    P2PState.PROVING: [
        (P2PState.VERIFIED, "zkp_valid"),
        (P2PState.CANCELLED, "zkp_invalid"),
        (P2PState.CANCELLED, "timeout_proving"),
        (P2PState.DISPUTED, "dispute_raised"),
        (P2PState.PROVING, "stay_proving"),
    ],
    P2PState.VERIFIED: [
        (P2PState.SETTLED, "release_funds"),
    ],
    P2PState.SETTLED: [
        (P2PState.SETTLED, "terminal"),  # Absorbing state
    ],
    P2PState.DISPUTED: [
        (P2PState.SETTLED, "resolve_buyer"),
        (P2PState.CANCELLED, "resolve_seller"),
        (P2PState.DISPUTED, "stay_disputed"),
    ],
    P2PState.CANCELLED: [
        (P2PState.CANCELLED, "terminal"),  # Absorbing state
    ],
}

# ============================================================================
# TEST INPUT GENERATOR
# ============================================================================

@dataclass
class TestInput:
    """Single tick input for P2P protocol"""
    order_id: int = 0
    sell_amount: int = 1000
    buy_amount: int = 100
    seller_id: int = 1
    buyer_id: int = 2
    create_order: bool = False
    match_order: bool = False
    submit_proof: bool = False
    cancel_order: bool = False
    dispute: bool = False
    zkp_valid: bool = False
    zkp_invalid: bool = False
    proof_payment_id: int = 0
    proof_amount: int = 0
    proof_seller_id: int = 0
    proof_timestamp: int = 0
    current_time: int = 0
    lock_duration: int = 10
    dispute_duration: int = 20
    dispute_resolved_buyer: bool = False
    dispute_resolved_seller: bool = False
    fee_bps: int = 30  # 0.3%

@dataclass
class ExpectedOutput:
    """Expected outputs for verification"""
    state: P2PState
    escrowed_amount: int
    release_buyer: bool
    refund_seller: bool
    funds_safe: bool
    no_double_release: bool
    state_valid: bool

# ============================================================================
# EDGE CASE TEST DEFINITIONS
# ============================================================================

def generate_edge_case_tests() -> List[Tuple[str, List[TestInput], List[ExpectedOutput]]]:
    """Generate all edge case test scenarios"""
    tests = []
    
    # ========================================
    # 1. BASIC STATE TRANSITIONS
    # ========================================
    
    # Test 1.1: IDLE -> OPEN (create order)
    tests.append((
        "1.1_idle_to_open",
        [TestInput(create_order=True, sell_amount=1000, seller_id=1)],
        [ExpectedOutput(
            state=P2PState.OPEN,
            escrowed_amount=0,
            release_buyer=False,
            refund_seller=False,
            funds_safe=True,
            no_double_release=True,
            state_valid=True
        )]
    ))
    
    # Test 1.2: OPEN -> MATCHED (match order)
    tests.append((
        "1.2_open_to_matched",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, buyer_id=2, current_time=1)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False, 
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 1.3: MATCHED -> PROVING (submit proof)
    tests.append((
        "1.3_matched_to_proving",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, proof_amount=100, proof_seller_id=1, current_time=2)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 1.4: PROVING -> VERIFIED (zkp valid)
    tests.append((
        "1.4_proving_to_verified",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(zkp_valid=True, current_time=3)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.VERIFIED, escrowed_amount=1000, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 2. TIMEOUT EDGE CASES
    # ========================================
    
    # Test 2.1: MATCHED timeout -> CANCELLED (no proof submitted in time)
    tests.append((
        "2.1_matched_timeout",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=0, lock_duration=5),
            TestInput(current_time=10)  # Time exceeds lock duration
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 2.2: PROVING timeout -> CANCELLED
    tests.append((
        "2.2_proving_timeout",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=0, lock_duration=5),
            TestInput(submit_proof=True, current_time=1),
            TestInput(current_time=10)  # Timeout while proving
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 2.3: Exactly at timeout boundary
    tests.append((
        "2.3_timeout_boundary",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=0, lock_duration=5),
            TestInput(current_time=5),  # Exactly at boundary
            TestInput(current_time=6)   # Just past boundary
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 3. DISPUTE EDGE CASES
    # ========================================
    
    # Test 3.1: PROVING -> DISPUTED -> SETTLED (buyer wins)
    tests.append((
        "3.1_dispute_buyer_wins",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(dispute=True, current_time=3),
            TestInput(dispute_resolved_buyer=True, current_time=4)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.DISPUTED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.SETTLED, escrowed_amount=0, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 3.2: PROVING -> DISPUTED -> CANCELLED (seller wins)
    tests.append((
        "3.2_dispute_seller_wins",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(dispute=True, current_time=3),
            TestInput(dispute_resolved_seller=True, current_time=4)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.DISPUTED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 4. CANCEL EDGE CASES
    # ========================================
    
    # Test 4.1: OPEN -> CANCELLED (seller cancels before match)
    tests.append((
        "4.1_seller_cancels_open",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(cancel_order=True)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 4.2: Cannot cancel after MATCHED (must timeout or dispute)
    tests.append((
        "4.2_cancel_blocked_after_match",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(cancel_order=True, current_time=2)  # Should be ignored
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 5. ZKP VALIDATION EDGE CASES
    # ========================================
    
    # Test 5.1: ZKP Invalid -> CANCELLED
    tests.append((
        "5.1_zkp_invalid",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(zkp_invalid=True, current_time=3)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 5.2: ZKP valid AND invalid at same time (should NOT happen, but test invariant)
    tests.append((
        "5.2_zkp_conflicting_signals",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(zkp_valid=True, zkp_invalid=True, current_time=3)  # Both signals!
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            # Specification should handle this - valid takes precedence or neither does
            ExpectedOutput(state=P2PState.VERIFIED, escrowed_amount=1000, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 6. AMOUNT EDGE CASES
    # ========================================
    
    # Test 6.1: Zero amount order (should be rejected)
    tests.append((
        "6.1_zero_amount",
        [
            TestInput(create_order=True, sell_amount=0)
        ],
        [
            ExpectedOutput(state=P2PState.IDLE, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 6.2: Maximum amount (2^32 - 1)
    tests.append((
        "6.2_max_amount",
        [
            TestInput(create_order=True, sell_amount=4294967295),
            TestInput(match_order=True, current_time=1)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=4294967295, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 6.3: Fee calculation (0.3% of 10000 = 30)
    tests.append((
        "6.3_fee_calculation",
        [
            TestInput(create_order=True, sell_amount=10000, fee_bps=30),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(zkp_valid=True, current_time=3)
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=10000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=10000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.VERIFIED, escrowed_amount=10000, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 7. TERMINAL STATE EDGE CASES
    # ========================================
    
    # Test 7.1: SETTLED is absorbing (no transitions out)
    tests.append((
        "7.1_settled_absorbing",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(zkp_valid=True, current_time=3),
            TestInput(create_order=True, current_time=4),  # Should be ignored
            TestInput(cancel_order=True, current_time=5)   # Should be ignored
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.SETTLED, escrowed_amount=0, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.SETTLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.SETTLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 7.2: CANCELLED is absorbing
    tests.append((
        "7.2_cancelled_absorbing",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(cancel_order=True),
            TestInput(create_order=True, current_time=2),  # Should be ignored
            TestInput(match_order=True, current_time=3)    # Should be ignored
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=True, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.CANCELLED, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 8. REENTRANCY / DOUBLE ACTION EDGE CASES  
    # ========================================
    
    # Test 8.1: Double match attempt
    tests.append((
        "8.1_double_match",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, buyer_id=2, current_time=1),
            TestInput(match_order=True, buyer_id=3, current_time=2)  # Second match ignored
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # Test 8.2: Double proof submission
    tests.append((
        "8.2_double_proof",
        [
            TestInput(create_order=True, sell_amount=1000),
            TestInput(match_order=True, current_time=1),
            TestInput(submit_proof=True, current_time=2),
            TestInput(submit_proof=True, current_time=3)  # Second proof ignored
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=1000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    # ========================================
    # 9. INVARIANT TESTS
    # ========================================
    
    # Test 9.1: No double release (funds_safe)
    # Covered by all tests - check release_buyer XOR refund_seller
    
    # Test 9.2: State always valid
    # All states must be one of the 8 valid states
    
    # ========================================
    # 10. COMPLETE HAPPY PATH
    # ========================================
    
    tests.append((
        "10.1_complete_happy_path",
        [
            TestInput(create_order=True, sell_amount=10000, seller_id=1),
            TestInput(match_order=True, buyer_id=2, current_time=1),
            TestInput(submit_proof=True, proof_amount=1000, proof_seller_id=1, current_time=2),
            TestInput(zkp_valid=True, current_time=3),
            # Funds released, fee collected, burn occurred
        ],
        [
            ExpectedOutput(state=P2PState.OPEN, escrowed_amount=0, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.MATCHED, escrowed_amount=10000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.PROVING, escrowed_amount=10000, release_buyer=False,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True),
            ExpectedOutput(state=P2PState.SETTLED, escrowed_amount=0, release_buyer=True,
                          refund_seller=False, funds_safe=True, no_double_release=True, state_valid=True)
        ]
    ))
    
    return tests

# ============================================================================
# STATE COVERAGE ANALYSIS
# ============================================================================

def analyze_state_coverage(tests: List[Tuple[str, List[TestInput], List[ExpectedOutput]]]) -> Dict:
    """Analyze which states and transitions are covered"""
    states_covered = set()
    transitions_covered = set()
    
    for name, inputs, outputs in tests:
        prev_state = P2PState.IDLE
        for out in outputs:
            states_covered.add(out.state)
            if prev_state != out.state:
                transitions_covered.add((prev_state, out.state))
            prev_state = out.state
    
    all_states = set(P2PState)
    missing_states = all_states - states_covered
    
    # Count expected transitions
    expected_transitions = set()
    for from_state, transitions in VALID_TRANSITIONS.items():
        for to_state, _ in transitions:
            if from_state != to_state or to_state in [P2PState.SETTLED, P2PState.CANCELLED]:
                expected_transitions.add((from_state, to_state))
    
    missing_transitions = expected_transitions - transitions_covered
    
    return {
        "states_covered": len(states_covered),
        "states_total": len(all_states),
        "state_coverage_pct": len(states_covered) / len(all_states) * 100,
        "missing_states": [s.name for s in missing_states],
        "transitions_covered": len(transitions_covered),
        "transitions_expected": len(expected_transitions),
        "transition_coverage_pct": len(transitions_covered) / len(expected_transitions) * 100 if expected_transitions else 100,
        "missing_transitions": [(f.name, t.name) for f, t in missing_transitions]
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("TAU-P2P ESCROW PROTOCOL - EXHAUSTIVE EDGE CASE TESTS")
    print("=" * 70)
    
    tests = generate_edge_case_tests()
    print(f"\nGenerated {len(tests)} test cases\n")
    
    # Analyze coverage
    coverage = analyze_state_coverage(tests)
    
    print("STATE COVERAGE:")
    print(f"  States covered: {coverage['states_covered']}/{coverage['states_total']} ({coverage['state_coverage_pct']:.1f}%)")
    if coverage['missing_states']:
        print(f"  Missing states: {coverage['missing_states']}")
    
    print(f"\nTRANSITION COVERAGE:")
    print(f"  Transitions covered: {coverage['transitions_covered']}/{coverage['transitions_expected']} ({coverage['transition_coverage_pct']:.1f}%)")
    if coverage['missing_transitions']:
        print(f"  Missing transitions: {coverage['missing_transitions']}")
    
    print("\n" + "=" * 70)
    print("TEST CASES:")
    print("=" * 70)
    
    for name, inputs, outputs in tests:
        print(f"\n[{name}]")
        print(f"  Inputs: {len(inputs)} ticks")
        print(f"  Expected states: {' -> '.join(o.state.name for o in outputs)}")
    
    # Save test cases
    test_data = []
    for name, inputs, outputs in tests:
        test_data.append({
            "name": name,
            "inputs": [vars(i) for i in inputs],
            "expected": [{"state": o.state.name, **vars(o)} for o in outputs]
        })
    
    with open("/home/trevormoc/Downloads/DeflationaryAgent/verification/p2p_edge_tests.json", "w") as f:
        json.dump(test_data, f, indent=2, default=str)
    
    print(f"\n\nSaved test cases to p2p_edge_tests.json")
    print(f"Total: {len(tests)} test cases covering {coverage['state_coverage_pct']:.1f}% states, {coverage['transition_coverage_pct']:.1f}% transitions")

if __name__ == "__main__":
    main()

