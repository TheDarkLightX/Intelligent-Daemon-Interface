"""
ESSO-Verified Python Kernels

This package contains auto-generated, formally verified state machine
implementations from ESSO (Evolutionary Spec Search Optimizer).

Each kernel is:
- Generated from ESSO-IR models
- Verified with Z3+CVC5 SMT solvers (Inductive k=1)
- Standalone with no external dependencies
- Includes invariant checking on every transition

Usage:
    from idi.ian.network.kernels import circuit_breaker_fsm_ref as cb

    state = cb.init_state()
    result = cb.step(state, cb.Command(tag='record_failure', args={}))
    if result.ok:
        state = result.state
"""

# FSM Kernels (17 verified models)
from . import circuit_breaker_fsm_ref
from . import sync_state_fsm_ref
from . import consensus_state_fsm_ref
from . import bond_status_fsm_ref
from . import slot_state_fsm_ref
from . import task_state_fsm_ref
from . import fraud_proof_fsm_ref
from . import peer_state_fsm_ref
from . import rate_limiter_ref
from . import replay_protection_ref
from . import mempool_lifecycle_fsm_ref
from . import challenge_bond_fsm_ref
from . import evaluation_quorum_fsm_ref
from . import fast_lane_fsm_ref
from . import peer_tier_fsm_ref

# Micro-Kernels (data structure patterns)
from . import priority_queue_kernel_ref
from . import threshold_gate_kernel_ref

__all__ = [
    # FSM Kernels
    'circuit_breaker_fsm_ref',
    'sync_state_fsm_ref',
    'consensus_state_fsm_ref',
    'bond_status_fsm_ref',
    'slot_state_fsm_ref',
    'task_state_fsm_ref',
    'fraud_proof_fsm_ref',
    'peer_state_fsm_ref',
    'rate_limiter_ref',
    'replay_protection_ref',
    'mempool_lifecycle_fsm_ref',
    'challenge_bond_fsm_ref',
    'evaluation_quorum_fsm_ref',
    'fast_lane_fsm_ref',
    'peer_tier_fsm_ref',
    # Micro-Kernels
    'priority_queue_kernel_ref',
    'threshold_gate_kernel_ref',
]
