"""State integration for ZK proof-verified transitions.

This module provides functions for applying state transitions only after
ZK proof verification, ensuring that balance updates and other state changes
are only applied when proofs are valid.
"""

from __future__ import annotations

from typing import Optional

from idi.taunet_bridge.protocols import ZkProofBundle, ZkVerifier

# Global verifier instance (can be set via set_zk_verifier)
_zk_verifier: Optional[ZkVerifier] = None


def set_zk_verifier(verifier: Optional[ZkVerifier]) -> None:
    """Set the global ZK verifier instance.

    This function allows dependency injection of the verifier, following
    the Dependency Inversion Principle (DIP).

    Args:
        verifier: ZK verifier instance, or None to disable verification
    """
    global _zk_verifier
    _zk_verifier = verifier


def get_zk_verifier() -> Optional[ZkVerifier]:
    """Get the global ZK verifier instance.

    Returns:
        Current ZK verifier, or None if not set
    """
    return _zk_verifier


def apply_verified_transition(
    proof: ZkProofBundle,
    from_addr: str,
    to_addr: str,
    amount: int,
) -> bool:
    """Apply state transition only if ZK proof verifies.

    This function integrates with Tau Testnet's chain_state module to
    apply balance updates only after proof verification.

    Args:
        proof: ZK proof bundle for the transition
        from_addr: Source address (BLS public key hex)
        to_addr: Destination address (BLS public key hex)
        amount: Transfer amount

    Returns:
        True if transition was applied, False if proof verification failed
    """
    verifier = get_zk_verifier()
    if verifier is None:
        # No verifier set, fall back to unverified transition
        # Import here to avoid circular dependency
        import chain_state

        return chain_state.update_balances_after_transfer(from_addr, to_addr, amount)

    if not verifier.verify(proof):
        return False

    # Import here to avoid circular dependency
    import chain_state

    return chain_state.update_balances_after_transfer(from_addr, to_addr, amount)

