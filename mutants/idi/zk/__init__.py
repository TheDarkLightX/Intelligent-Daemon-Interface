"""IDI Zero-Knowledge Proof Integration.

Provides ZK proof generation and verification for Q-table based agents.
"""

from .proof_manager import ProofBundle, generate_proof, verify_proof
from .witness_generator import (
    QTableEntry,
    QTableWitness,
    MerkleTree,
    generate_witness_from_q_table,
    serialize_witness,
)
from .merkle_tree import MerkleTreeBuilder
from .workflow import WorkflowResult, run_training_to_proof_workflow
from .qtable_prover import generate_qtable_proof, verify_qtable_proof
from .training_integration import (
    generate_proofs_from_training_output,
    verify_training_proofs,
)
from .tau_integration import (
    verify_before_tau_execution,
    execute_tau_with_proof_verification,
    log_proof_verification,
)

__all__ = [
    "ProofBundle",
    "generate_proof",
    "verify_proof",
    "QTableEntry",
    "QTableWitness",
    "MerkleTree",
    "generate_witness_from_q_table",
    "serialize_witness",
    "MerkleTreeBuilder",
    "WorkflowResult",
    "run_training_to_proof_workflow",
    "generate_qtable_proof",
    "verify_qtable_proof",
    "generate_proofs_from_training_output",
    "verify_training_proofs",
    "verify_before_tau_execution",
    "execute_tau_with_proof_verification",
    "log_proof_verification",
]

