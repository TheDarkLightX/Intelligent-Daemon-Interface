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
]

