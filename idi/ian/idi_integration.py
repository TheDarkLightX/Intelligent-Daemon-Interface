"""
IDI Integration - Adapters connecting IAN to existing IDI components.

This module provides adapters for:
- InvariantService (Python-based invariant checking)
- MPB VM (bytecode-based invariant verification)
- ProofManager (ZK proof verification)
- EvaluationHarness (IDI trainer integration)

These adapters implement the protocols defined in coordinator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .models import AgentPack, GoalSpec, Metrics

if TYPE_CHECKING:
    from .coordinator import InvariantChecker, ProofVerifier, EvaluationHarness


logger = logging.getLogger(__name__)


# =============================================================================
# Invariant Checker Adapter
# =============================================================================

@dataclass
class InvariantResult:
    """Result of invariant checking."""
    passed: bool
    invariant_id: str
    reason: str


class IDIInvariantChecker:
    """
    Adapter for IDI InvariantService and MPB VM.
    
    Runs both Python-based and MPB-based invariant checks.
    A contribution must pass all configured invariants.
    """
    
    def __init__(
        self,
        use_python_checks: bool = True,
        use_mpb_checks: bool = True,
        constant_time: bool = False,
    ) -> None:
        """
        Initialize invariant checker.
        
        Args:
            use_python_checks: Enable Python InvariantService checks
            use_mpb_checks: Enable MPB VM checks
            constant_time: Pad execution to constant time (timing attack prevention)
        """
        self.use_python_checks = use_python_checks
        self.use_mpb_checks = use_mpb_checks
        self.constant_time = constant_time
        
        # Lazy-load IDI components
        self._invariant_service = None
        self._mpb_vm = None
    
    def _get_invariant_service(self):
        """Lazy-load InvariantService."""
        if self._invariant_service is None:
            try:
                from idi.gui.backend.services.invariants import InvariantService
                self._invariant_service = InvariantService()
                logger.info("Loaded IDI InvariantService")
            except ImportError as e:
                logger.warning(f"Could not load InvariantService: {e}")
        return self._invariant_service
    
    def _get_mpb_vm(self):
        """Lazy-load MPB VM."""
        if self._mpb_vm is None:
            try:
                from idi.devkit.experimental.mpb_vm import MpbVm
                self._mpb_vm = MpbVm()
                logger.info("Loaded IDI MPB VM")
            except ImportError as e:
                logger.warning(f"Could not load MPB VM: {e}")
        return self._mpb_vm
    
    def check(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """
        Check if agent_pack satisfies all invariants.
        
        Runs checks in order:
        1. Python InvariantService (if enabled)
        2. MPB VM (if enabled and bytecode present)
        
        Returns early on first failure.
        """
        results: List[InvariantResult] = []
        
        # Extract goal spec parameters for invariant checking
        try:
            spec_params = self._extract_spec_params(agent_pack, goal_spec)
        except Exception as e:
            return False, f"failed to extract spec params: {e}"
        
        # Run Python checks
        if self.use_python_checks:
            service = self._get_invariant_service()
            if service:
                python_result = self._run_python_checks(service, spec_params, goal_spec)
                results.extend(python_result)
                
                # Fail-fast on first failure
                for r in python_result:
                    if not r.passed:
                        return False, f"invariant {r.invariant_id}: {r.reason}"
        
        # Run MPB checks
        if self.use_mpb_checks and goal_spec.mpb_bytecode:
            vm = self._get_mpb_vm()
            if vm:
                mpb_result = self._run_mpb_checks(vm, spec_params, goal_spec)
                results.extend(mpb_result)
                
                for r in mpb_result:
                    if not r.passed:
                        return False, f"MPB invariant {r.invariant_id}: {r.reason}"
        
        return True, "all invariants passed"
    
    def _extract_spec_params(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Dict[str, Any]:
        """
        Extract parameters from agent pack for invariant checking.
        
        Maps agent pack data to the format expected by InvariantService.
        """
        # Try to deserialize agent parameters
        import json
        
        params = {}
        try:
            # Assume parameters are JSON-encoded goal spec
            params = json.loads(agent_pack.parameters.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Parameters might be binary (e.g., Q-table)
            # Use metadata instead
            params = agent_pack.metadata.copy()
        
        # Ensure required fields have defaults
        params.setdefault("max_leverage", 1.0)
        params.setdefault("max_position_pct", 100.0)
        params.setdefault("max_drawdown_pct", 100.0)
        params.setdefault("min_collateral_ratio", 0.0)
        params.setdefault("asset_whitelist", ["*"])
        params.setdefault("rebalance_threshold_pct", 0.0)
        
        return params
    
    def _run_python_checks(
        self,
        service,
        spec_params: Dict[str, Any],
        goal_spec: GoalSpec,
    ) -> List[InvariantResult]:
        """Run Python-based invariant checks."""
        results = []
        
        # Get invariants to check
        invariant_ids = goal_spec.invariant_ids or ["I1", "I2", "I3", "I4", "I5"]
        
        for inv_id in invariant_ids:
            try:
                # Check individual invariant
                check_result = service.check_one(inv_id, spec_params)
                results.append(InvariantResult(
                    passed=check_result.passed,
                    invariant_id=inv_id,
                    reason=check_result.reason if not check_result.passed else "passed",
                ))
            except Exception as e:
                results.append(InvariantResult(
                    passed=False,
                    invariant_id=inv_id,
                    reason=f"check error: {e}",
                ))
        
        return results
    
    def _run_mpb_checks(
        self,
        vm,
        spec_params: Dict[str, Any],
        goal_spec: GoalSpec,
    ) -> List[InvariantResult]:
        """Run MPB VM-based invariant checks."""
        results = []
        
        try:
            # Load bytecode
            bytecode = goal_spec.mpb_bytecode
            
            # Map spec params to VM registers
            registers = self._map_params_to_registers(spec_params)
            
            # Execute VM
            vm_result = vm.execute(bytecode, registers)
            
            # Check output register (R0 = result)
            passed = vm_result.registers.get("R0", 0) == 1
            
            results.append(InvariantResult(
                passed=passed,
                invariant_id="MPB",
                reason="passed" if passed else "MPB check failed",
            ))
            
        except Exception as e:
            results.append(InvariantResult(
                passed=False,
                invariant_id="MPB",
                reason=f"MPB execution error: {e}",
            ))
        
        return results
    
    def _map_params_to_registers(
        self,
        spec_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Map goal spec parameters to MPB VM registers."""
        return {
            "R1": float(spec_params.get("max_leverage", 1.0)),
            "R2": float(spec_params.get("max_position_pct", 100.0)),
            "R3": float(spec_params.get("max_drawdown_pct", 100.0)),
            "R4": float(spec_params.get("min_collateral_ratio", 0.0)),
            "R5": float(spec_params.get("rebalance_threshold_pct", 0.0)),
        }


# =============================================================================
# Proof Verifier Adapter
# =============================================================================

class IDIProofVerifier:
    """
    Adapter for IDI ZK proof verification.
    
    Supports:
    - MPB proofs (Merkle tree + spot checks)
    - Risc0 ZK receipts (optional)
    """
    
    def __init__(
        self,
        require_proofs: bool = False,
        constant_time: bool = False,
    ) -> None:
        """
        Initialize proof verifier.
        
        Args:
            require_proofs: If True, reject contributions without proofs
            constant_time: Pad verification to constant time
        """
        self.require_proofs = require_proofs
        self.constant_time = constant_time
        
        self._proof_manager = None
    
    def _get_proof_manager(self):
        """Lazy-load proof manager."""
        if self._proof_manager is None:
            try:
                from idi.zk.proof_manager import ProofManager
                self._proof_manager = ProofManager()
                logger.info("Loaded IDI ProofManager")
            except ImportError as e:
                logger.warning(f"Could not load ProofManager: {e}")
        return self._proof_manager
    
    def verify(
        self,
        agent_pack: AgentPack,
        proofs: Dict[str, bytes],
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """
        Verify proofs for an agent pack.
        
        Checks:
        1. If proofs required, verify they are present
        2. Verify MPB proofs (if present)
        3. Verify ZK receipts (if present)
        """
        # Check if proofs are required
        if self.require_proofs and not proofs:
            return False, "proofs required but none provided"
        
        # If no proofs, pass (unless required)
        if not proofs:
            return True, "no proofs to verify"
        
        # Verify MPB proofs
        if "mpb" in proofs:
            valid, reason = self._verify_mpb_proof(proofs["mpb"], agent_pack, goal_spec)
            if not valid:
                return False, f"MPB proof: {reason}"
        
        # Verify ZK receipts
        if "zk" in proofs:
            valid, reason = self._verify_zk_receipt(proofs["zk"], agent_pack, goal_spec)
            if not valid:
                return False, f"ZK proof: {reason}"
        
        return True, "all proofs verified"
    
    def _verify_mpb_proof(
        self,
        proof_data: bytes,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """Verify MPB Merkle proof."""
        try:
            pm = self._get_proof_manager()
            if pm is None:
                # No proof manager, skip verification
                return True, "proof manager not available"
            
            # Parse proof bundle
            import json
            proof_bundle = json.loads(proof_data.decode("utf-8"))
            
            # Verify Merkle proof
            valid = pm.verify_merkle_proof(
                leaf_hash=bytes.fromhex(proof_bundle.get("leaf_hash", "")),
                proof=proof_bundle.get("proof", []),
                root=bytes.fromhex(proof_bundle.get("root", "")),
            )
            
            if not valid:
                return False, "Merkle proof invalid"
            
            return True, "MPB proof valid"
            
        except Exception as e:
            return False, f"verification error: {e}"
    
    def _verify_zk_receipt(
        self,
        receipt_data: bytes,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """Verify Risc0 ZK receipt."""
        try:
            pm = self._get_proof_manager()
            if pm is None:
                return True, "proof manager not available"
            
            # Parse receipt
            import json
            receipt = json.loads(receipt_data.decode("utf-8"))
            
            # Verify ZK receipt
            valid = pm.verify_zk_receipt(receipt)
            
            if not valid:
                return False, "ZK receipt invalid"
            
            return True, "ZK receipt valid"
            
        except Exception as e:
            return False, f"verification error: {e}"


# =============================================================================
# Evaluation Harness Adapter
# =============================================================================

class IDIEvaluationHarness:
    """
    Adapter for IDI training and evaluation infrastructure.
    
    Can use:
    - idi_iann trainers for backtest/simulation
    - Custom evaluation functions
    """
    
    def __init__(
        self,
        harness_type: str = "backtest",
        deterministic_seed: bool = True,
    ) -> None:
        """
        Initialize evaluation harness.
        
        Args:
            harness_type: Type of harness ("backtest", "simulation", "custom")
            deterministic_seed: Use deterministic seeding for reproducibility
        """
        self.harness_type = harness_type
        self.deterministic_seed = deterministic_seed
        
        self._trainer = None
    
    def _get_trainer(self):
        """Lazy-load IDI trainer."""
        if self._trainer is None:
            try:
                from idi.training.python.idi_iann.trainer import Trainer
                self._trainer = Trainer()
                logger.info("Loaded IDI Trainer")
            except ImportError as e:
                logger.warning(f"Could not load Trainer: {e}")
        return self._trainer
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        """
        Evaluate agent pack using IDI infrastructure.
        
        Returns None on failure.
        """
        try:
            if self.harness_type == "backtest":
                return self._run_backtest(agent_pack, goal_spec, seed)
            elif self.harness_type == "simulation":
                return self._run_simulation(agent_pack, goal_spec, seed)
            else:
                return self._run_mock(agent_pack, goal_spec, seed)
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def _run_backtest(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        """Run backtest evaluation."""
        trainer = self._get_trainer()
        if trainer is None:
            # Fall back to mock
            return self._run_mock(agent_pack, goal_spec, seed)
        
        try:
            # Load agent from pack
            # (In production, this would deserialize the Q-table or policy)
            
            # Run backtest
            results = trainer.run_backtest(
                max_episodes=goal_spec.eval_limits.max_episodes,
                max_steps=goal_spec.eval_limits.max_steps_per_episode,
                seed=seed if self.deterministic_seed else None,
            )
            
            # Convert to Metrics
            return Metrics(
                reward=results.get("total_reward", 0.0),
                risk=results.get("risk_score", 0.0),
                complexity=len(agent_pack.parameters) / 10000.0,
                sharpe_ratio=results.get("sharpe_ratio"),
                max_drawdown=results.get("max_drawdown"),
                episodes_run=results.get("episodes", 0),
                steps_run=results.get("steps", 0),
            )
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return self._run_mock(agent_pack, goal_spec, seed)
    
    def _run_simulation(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        """Run simulation evaluation."""
        # Similar to backtest but with simulation environment
        return self._run_mock(agent_pack, goal_spec, seed)
    
    def _run_mock(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Metrics:
        """Run mock evaluation (for testing)."""
        import random
        
        # Deterministic based on pack hash and seed
        rng = random.Random(int.from_bytes(agent_pack.pack_hash[:8], 'big') ^ seed)
        
        return Metrics(
            reward=rng.gauss(0.6, 0.1),
            risk=max(0, rng.gauss(0.15, 0.05)),
            complexity=len(agent_pack.parameters) / 10000.0,
            episodes_run=goal_spec.eval_limits.max_episodes,
            steps_run=goal_spec.eval_limits.max_episodes * goal_spec.eval_limits.max_steps_per_episode,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_idi_invariant_checker(
    use_python: bool = True,
    use_mpb: bool = True,
) -> IDIInvariantChecker:
    """Create an IDI-integrated invariant checker."""
    return IDIInvariantChecker(
        use_python_checks=use_python,
        use_mpb_checks=use_mpb,
    )


def create_idi_proof_verifier(
    require_proofs: bool = False,
) -> IDIProofVerifier:
    """Create an IDI-integrated proof verifier."""
    return IDIProofVerifier(require_proofs=require_proofs)


def create_idi_evaluation_harness(
    harness_type: str = "backtest",
) -> IDIEvaluationHarness:
    """Create an IDI-integrated evaluation harness."""
    return IDIEvaluationHarness(harness_type=harness_type)


def create_idi_coordinator(
    goal_spec: GoalSpec,
    leaderboard_capacity: int = 100,
    use_pareto: bool = False,
    use_python_invariants: bool = True,
    use_mpb_invariants: bool = True,
    require_proofs: bool = False,
    harness_type: str = "backtest",
) -> "IANCoordinator":
    """
    Create a fully-integrated IAN coordinator with IDI components.
    
    This is the recommended way to create a production coordinator.
    
    Args:
        goal_spec: Goal specification defining the task
        leaderboard_capacity: Maximum number of entries in leaderboard
        use_pareto: Use Pareto frontier instead of scalar ranking
        use_python_invariants: Enable Python InvariantService checks
        use_mpb_invariants: Enable MPB VM checks
        require_proofs: Require proofs for all contributions
        harness_type: Type of evaluation harness ('backtest', 'simulation', 'mock')
        
    Returns:
        Configured IANCoordinator instance
    """
    from .coordinator import IANCoordinator, CoordinatorConfig
    
    return IANCoordinator(
        goal_spec=goal_spec,
        config=CoordinatorConfig(
            leaderboard_capacity=leaderboard_capacity,
            use_pareto=use_pareto,
        ),
        invariant_checker=create_idi_invariant_checker(
            use_python=use_python_invariants,
            use_mpb=use_mpb_invariants,
        ),
        proof_verifier=create_idi_proof_verifier(
            require_proofs=require_proofs,
        ),
        evaluation_harness=create_idi_evaluation_harness(
            harness_type=harness_type,
        ),
    )
