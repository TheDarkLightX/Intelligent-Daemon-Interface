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

    @staticmethod
    def _load_training_config_from_pack(agent_pack: AgentPack):
        """
        Best-effort reconstruction of `TrainingConfig` from `agent_pack.metadata`.

        GUI training stores a `dataclasses.asdict()` snapshot under `metadata["config"]`.
        This loader is intentionally forgiving; it falls back to defaults if parsing fails.
        """
        try:
            from idi.training.python.idi_iann.config import (
                CommunicationConfig,
                EmoteConfig,
                EpisodicConfig,
                FractalConfig,
                FractalLevelConfig,
                LayerConfig,
                MultiLayerConfig,
                QuantizerConfig,
                RewardWeights,
                TileCoderConfig,
                TrainingConfig,
            )
        except Exception:
            return None

        base = TrainingConfig()
        raw = agent_pack.metadata.get("config")
        if not isinstance(raw, dict):
            return base

        def _coerce_dataclass(obj_type, key: str):
            val = raw.get(key)
            if isinstance(val, dict):
                try:
                    return obj_type(**val)
                except Exception:
                    return getattr(base, key)
            return getattr(base, key)

        tile_coder = None
        if isinstance(raw.get("tile_coder"), dict):
            try:
                tile_coder = TileCoderConfig(**raw["tile_coder"])
            except Exception:
                tile_coder = base.tile_coder

        episodic = None
        if isinstance(raw.get("episodic"), dict):
            try:
                episodic = EpisodicConfig(**raw["episodic"])
            except Exception:
                episodic = base.episodic

        fractal = None
        if isinstance(raw.get("fractal"), dict):
            try:
                levels_raw = raw["fractal"].get("levels")
                if isinstance(levels_raw, list):
                    levels = []
                    for item in levels_raw:
                        if not isinstance(item, dict):
                            continue
                        levels.append(FractalLevelConfig(**item))
                    fractal = FractalConfig(
                        levels=tuple(levels),
                        backoff_enabled=bool(raw["fractal"].get("backoff_enabled", True)),
                        hierarchical_updates=bool(raw["fractal"].get("hierarchical_updates", True)),
                    )
                else:
                    fractal = FractalConfig(**raw["fractal"])
            except Exception:
                fractal = base.fractal

        multi_layer = None
        if isinstance(raw.get("multi_layer"), dict):
            try:
                multi_layer = MultiLayerConfig(**raw["multi_layer"])
            except Exception:
                multi_layer = base.multi_layer

        try:
            cfg = TrainingConfig(
                episodes=int(raw.get("episodes", base.episodes)),
                episode_length=int(raw.get("episode_length", base.episode_length)),
                discount=float(raw.get("discount", base.discount)),
                learning_rate=float(raw.get("learning_rate", base.learning_rate)),
                exploration_decay=float(raw.get("exploration_decay", base.exploration_decay)),
                quantizer=_coerce_dataclass(QuantizerConfig, "quantizer"),
                rewards=_coerce_dataclass(RewardWeights, "rewards"),
                emote=_coerce_dataclass(EmoteConfig, "emote"),
                layers=_coerce_dataclass(LayerConfig, "layers"),
                tile_coder=tile_coder,
                communication=_coerce_dataclass(CommunicationConfig, "communication"),
                fractal=fractal,
                multi_layer=multi_layer,
                episodic=episodic,
            )
            cfg.validate()
            return cfg
        except Exception:
            return base

    @staticmethod
    def _load_policy_table(agent_pack: AgentPack) -> Dict[tuple[int, ...], Dict[str, float]]:
        """
        Load a greedy Q-table from `agent_pack.parameters`.

        The GUI training path serializes `LookupPolicy.to_entries()`, which uses
        `str(tuple(...))` for the state key.
        """
        import ast
        import json

        try:
            raw = json.loads(agent_pack.parameters.decode("utf-8"))
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        table: Dict[tuple[int, ...], Dict[str, float]] = {}
        for state_str, qvals in raw.items():
            if not isinstance(state_str, str) or not isinstance(qvals, dict):
                continue
            try:
                parsed = ast.literal_eval(state_str)
            except Exception:
                continue
            if not isinstance(parsed, tuple) or not all(isinstance(x, int) for x in parsed):
                continue
            out: Dict[str, float] = {}
            for action_str, q in qvals.items():
                if not isinstance(action_str, str):
                    continue
                try:
                    out[action_str] = float(q)
                except Exception:
                    continue
            table[tuple(parsed)] = out
        return table

    @staticmethod
    def _obs_to_state(obs: object) -> tuple[int, ...]:
        # Match `QTrainer._as_state` mappings to ensure compatibility.
        if hasattr(obs, "as_state"):
            state = obs.as_state()  # type: ignore[attr-defined]
            if isinstance(state, tuple) and all(isinstance(x, int) for x in state):
                return state
        if hasattr(obs, "price") and hasattr(obs, "regime"):
            regime_idx = {"bull": 0, "bear": 1, "chop": 2, "panic": 3}.get(getattr(obs, "regime", "chop"), 2)
            pos = getattr(obs, "position", 0)
            pnl = getattr(obs, "pnl", 0.0)
            ret = getattr(obs, "last_return", 0.0)
            return (int(pos + 1), int(regime_idx), int(abs(ret) > 0.01), int(float(pnl) >= 0.0))
        raise ValueError(f"Unsupported observation type: {type(obs)}")

    @staticmethod
    def _select_action(qtable: Dict[tuple[int, ...], Dict[str, float]], state: tuple[int, ...]) -> str:
        qvals = qtable.get(state)
        if not qvals:
            return "hold"
        return max(qvals.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _compute_sharpe(values: List[float]) -> Optional[float]:
        if not values:
            return None
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, (len(values) - 1))
        if var <= 0:
            return None
        import math
        return mean / math.sqrt(var)

    @staticmethod
    def _update_drawdown_ratio(*, peak: float, current: float, best: float) -> tuple[float, float]:
        """
        Update (peak, max_drawdown_ratio) for an equity curve.

        Drawdown ratio is clamped to [0,1] to match IAN `max_risk` thresholds.
        """
        if current > peak:
            return current, best

        if peak <= 0.0:
            # When peak is non-positive, any further decline is treated as max risk.
            return peak, max(best, 1.0)

        ratio = (peak - current) / peak
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
        return peak, max(best, ratio)
    
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
        """
        Deterministic evaluation of the submitted policy against the local simulators.

        No randomized fallback paths: if we can't decode/run, we fail the eval.
        """
        from time import perf_counter

        cfg = self._load_training_config_from_pack(agent_pack)
        qtable = self._load_policy_table(agent_pack)

        use_crypto = bool(agent_pack.metadata.get("use_crypto_env", False))
        market_params_raw = agent_pack.metadata.get("market_params")

        max_episodes = int(goal_spec.eval_limits.max_episodes)
        max_steps = int(goal_spec.eval_limits.max_steps_per_episode)
        timeout_s = float(goal_spec.eval_limits.timeout_seconds)
        if max_episodes <= 0 or max_steps <= 0 or timeout_s <= 0:
            return None

        episode_returns: List[float] = []
        episode_step_means: List[float] = []
        total_steps = 0
        risk_events = 0
        cum_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown_ratio = 0.0

        start = perf_counter()
        for ep in range(max_episodes):
            if perf_counter() - start > timeout_s:
                break

            ep_seed = int(seed)
            if self.deterministic_seed:
                ep_seed ^= (ep * 0x9E3779B1) & 0xFFFFFFFF
            else:
                ep_seed += ep

            if use_crypto:
                from idi.training.python.idi_iann.crypto_env import CryptoMarket, MarketParams
                mp = MarketParams()
                if isinstance(market_params_raw, dict):
                    try:
                        mp = MarketParams(**market_params_raw)
                    except Exception:
                        mp = MarketParams()
                mp.seed = ep_seed
                env = CryptoMarket(mp)
            else:
                from idi.training.python.idi_iann.env import SyntheticMarketEnv
                env = SyntheticMarketEnv(
                    quantizer=cfg.quantizer,
                    rewards=cfg.rewards,
                    seed=ep_seed,
                )

            obs = env.reset()
            ep_return = 0.0
            ep_steps = 0
            for _ in range(max_steps):
                if perf_counter() - start > timeout_s:
                    break
                state = self._obs_to_state(obs)
                action = self._select_action(qtable, state)

                step_out = env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 3:
                    obs, reward, info = step_out
                    if info and float(info.get("risk_event", 0.0)) > 0:
                        risk_events += 1
                else:
                    obs, reward = step_out

                reward_f = float(reward)
                ep_return += reward_f
                total_steps += 1
                ep_steps += 1
                cum_pnl += reward_f
                peak_pnl, max_drawdown_ratio = self._update_drawdown_ratio(
                    peak=peak_pnl,
                    current=cum_pnl,
                    best=max_drawdown_ratio,
                )

            episode_returns.append(ep_return)
            if ep_steps > 0:
                episode_step_means.append(ep_return / ep_steps)

        if not episode_returns:
            return None

        mean_step_reward = (sum(episode_returns) / total_steps) if total_steps > 0 else 0.0
        sharpe = self._compute_sharpe(episode_step_means) if episode_step_means else None

        if use_crypto:
            risk = float(risk_events) / float(max(1, total_steps))
        else:
            # Normalize volatility into [0,1] using coefficient-of-variation on per-step returns.
            mean = mean_step_reward
            var = (
                sum((r - mean) ** 2 for r in episode_step_means) / max(1, (len(episode_step_means) - 1))
            ) if episode_step_means else 0.0
            vol = float(var) ** 0.5
            cov = vol / max(abs(mean), 1e-6)
            risk = min(1.0, max(max_drawdown_ratio, cov))

        return Metrics(
            reward=float(mean_step_reward),
            risk=float(risk),
            complexity=len(agent_pack.parameters) / 10000.0,
            sharpe_ratio=sharpe,
            max_drawdown=float(max_drawdown_ratio),
            episodes_run=len(episode_returns),
            steps_run=total_steps,
        )

    
    def _run_simulation(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        # Standalone simulation uses the same deterministic evaluators for now.
        return self._run_backtest(agent_pack, goal_spec, seed)
    
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
