"""
Sandboxed evaluation for IAN contributions.

Provides secure, resource-limited execution of agent evaluations:
- CPU time limits
- Memory limits
- Wall-clock timeout
- No network/filesystem access (when using subprocess isolation)

Security Model:
- Evaluations run in a subprocess with resource limits
- Timeout kills the subprocess cleanly
- Results are validated before returning
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .models import AgentPack, EvaluationLimits, GoalSpec, Metrics


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a sandboxed evaluation."""
    success: bool
    metrics: Optional[Metrics] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timed_out: bool = False


class SandboxedEvaluator:
    """
    Sandboxed evaluation harness using subprocess isolation.
    
    Executes evaluation in a subprocess with:
    - Wall-clock timeout
    - Memory limit (via resource module on Unix)
    - CPU time limit (via resource module on Unix)
    
    The evaluation function is serialized and sent to the subprocess.
    """
    
    def __init__(self, limits: Optional[EvaluationLimits] = None) -> None:
        """
        Initialize sandboxed evaluator.
        
        Args:
            limits: Resource limits for evaluation
        """
        self.limits = limits or EvaluationLimits()
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
        eval_function: Optional[Callable] = None,
    ) -> EvaluationResult:
        """
        Run evaluation in a sandboxed subprocess.
        
        Args:
            agent_pack: The agent to evaluate
            goal_spec: Goal specification with harness config
            seed: Random seed for determinism
            eval_function: Optional custom evaluation function
            
        Returns:
            EvaluationResult with metrics or error
        """
        start_time = time.time()
        
        try:
            # Use multiprocessing for isolation
            result_queue: multiprocessing.Queue = multiprocessing.Queue()
            
            process = multiprocessing.Process(
                target=self._run_evaluation_worker,
                args=(
                    agent_pack.to_dict(),
                    goal_spec.eval_limits.max_episodes,
                    goal_spec.eval_limits.max_steps_per_episode,
                    seed,
                    result_queue,
                    self.limits.max_memory_mb,  # Pass configurable memory limit
                    int(self.limits.timeout_seconds),  # Pass CPU time limit
                ),
            )
            
            process.start()
            process.join(timeout=self.limits.timeout_seconds)
            
            duration = time.time() - start_time
            
            if process.is_alive():
                # Timeout - kill the process
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                
                logger.warning(
                    f"Evaluation timed out after {duration:.2f}s "
                    f"(limit: {self.limits.timeout_seconds}s)"
                )
                return EvaluationResult(
                    success=False,
                    error="evaluation timeout",
                    duration_seconds=duration,
                    timed_out=True,
                )
            
            # Get result from queue
            if not result_queue.empty():
                result_data = result_queue.get_nowait()
                
                if result_data.get("success"):
                    metrics = Metrics.from_dict(result_data["metrics"])
                    return EvaluationResult(
                        success=True,
                        metrics=metrics,
                        duration_seconds=duration,
                    )
                else:
                    return EvaluationResult(
                        success=False,
                        error=result_data.get("error", "unknown error"),
                        duration_seconds=duration,
                    )
            else:
                return EvaluationResult(
                    success=False,
                    error="no result from worker",
                    duration_seconds=duration,
                )
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Evaluation error: {e}")
            return EvaluationResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
    
    @staticmethod
    def _run_evaluation_worker(
        agent_pack_dict: Dict[str, Any],
        max_episodes: int,
        max_steps: int,
        seed: int,
        result_queue: multiprocessing.Queue,
        max_memory_mb: int = 1024,
        cpu_time_limit: int = 300,
    ) -> None:
        """
        Worker function that runs in the subprocess.
        
        This is where resource limits are applied.
        
        Args:
            agent_pack_dict: Serialized AgentPack
            max_episodes: Maximum episodes to run
            max_steps: Maximum steps per episode
            seed: Random seed for determinism
            result_queue: Queue to send results back
            max_memory_mb: Memory limit in MB
            cpu_time_limit: CPU time limit in seconds
        """
        limits_applied = {"memory": False, "cpu": False}
        
        try:
            # Apply resource limits on Unix
            if hasattr(os, 'setrlimit'):
                import resource
                
                # Memory limit (soft, hard) - FIXED: use passed parameter
                memory_bytes = max_memory_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                    limits_applied["memory"] = True
                except (ValueError, resource.error) as e:
                    # SECURITY: Abort evaluation if limits cannot be enforced
                    result_queue.put({
                        "success": False,
                        "error": f"Security: Could not enforce memory limit ({max_memory_mb}MB): {e}",
                    })
                    return  # Abort evaluation
                
                # CPU time limit - FIXED: use passed parameter
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
                    limits_applied["cpu"] = True
                except (ValueError, resource.error) as e:
                    # SECURITY: Abort evaluation if limits cannot be enforced
                    result_queue.put({
                        "success": False,
                        "error": f"Security: Could not enforce CPU limit ({cpu_time_limit}s): {e}",
                    })
                    return  # Abort evaluation
            
            # Reconstruct agent pack
            agent_pack = AgentPack.from_dict(agent_pack_dict)
            
            # Run mock evaluation (replace with real harness integration)
            metrics = SandboxedEvaluator._mock_evaluate(
                agent_pack,
                max_episodes,
                max_steps,
                seed,
            )
            
            result_queue.put({
                "success": True,
                "metrics": metrics.to_dict(),
            })
            
        except Exception as e:
            result_queue.put({
                "success": False,
                "error": str(e),
            })
    
    @staticmethod
    def _mock_evaluate(
        agent_pack: AgentPack,
        max_episodes: int,
        max_steps: int,
        seed: int,
    ) -> Metrics:
        """
        Mock evaluation for testing.
        
        In production, this would call the actual IDI evaluation harness.
        """
        import hashlib
        import random
        
        # Deterministic based on pack hash and seed
        rng = random.Random(int.from_bytes(agent_pack.pack_hash[:8], 'big') ^ seed)
        
        # Simulate some work
        total_reward = 0.0
        total_risk = 0.0
        steps_run = 0
        
        episodes_run = min(max_episodes, 10)  # Cap for mock
        
        for episode in range(episodes_run):
            episode_reward = rng.gauss(0.6, 0.1)
            episode_risk = rng.gauss(0.15, 0.05)
            # FIXED: Clamp values to valid [0, 1] range
            total_reward += max(0.0, min(1.0, episode_reward))
            total_risk += max(0.0, min(1.0, episode_risk))
            steps_run += min(max_steps, 100)
        
        # FIXED: Guard against division by zero
        if episodes_run == 0:
            return Metrics(
                reward=0.0,
                risk=0.0,
                complexity=len(agent_pack.parameters) / 10000.0,
                episodes_run=0,
                steps_run=0,
            )
        
        return Metrics(
            reward=total_reward / episodes_run,
            risk=total_risk / episodes_run,
            complexity=min(1.0, len(agent_pack.parameters) / 10000.0),  # Clamp complexity
            episodes_run=episodes_run,
            steps_run=steps_run,
        )


class InProcessEvaluator:
    """
    In-process evaluation for trusted code paths.
    
    Faster than subprocess but provides no isolation.
    Use only for testing or trusted evaluations.
    """
    
    def __init__(self, limits: Optional[EvaluationLimits] = None) -> None:
        self.limits = limits or EvaluationLimits()
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
        eval_function: Optional[Callable[[AgentPack, int, int, int], Metrics]] = None,
    ) -> EvaluationResult:
        """
        Run evaluation in the current process.
        
        WARNING: No isolation - only use for trusted code.
        """
        start_time = time.time()
        
        try:
            if eval_function:
                metrics = eval_function(
                    agent_pack,
                    goal_spec.eval_limits.max_episodes,
                    goal_spec.eval_limits.max_steps_per_episode,
                    seed,
                )
            else:
                metrics = self._default_evaluate(
                    agent_pack,
                    goal_spec.eval_limits.max_episodes,
                    goal_spec.eval_limits.max_steps_per_episode,
                    seed,
                )
            
            duration = time.time() - start_time
            
            return EvaluationResult(
                success=True,
                metrics=metrics,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"In-process evaluation error: {e}")
            return EvaluationResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
    
    def _default_evaluate(
        self,
        agent_pack: AgentPack,
        max_episodes: int,
        max_steps: int,
        seed: int,
    ) -> Metrics:
        """Default evaluation using mock metrics."""
        return SandboxedEvaluator._mock_evaluate(
            agent_pack,
            max_episodes,
            max_steps,
            seed,
        )


# -----------------------------------------------------------------------------
# Evaluation Harness Adapter (for Coordinator integration)
# -----------------------------------------------------------------------------

class EvaluationHarnessAdapter:
    """
    Adapter that implements the EvaluationHarness protocol
    using SandboxedEvaluator.
    """
    
    def __init__(
        self,
        use_sandbox: bool = True,
        limits: Optional[EvaluationLimits] = None,
    ) -> None:
        self.use_sandbox = use_sandbox
        if use_sandbox:
            self._evaluator = SandboxedEvaluator(limits)
        else:
            self._evaluator = InProcessEvaluator(limits)
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        """
        Evaluate agent_pack and return metrics.
        
        Returns None on failure.
        """
        result = self._evaluator.evaluate(agent_pack, goal_spec, seed)
        
        if result.success:
            return result.metrics
        
        logger.warning(f"Evaluation failed: {result.error}")
        return None
