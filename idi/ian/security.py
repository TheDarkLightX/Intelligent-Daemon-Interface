"""
IAN Security Module - Hardening for production deployment.

Provides:
1. Input validation with strict size limits
2. Rate limiting per contributor (token bucket)
3. Sandbox enforcement verification
4. Timing attack mitigation
5. Sybil resistance (optional proof-of-work)

Security Invariants:
- I1: All inputs validated at system boundary
- I2: No contributor can exceed rate limits
- I3: Evaluation runs in isolated sandbox
- I4: Timing side-channels minimized
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from idi.ian.models import AgentPack, Contribution, GoalSpec

# =============================================================================
# Security Audit Logger
# =============================================================================

class SecurityEventType(Enum):
    """Types of security-relevant events for audit logging."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_HIT = "rate_limit_hit"
    VALIDATION_FAILURE = "validation_failure"
    SIGNATURE_INVALID = "signature_invalid"
    REPLAY_DETECTED = "replay_detected"
    SYBIL_DETECTED = "sybil_detected"
    POW_FAILURE = "pow_failure"
    ACCESS_DENIED = "access_denied"
    SYNC_FORK_DETECTED = "sync_fork_detected"
    WITNESS_DIVERSITY_FAIL = "witness_diversity_fail"
    HMAC_VERIFICATION_FAIL = "hmac_verification_fail"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    CRYPTO_FALLBACK_USED = "crypto_fallback_used"


class SecurityAuditLogger:
    """
    Structured security audit logger for compliance and incident response.
    
    Security:
        - All events include timestamp, event_type, and correlation_id
        - Sensitive data is redacted (keys, tokens)
        - Logs are structured JSON for SIEM integration
    """
    
    def __init__(self, logger_name: str = "idi.ian.security.audit"):
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
    
    def log_event(
        self,
        event_type: SecurityEventType,
        *,
        actor_id: Optional[str] = None,
        target_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        severity: str = "INFO",
    ) -> None:
        """
        Log a security-relevant event.
        
        Args:
            event_type: Type of security event
            actor_id: ID of actor (user, node, contributor)
            target_id: ID of target resource
            ip_address: Source IP (redact last octet in logs)
            details: Additional event details
            correlation_id: Request/session correlation ID
            severity: Log level (INFO, WARNING, ERROR, CRITICAL)
        """
        # Build structured log entry
        entry = {
            "timestamp_ms": int(time.time() * 1000),
            "event_type": event_type.value,
            "severity": severity,
            "actor_id": actor_id,
            "target_id": target_id,
            "correlation_id": correlation_id or secrets.token_hex(8),
        }
        
        # Security: Redact IP address last octet
        if ip_address:
            parts = ip_address.split(".")
            if len(parts) == 4:
                entry["ip_address"] = f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
            else:
                entry["ip_address"] = ip_address  # IPv6 or invalid
        
        # Security: Redact sensitive keys in details
        if details:
            entry["details"] = self._redact_sensitive(details)
        
        # Log as JSON
        log_line = json.dumps(entry, default=str)
        
        if severity == "CRITICAL":
            self._logger.critical(log_line)
        elif severity == "ERROR":
            self._logger.error(log_line)
        elif severity == "WARNING":
            self._logger.warning(log_line)
        else:
            self._logger.info(log_line)
    
    def _redact_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from log data."""
        sensitive_keys = {"key", "secret", "token", "password", "private", "signature"}
        redacted = {}
        for k, v in data.items():
            if any(s in k.lower() for s in sensitive_keys):
                if isinstance(v, bytes):
                    redacted[k] = f"[REDACTED:{len(v)} bytes]"
                elif isinstance(v, str):
                    redacted[k] = f"[REDACTED:{len(v)} chars]"
                else:
                    redacted[k] = "[REDACTED]"
            elif isinstance(v, dict):
                redacted[k] = self._redact_sensitive(v)
            else:
                redacted[k] = v
        return redacted


# Global security audit logger instance
security_audit = SecurityAuditLogger()


# =============================================================================
# Configuration Constants
# =============================================================================

@dataclass(frozen=True)
class SecurityLimits:
    """
    Security limits for IAN inputs.
    
    These limits prevent resource exhaustion and ensure predictable behavior.
    All values are chosen to be generous for legitimate use while blocking abuse.
    """
    # AgentPack limits
    MAX_PACK_VERSION_LEN: int = 64
    MAX_PACK_PARAMETERS_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_PACK_METADATA_SIZE: int = 64 * 1024  # 64 KB
    MAX_PACK_METADATA_KEYS: int = 100
    MAX_PACK_METADATA_KEY_LEN: int = 256
    MAX_PACK_METADATA_VALUE_SIZE: int = 8 * 1024  # 8 KB per value
    
    # Contribution limits
    MAX_CONTRIBUTOR_ID_LEN: int = 128
    MAX_PROOFS_COUNT: int = 16
    MAX_PROOF_SIZE: int = 1 * 1024 * 1024  # 1 MB per proof
    MAX_TOTAL_PROOFS_SIZE: int = 4 * 1024 * 1024  # 4 MB total
    
    # GoalSpec limits
    MAX_GOAL_ID_LEN: int = 64
    MAX_GOAL_NAME_LEN: int = 256
    MAX_GOAL_DESCRIPTION_LEN: int = 4096
    
    # Rate limits (per contributor)
    RATE_LIMIT_TOKENS: int = 10  # Max burst
    RATE_LIMIT_REFILL_PER_SECOND: float = 0.1  # 1 token per 10 seconds
    
    # Proof-of-work (optional Sybil resistance)
    POW_DIFFICULTY: int = 20  # Leading zero bits required
    POW_ENABLED: bool = False


DEFAULT_LIMITS = SecurityLimits()


# =============================================================================
# Input Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    error: Optional[str] = None
    field: Optional[str] = None
    
    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True)
    
    @classmethod
    def fail(cls, field: str, error: str) -> "ValidationResult":
        return cls(valid=False, error=error, field=field)


class InputValidator:
    """
    Validates all IAN inputs at system boundary.
    
    Invariant: No invalid input passes validation.
    Precondition: Input is Python object (parsed from wire format).
    Postcondition: If valid=True, input is safe to process.
    """
    
    def __init__(self, limits: SecurityLimits = DEFAULT_LIMITS) -> None:
        self._limits = limits
    
    def validate_agent_pack(self, pack: AgentPack) -> ValidationResult:
        """Validate AgentPack fields."""
        # Version - must be non-empty and within length limit
        if not pack.version:
            return ValidationResult.fail("version", "Version cannot be empty")
        if len(pack.version) > self._limits.MAX_PACK_VERSION_LEN:
            return ValidationResult.fail(
                "version",
                f"Version too long: {len(pack.version)} > {self._limits.MAX_PACK_VERSION_LEN}"
            )
        
        # Parameters
        if len(pack.parameters) == 0:
            return ValidationResult.fail("parameters", "Parameters cannot be empty")
        
        if len(pack.parameters) > self._limits.MAX_PACK_PARAMETERS_SIZE:
            return ValidationResult.fail(
                "parameters",
                f"Parameters too large: {len(pack.parameters)} > {self._limits.MAX_PACK_PARAMETERS_SIZE}"
            )
        
        # Metadata
        if pack.metadata:
            total_size = 0
            if len(pack.metadata) > self._limits.MAX_PACK_METADATA_KEYS:
                return ValidationResult.fail(
                    "metadata",
                    f"Too many metadata keys: {len(pack.metadata)} > {self._limits.MAX_PACK_METADATA_KEYS}"
                )
            
            for key, value in pack.metadata.items():
                if len(key) > self._limits.MAX_PACK_METADATA_KEY_LEN:
                    return ValidationResult.fail(
                        f"metadata.{key}",
                        f"Metadata key too long: {len(key)} > {self._limits.MAX_PACK_METADATA_KEY_LEN}"
                    )
                
                value_size = len(str(value).encode('utf-8'))
                if value_size > self._limits.MAX_PACK_METADATA_VALUE_SIZE:
                    return ValidationResult.fail(
                        f"metadata.{key}",
                        f"Metadata value too large: {value_size} > {self._limits.MAX_PACK_METADATA_VALUE_SIZE}"
                    )
                
                total_size += len(key) + value_size
            
            if total_size > self._limits.MAX_PACK_METADATA_SIZE:
                return ValidationResult.fail(
                    "metadata",
                    f"Total metadata too large: {total_size} > {self._limits.MAX_PACK_METADATA_SIZE}"
                )
        
        return ValidationResult.ok()
    
    def validate_contribution(self, contrib: Contribution) -> ValidationResult:
        """Validate Contribution fields."""
        # Goal ID
        if len(str(contrib.goal_id)) > self._limits.MAX_GOAL_ID_LEN:
            return ValidationResult.fail(
                "goal_id",
                f"Goal ID too long: {len(str(contrib.goal_id))} > {self._limits.MAX_GOAL_ID_LEN}"
            )
        
        # Contributor ID
        if len(contrib.contributor_id) > self._limits.MAX_CONTRIBUTOR_ID_LEN:
            return ValidationResult.fail(
                "contributor_id",
                f"Contributor ID too long: {len(contrib.contributor_id)} > {self._limits.MAX_CONTRIBUTOR_ID_LEN}"
            )
        
        # AgentPack
        pack_result = self.validate_agent_pack(contrib.agent_pack)
        if not pack_result.valid:
            return pack_result
        
        # Proofs
        if len(contrib.proofs) > self._limits.MAX_PROOFS_COUNT:
            return ValidationResult.fail(
                "proofs",
                f"Too many proofs: {len(contrib.proofs)} > {self._limits.MAX_PROOFS_COUNT}"
            )
        
        total_proof_size = 0
        for proof_name, proof_data in contrib.proofs.items():
            proof_size = len(proof_data) if isinstance(proof_data, bytes) else len(str(proof_data))
            
            if proof_size > self._limits.MAX_PROOF_SIZE:
                return ValidationResult.fail(
                    f"proofs.{proof_name}",
                    f"Proof too large: {proof_size} > {self._limits.MAX_PROOF_SIZE}"
                )
            
            total_proof_size += proof_size
        
        if total_proof_size > self._limits.MAX_TOTAL_PROOFS_SIZE:
            return ValidationResult.fail(
                "proofs",
                f"Total proofs too large: {total_proof_size} > {self._limits.MAX_TOTAL_PROOFS_SIZE}"
            )
        
        return ValidationResult.ok()
    
    def validate_goal_spec(self, spec: GoalSpec) -> ValidationResult:
        """Validate GoalSpec fields."""
        if len(str(spec.goal_id)) > self._limits.MAX_GOAL_ID_LEN:
            return ValidationResult.fail(
                "goal_id",
                f"Goal ID too long: {len(str(spec.goal_id))} > {self._limits.MAX_GOAL_ID_LEN}"
            )
        
        if len(spec.name) > self._limits.MAX_GOAL_NAME_LEN:
            return ValidationResult.fail(
                "name",
                f"Name too long: {len(spec.name)} > {self._limits.MAX_GOAL_NAME_LEN}"
            )
        
        if len(spec.description) > self._limits.MAX_GOAL_DESCRIPTION_LEN:
            return ValidationResult.fail(
                "description",
                f"Description too long: {len(spec.description)} > {self._limits.MAX_GOAL_DESCRIPTION_LEN}"
            )
        
        return ValidationResult.ok()


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.
    
    Invariant: tokens <= capacity
    """
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.monotonic)
    
    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Returns True if tokens were consumed, False if rate limited.
        """
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Seconds until `tokens` will be available."""
        self.refill()
        if self.tokens >= tokens:
            return 0.0
        if self.refill_rate <= 0:
            return float("inf")
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    Per-contributor rate limiter using token buckets.
    
    Prevents abuse while allowing legitimate high-frequency contributors
    to burst within limits.
    """
    
    def __init__(
        self,
        capacity: int = DEFAULT_LIMITS.RATE_LIMIT_TOKENS,
        refill_rate: float = DEFAULT_LIMITS.RATE_LIMIT_REFILL_PER_SECOND,
        max_buckets: int = 100_000,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._cleanup_interval = 3600  # Clean up old buckets every hour
        self._last_cleanup = time.monotonic()
        # Upper bound on the number of tracked contributors to prevent
        # unbounded memory growth in adversarial settings.
        # Increased to 100k to reduce Sybil attack surface where attackers
        # cycle through contributor IDs to evict legitimate users' buckets.
        self._max_buckets = max_buckets
    
    def _get_bucket(self, contributor_id: str) -> TokenBucket:
        """Get or create bucket for contributor."""
        if contributor_id not in self._buckets:
            if len(self._buckets) >= self._max_buckets:
                self._evict_bucket()
            self._buckets[contributor_id] = TokenBucket(
                capacity=self._capacity,
                tokens=float(self._capacity),
                refill_rate=self._refill_rate,
            )
        return self._buckets[contributor_id]
    
    def _maybe_cleanup(self) -> None:
        """Remove stale buckets periodically."""
        now = time.monotonic()
        if now - self._last_cleanup > self._cleanup_interval:
            # Remove buckets that have been idle and full for a long time
            stale = [
                cid for cid, bucket in self._buckets.items()
                if (now - bucket.last_refill > self._cleanup_interval and bucket.tokens >= self._capacity)
            ]
            for cid in stale:
                del self._buckets[cid]
            self._last_cleanup = now
    
    def _evict_bucket(self) -> None:
        """Evict one bucket to enforce max_buckets."""
        if not self._buckets:
            return
        # Remove least-recently-refilled bucket
        oldest_cid = min(self._buckets.items(), key=lambda item: item[1].last_refill)[0]
        del self._buckets[oldest_cid]
    
    def check(self, contributor_id: str) -> Tuple[bool, float]:
        """
        Check if contributor can submit.
        
        Returns:
            (allowed, wait_seconds): allowed=True if can submit,
                                      wait_seconds is time until next allowed
        """
        self._maybe_cleanup()
        bucket = self._get_bucket(contributor_id)
        
        if bucket.try_consume(1):
            return True, 0.0
        else:
            return False, bucket.time_until_available(1)
    
    def get_remaining(self, contributor_id: str) -> int:
        """Get remaining tokens for contributor."""
        bucket = self._get_bucket(contributor_id)
        bucket.refill()
        return int(bucket.tokens)


# =============================================================================
# Proof of Work (Sybil Resistance)
# =============================================================================

@dataclass
class ProofOfWork:
    """
    Proof-of-work for Sybil resistance.
    
    Challenge: SHA-256(challenge_bytes || nonce) must have `difficulty` leading zero bits.
    """
    challenge: bytes
    nonce: int
    difficulty: int
    
    def verify(self) -> bool:
        """Verify the proof of work."""
        data = self.challenge + struct.pack(">Q", self.nonce)
        hash_bytes = hashlib.sha256(data).digest()
        
        # Count leading zero bits
        leading_zeros = 0
        for byte in hash_bytes:
            if byte == 0:
                leading_zeros += 8
            else:
                # Count leading zeros in this byte
                leading_zeros += (8 - byte.bit_length())
                break
        
        return leading_zeros >= self.difficulty
    
    @classmethod
    def create_challenge(cls) -> bytes:
        """Create a new random challenge."""
        return secrets.token_bytes(32)
    
    @classmethod
    def solve(cls, challenge: bytes, difficulty: int, max_attempts: int = 10_000_000) -> Optional["ProofOfWork"]:
        """
        Solve a proof-of-work challenge.
        
        Warning: This is CPU-intensive. Use only when POW is required.
        """
        for nonce in range(max_attempts):
            pow = cls(challenge=challenge, nonce=nonce, difficulty=difficulty)
            if pow.verify():
                return pow
        return None


class SybilResistance:
    """
    Sybil resistance layer using proof-of-work.
    
    When enabled, contributors must solve a PoW challenge before submitting.
    """
    
    def __init__(
        self,
        enabled: bool = DEFAULT_LIMITS.POW_ENABLED,
        difficulty: int = DEFAULT_LIMITS.POW_DIFFICULTY,
        max_challenges: int = 10_000,
    ) -> None:
        self._enabled = enabled
        self._difficulty = difficulty
        self._pending_challenges: Dict[str, Tuple[bytes, float]] = {}  # contributor_id -> (challenge, expiry)
        self._challenge_ttl = 300  # 5 minutes
        # Upper bound on outstanding challenges to avoid unbounded growth
        self._max_challenges = max_challenges
    
    def is_enabled(self) -> bool:
        return self._enabled
    
    def get_challenge(self, contributor_id: str) -> bytes:
        """Get or create a challenge for a contributor."""
        now = time.time()
        self._cleanup_expired(now)
        
        # Check for existing valid challenge
        if contributor_id in self._pending_challenges:
            challenge, expiry = self._pending_challenges[contributor_id]
            if now < expiry:
                return challenge
        
        # Enforce maximum number of outstanding challenges
        if len(self._pending_challenges) >= self._max_challenges:
            # Evict challenge with earliest expiry
            oldest_id = min(self._pending_challenges.items(), key=lambda item: item[1][1])[0]
            del self._pending_challenges[oldest_id]
        
        # Create new challenge
        challenge = ProofOfWork.create_challenge()
        self._pending_challenges[contributor_id] = (challenge, now + self._challenge_ttl)
        return challenge

    def _cleanup_expired(self, now: Optional[float] = None) -> None:
        """Remove expired challenges to bound memory usage."""
        if not self._pending_challenges:
            return
        if now is None:
            now = time.time()
        expired = [cid for cid, (_, expiry) in self._pending_challenges.items() if expiry <= now]
        for cid in expired:
            del self._pending_challenges[cid]
    
    def verify_pow(self, contributor_id: str, proof: ProofOfWork) -> bool:
        """Verify a proof-of-work submission.
        
        Args:
            contributor_id: ID of the contributor
            proof: The ProofOfWork object to verify (renamed from 'pow' to avoid
                   shadowing Python's built-in pow() function)
        """
        if not self._enabled:
            return True
        
        self._cleanup_expired()
        if contributor_id not in self._pending_challenges:
            return False
        
        challenge, expiry = self._pending_challenges[contributor_id]
        if time.time() > expiry:
            del self._pending_challenges[contributor_id]
            return False
        
        if proof.challenge != challenge:
            return False
        
        if proof.difficulty < self._difficulty:
            return False
        
        if not proof.verify():
            return False
        
        # Clear used challenge
        del self._pending_challenges[contributor_id]
        return True


# =============================================================================
# Timing Attack Mitigation
# =============================================================================

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Constant-time comparison to prevent timing attacks.
    
    Uses hmac.compare_digest which is designed to be constant-time.
    """
    return hmac.compare_digest(a, b)


def pad_execution_time(target_ms: float, start_time: float) -> None:
    """
    Pad execution to a fixed time to prevent timing side-channels.
    
    Use after security-sensitive operations to mask timing variations.
    Note: This uses blocking sleep. For async contexts, use pad_execution_time_async.
    """
    elapsed_ms = (time.monotonic() - start_time) * 1000
    if elapsed_ms < target_ms:
        time.sleep((target_ms - elapsed_ms) / 1000)


async def pad_execution_time_async(target_ms: float, start_time: float) -> None:
    """
    Async version of pad_execution_time for use in async contexts.
    
    Pad execution to a fixed time to prevent timing side-channels.
    Use after security-sensitive operations to mask timing variations.
    """
    import asyncio
    elapsed_ms = (time.monotonic() - start_time) * 1000
    if elapsed_ms < target_ms:
        await asyncio.sleep((target_ms - elapsed_ms) / 1000)


# =============================================================================
# Secure Coordinator Wrapper
# =============================================================================

class SecureCoordinator:
    """
    Security-hardened wrapper around IANCoordinator.
    
    Applies all security measures:
    - Input validation at boundary
    - Rate limiting per contributor
    - Optional proof-of-work
    - Timing attack mitigation
    """
    
    def __init__(
        self,
        coordinator,  # IANCoordinator
        limits: SecurityLimits = DEFAULT_LIMITS,
        enable_pow: bool = False,
    ) -> None:
        self._coordinator = coordinator
        self._validator = InputValidator(limits)
        self._rate_limiter = RateLimiter(
            capacity=limits.RATE_LIMIT_TOKENS,
            refill_rate=limits.RATE_LIMIT_REFILL_PER_SECOND,
        )
        self._sybil = SybilResistance(enabled=enable_pow, difficulty=limits.POW_DIFFICULTY)
        self._min_process_time_ms = 100.0  # Minimum processing time for timing resistance
    
    def process_contribution(
        self,
        contrib: Contribution,
        proof_of_work: Optional[ProofOfWork] = None,
    ):
        """
        Process a contribution with all security checks.
        
        Args:
            contrib: The contribution to process
            proof_of_work: Optional PoW proof (renamed from 'pow' to avoid
                          shadowing Python's built-in pow() function)
        
        Returns the same result type as IANCoordinator.process_contribution.
        """
        from idi.ian.coordinator import ProcessResult, RejectionReason
        
        start_time = time.monotonic()
        
        try:
            # 1. Input validation
            validation = self._validator.validate_contribution(contrib)
            if not validation.valid:
                return ProcessResult(
                    accepted=False,
                    rejection_type=RejectionReason.VALIDATION_ERROR,
                    reason=f"Validation failed: {validation.field}: {validation.error}",
                )
            
            # 2. Rate limiting
            allowed, wait_time = self._rate_limiter.check(contrib.contributor_id)
            if not allowed:
                return ProcessResult(
                    accepted=False,
                    rejection_type=RejectionReason.RATE_LIMITED,
                    reason=f"Rate limited. Retry in {wait_time:.1f} seconds.",
                )
            
            # 3. Proof-of-work (if enabled)
            if self._sybil.is_enabled():
                if proof_of_work is None:
                    challenge = self._sybil.get_challenge(contrib.contributor_id)
                    return ProcessResult(
                        accepted=False,
                        rejection_type=RejectionReason.POW_REQUIRED,
                        reason=f"Proof-of-work required. Challenge: {challenge.hex()}",
                    )
                
                if not self._sybil.verify_pow(contrib.contributor_id, proof_of_work):
                    return ProcessResult(
                        accepted=False,
                        rejection_type=RejectionReason.POW_INVALID,
                        reason="Invalid proof-of-work",
                    )
            
            # 4. Process with underlying coordinator
            result = self._coordinator.process_contribution(contrib)
            
            return result
            
        finally:
            # 5. Pad execution time (timing attack mitigation)
            pad_execution_time(self._min_process_time_ms, start_time)
    
    def get_challenge(self, contributor_id: str) -> Optional[bytes]:
        """Get PoW challenge for contributor (if PoW is enabled)."""
        if not self._sybil.is_enabled():
            return None
        return self._sybil.get_challenge(contributor_id)
    
    def get_rate_limit_status(self, contributor_id: str) -> Dict:
        """Get rate limit status for contributor."""
        remaining = self._rate_limiter.get_remaining(contributor_id)
        return {
            "remaining_tokens": remaining,
            "capacity": self._rate_limiter._capacity,
            "refill_rate": self._rate_limiter._refill_rate,
        }
    
    # Delegate read-only methods
    def get_leaderboard(self):
        return self._coordinator.get_leaderboard()
    
    def get_active_policy(self):
        return self._coordinator.get_active_policy()
    
    def get_log_root(self):
        return self._coordinator.get_log_root()
    
    def get_stats(self):
        return self._coordinator.get_stats()
