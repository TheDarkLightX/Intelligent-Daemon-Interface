"""
FrontierSync: Bandwidth-optimal log synchronization with witness cosigning.

Security Controls:
- Replay attack prevention via nonces and timestamps
- Witness collusion resistance via threshold signatures
- Domain separation in signed messages (goal_id, session_id)
- Bounded sync sessions to prevent resource exhaustion
- Input validation on all external data
- Rate limiting on sync requests

Based on: QMDB, Certificate Transparency, Bitcoin Erlay

Author: DarkLightX
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from .iblt import HASH_SIZE, IBLT, IBLTConfig, estimate_iblt_size

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants
# =============================================================================

# Timestamp freshness window (prevents replay with old signatures)
FRESHNESS_WINDOW_MS = 60_000  # 60 seconds

# Maximum clock skew allowed between nodes
MAX_CLOCK_SKEW_MS = 5_000  # 5 seconds

# Nonce cache size (prevents replay within window)
MAX_NONCE_CACHE_SIZE = 10_000

# Session limits
MAX_ACTIVE_SESSIONS = 100
MAX_SESSION_DURATION_MS = 300_000  # 5 minutes
MAX_ENTRIES_PER_BATCH = 1000
MAX_SYNC_RETRIES = 3

# Async timeouts (seconds)
DEFAULT_ASYNC_TIMEOUT = 30.0
NONCE_CHECK_TIMEOUT = 5.0
RATE_LIMIT_CHECK_TIMEOUT = 5.0

# Witness cosigning
MIN_WITNESS_THRESHOLD = 2
MAX_WITNESS_THRESHOLD = 10
DEFAULT_WITNESS_THRESHOLD = 3
WITNESS_SIGNATURE_TIMEOUT_MS = 5_000

# IBLT defaults
DEFAULT_EXPECTED_DIFF = 1000


# =============================================================================
# Data Structures
# =============================================================================

class SyncStatus(Enum):
    """Status of a sync operation."""
    SUCCESS = "success"
    ALREADY_SYNCED = "already_synced"
    FORK_DETECTED = "fork_detected"
    PEER_AHEAD = "peer_ahead"
    PEER_BEHIND = "peer_behind"
    IBLT_DECODE_FAILED = "iblt_decode_failed"
    INVALID_COSIGNATURE = "invalid_cosignature"
    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class SyncDirection(Enum):
    """Direction of sync operation."""
    PULL = "pull"  # We're behind, receiving entries
    PUSH = "push"  # We're ahead, sending entries
    NONE = "none"  # Already in sync


@dataclass
class SyncState:
    """
    Compact representation of MMR state for sync comparison.

    Security: Includes goal_id for domain separation.
    """
    goal_id: str
    size: int
    frontier: list[bytes]  # MMR peak hashes
    version: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """Validate all fields for security."""
        # Security: Validate goal_id
        if not isinstance(self.goal_id, str) or len(self.goal_id) > 256:
            raise ValueError("goal_id must be string <= 256 chars")
        if not self.goal_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError("goal_id must be alphanumeric with _ or -")

        # Security: Validate size bounds
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("size must be non-negative integer")
        if self.size > 2**48:  # ~280 trillion entries max
            raise ValueError("size exceeds maximum allowed")

        # Security: Validate frontier
        if not isinstance(self.frontier, list):
            raise ValueError("frontier must be a list")
        if len(self.frontier) > 64:  # Max 64 peaks for 2^64 entries
            raise ValueError("frontier has too many peaks")
        for peak in self.frontier:
            if not isinstance(peak, bytes) or len(peak) != HASH_SIZE:
                raise ValueError(f"frontier peak must be {HASH_SIZE} bytes")

        # Security: Validate version
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("version must be non-negative integer")

        # Security: Validate timestamp
        if not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative integer")

    def root_hash(self) -> bytes:
        """Compute single root hash from frontier peaks."""
        if not self.frontier:
            return b'\x00' * HASH_SIZE

        # Bag peaks right-to-left (standard MMR)
        result = self.frontier[-1]
        for peak in reversed(self.frontier[:-1]):
            result = hashlib.sha256(peak + result).digest()
        return result

    def serialize(self) -> bytes:
        """Serialize for signing/hashing."""
        data = bytearray()
        data.extend(self.goal_id.encode('utf-8'))
        data.extend(b'\x00')  # Separator
        data.extend(self.size.to_bytes(8, 'big'))
        data.extend(self.version.to_bytes(8, 'big'))
        data.extend(self.timestamp_ms.to_bytes(8, 'big'))
        data.extend(len(self.frontier).to_bytes(4, 'big'))
        for peak in self.frontier:
            data.extend(peak)
        return bytes(data)

    def hash(self) -> bytes:
        """Compute hash of state for signing."""
        return hashlib.sha256(self.serialize()).digest()


@dataclass
class WitnessSignature:
    """
    Signature from a witness node on a sync state.

    Security:
        - Includes session_nonce for replay prevention
        - Includes timestamp for freshness checking
        - Domain-separated by goal_id in state
        - All fields validated on construction
    """
    witness_id: str
    public_key: bytes  # Ed25519 public key (32 bytes)
    signature: bytes   # Ed25519 signature (64 bytes)
    session_nonce: bytes  # Random nonce for this session (32 bytes)
    timestamp_ms: int

    def __post_init__(self) -> None:
        """Validate all fields for security."""
        # Security: Validate witness_id
        if not isinstance(self.witness_id, str) or len(self.witness_id) > 128:
            raise ValueError("witness_id must be string <= 128 chars")

        # Security: Validate public_key (Ed25519 = 32 bytes)
        if not isinstance(self.public_key, bytes) or len(self.public_key) != 32:
            raise ValueError("public_key must be 32 bytes (Ed25519)")

        # Security: Validate signature (Ed25519 = 64 bytes)
        if not isinstance(self.signature, bytes) or len(self.signature) != 64:
            raise ValueError("signature must be 64 bytes (Ed25519)")

        # Security: Validate session_nonce
        if not isinstance(self.session_nonce, bytes) or len(self.session_nonce) != 32:
            raise ValueError("session_nonce must be 32 bytes")

        # Security: Validate timestamp
        if not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative integer")

    def verify(self, state: SyncState) -> bool:
        """
        Verify signature over state hash.

        Security: Uses Ed25519 for strong signatures.
        """
        if not CRYPTO_AVAILABLE:
            logger.warning("cryptography not available, signature not verified")
            return False

        try:
            # Reconstruct signed message
            message = self._build_signed_message(state)

            # Load public key
            public_key = Ed25519PublicKey.from_public_bytes(self.public_key)

            # Verify signature
            public_key.verify(self.signature, message)
            return True

        except (InvalidSignature, ValueError) as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    def _build_signed_message(self, state: SyncState) -> bytes:
        """
        Build the message that was signed.

        Security: Domain separation via goal_id and session_nonce.
        """
        # Format: "FRONTIERSYNC_WITNESS_V1" || state_hash || session_nonce || timestamp
        return (
            b"FRONTIERSYNC_WITNESS_V1" +
            state.hash() +
            self.session_nonce +
            self.timestamp_ms.to_bytes(8, 'big')
        )


@dataclass
class CosignedSyncState:
    """
    Sync state with multiple witness signatures.

    Security:
        - Requires threshold signatures for validity
        - All signatures must be fresh (within FRESHNESS_WINDOW_MS)
        - Detects duplicate witness IDs
        - Validates witness diversity (min unique public keys)
    """
    state: SyncState
    witnesses: list[WitnessSignature]
    threshold: int
    min_unique_entities: int = 2  # Security: Prevent single-entity Sybil

    def is_valid(self, now_ms: int | None = None) -> tuple[bool, str]:
        """
        Validate cosigned state.

        Returns: (is_valid, error_message)

        Security checks:
            1. Minimum threshold met
            2. All signatures valid
            3. All timestamps fresh
            4. No duplicate witnesses
            5. Witness diversity (unique public keys)
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # Check threshold
        if len(self.witnesses) < self.threshold:
            return False, f"Insufficient witnesses: {len(self.witnesses)} < {self.threshold}"

        # Check for duplicates
        witness_ids = [w.witness_id for w in self.witnesses]
        if len(witness_ids) != len(set(witness_ids)):
            return False, "Duplicate witness IDs detected"

        # Security: Check witness diversity (unique public keys)
        # Prevents single entity from controlling multiple witness IDs
        unique_pubkeys = {w.public_key for w in self.witnesses}
        if len(unique_pubkeys) < self.min_unique_entities:
            return False, (
                f"Insufficient witness diversity: {len(unique_pubkeys)} unique keys "
                f"< {self.min_unique_entities} required"
            )

        valid_count = 0
        for witness in self.witnesses:
            # Check freshness
            age_ms = now_ms - witness.timestamp_ms
            if age_ms > FRESHNESS_WINDOW_MS:
                logger.warning(f"Stale witness signature from {witness.witness_id}: {age_ms}ms old")
                continue
            if age_ms < -MAX_CLOCK_SKEW_MS:
                logger.warning(f"Future witness signature from {witness.witness_id}: {-age_ms}ms ahead")
                continue

            # Verify signature
            if witness.verify(self.state):
                valid_count += 1

        if valid_count < self.threshold:
            return False, f"Insufficient valid signatures: {valid_count} < {self.threshold}"

        return True, ""


@dataclass
class ForkProof:
    """
    Cryptographic proof of a fork (two conflicting cosigned states).

    Security: Proves at least one witness signed conflicting histories.
    """
    state_a: CosignedSyncState
    state_b: CosignedSyncState
    conflicting_witnesses: list[str]  # Witnesses who signed both

    def is_valid(self) -> tuple[bool, str]:
        """Validate fork proof."""
        # States must be at same size
        if self.state_a.state.size != self.state_b.state.size:
            return False, "States are at different sizes (not a fork)"

        # States must have different frontiers
        if self.state_a.state.frontier == self.state_b.state.frontier:
            return False, "States are identical (no fork)"

        # Both states must be valid
        valid_a, err_a = self.state_a.is_valid()
        if not valid_a:
            return False, f"State A invalid: {err_a}"

        valid_b, err_b = self.state_b.is_valid()
        if not valid_b:
            return False, f"State B invalid: {err_b}"

        # Must have overlapping witnesses
        witnesses_a = {w.witness_id for w in self.state_a.witnesses}
        witnesses_b = {w.witness_id for w in self.state_b.witnesses}
        overlap = witnesses_a & witnesses_b

        if not overlap:
            return False, "No overlapping witnesses (cannot prove fork)"

        if set(self.conflicting_witnesses) != overlap:
            return False, "Conflicting witnesses list doesn't match overlap"

        return True, ""


@dataclass
class SyncSession:
    """
    State for an ongoing sync session.

    Security:
        - Bounded duration to prevent resource exhaustion
        - Unique session_id for replay prevention
        - Tracks peer for authentication
    """
    session_id: str  # Random unique ID
    peer_id: str
    goal_id: str
    direction: SyncDirection

    # State tracking
    local_state: SyncState | None = None
    peer_state: SyncState | None = None
    common_size: int = 0

    # Progress
    entries_requested: int = 0
    entries_received: int = 0
    entries_sent: int = 0

    # Timing
    started_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    last_activity_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # Error handling
    retries: int = 0
    last_error: str | None = None

    def is_expired(self, now_ms: int | None = None) -> bool:
        """Check if session has exceeded max duration."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return (now_ms - self.started_ms) > MAX_SESSION_DURATION_MS

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_ms = int(time.time() * 1000)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    status: SyncStatus
    method: str = "frontier"  # "frontier" or "iblt"
    entries_synced: int = 0
    entries_sent: int = 0
    bandwidth_saved_bytes: int = 0
    fork_proof: ForkProof | None = None
    error: str | None = None


# =============================================================================
# Nonce Cache for Replay Prevention
# =============================================================================

class NonceCache:
    """
    Bounded LRU cache for tracking used nonces.

    Security: Prevents signature replay attacks.
    """

    def __init__(self, max_size: int = MAX_NONCE_CACHE_SIZE):
        self._max_size = max_size
        self._cache: OrderedDict[bytes, int] = OrderedDict()  # nonce -> timestamp_ms
        self._lock = asyncio.Lock()

    async def check_and_add(self, nonce: bytes, timestamp_ms: int) -> bool:
        """
        Check if nonce has been used; if not, add it.

        Returns: True if nonce is fresh (not seen before)

        Security: Atomic check-and-add to prevent TOCTOU.
        """
        # Input validation
        if not isinstance(nonce, bytes) or len(nonce) != 32:
            raise ValueError("Nonce must be 32 bytes")
        if not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            raise ValueError("Timestamp must be a positive integer")

        async def _check_and_add_locked() -> bool:
            async with self._lock:
                if nonce in self._cache:
                    return False  # Replay detected

                # Add nonce
                self._cache[nonce] = timestamp_ms

                # Evict oldest if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

                return True

        try:
            return await asyncio.wait_for(
                _check_and_add_locked(),
                timeout=NONCE_CHECK_TIMEOUT,
            )
        except asyncio.TimeoutError as err:
            logger.warning("Nonce check timed out")
            raise RuntimeError("Nonce check timed out") from err

    async def cleanup_expired(self, now_ms: int | None = None) -> int:
        """Remove expired nonces. Returns count removed."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        cutoff = now_ms - FRESHNESS_WINDOW_MS
        removed = 0

        async with self._lock:
            # Find expired nonces
            expired = [
                nonce for nonce, ts in self._cache.items()
                if ts < cutoff
            ]
            for nonce in expired:
                del self._cache[nonce]
                removed += 1

        return removed


# =============================================================================
# Rate Limiter
# =============================================================================

class SyncRateLimiter:
    """
    Rate limiting for sync operations per peer.

    Security: Prevents DoS via excessive sync requests.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 10,
        max_bytes_per_minute: int = 100 * 1024 * 1024,  # 100 MB
    ):
        self._max_requests = max_requests_per_minute
        self._max_bytes = max_bytes_per_minute
        self._peer_requests: dict[str, list[int]] = {}  # peer_id -> [timestamps]
        self._peer_bytes: dict[str, list[tuple[int, int]]] = {}  # peer_id -> [(ts, bytes)]
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        peer_id: str,
        bytes_count: int = 0
    ) -> tuple[bool, str]:
        """
        Check if request from peer is allowed.

        Returns: (allowed, reason)
        """
        now_ms = int(time.time() * 1000)
        window_start = now_ms - 60_000  # 1 minute window

        async with self._lock:
            # Clean old entries
            if peer_id in self._peer_requests:
                self._peer_requests[peer_id] = [
                    ts for ts in self._peer_requests[peer_id]
                    if ts > window_start
                ]
            else:
                self._peer_requests[peer_id] = []

            if peer_id in self._peer_bytes:
                self._peer_bytes[peer_id] = [
                    (ts, b) for ts, b in self._peer_bytes[peer_id]
                    if ts > window_start
                ]
            else:
                self._peer_bytes[peer_id] = []

            # Check request count
            if len(self._peer_requests[peer_id]) >= self._max_requests:
                return False, f"Rate limit: {self._max_requests} requests/minute exceeded"

            # Check bytes
            total_bytes = sum(b for _, b in self._peer_bytes[peer_id])
            if total_bytes + bytes_count > self._max_bytes:
                return False, f"Rate limit: {self._max_bytes} bytes/minute exceeded"

            # Record this request
            self._peer_requests[peer_id].append(now_ms)
            if bytes_count > 0:
                self._peer_bytes[peer_id].append((now_ms, bytes_count))

            return True, ""


# =============================================================================
# Witness Coordinator
# =============================================================================

class WitnessCoordinator:
    """
    Coordinates witness cosigning for sync state checkpoints.

    Security:
        - Validates all signatures
        - Enforces freshness window
        - Tracks nonces to prevent replay
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey | None = None,
        node_id: str = "",
        threshold: int = DEFAULT_WITNESS_THRESHOLD,
    ):
        self._private_key = private_key
        self._node_id = node_id
        self._threshold = min(max(threshold, MIN_WITNESS_THRESHOLD), MAX_WITNESS_THRESHOLD)
        self._nonce_cache = NonceCache()
        self._known_witnesses: dict[str, bytes] = {}  # witness_id -> public_key

    def add_witness(self, witness_id: str, public_key: bytes) -> None:
        """Register a known witness."""
        if len(public_key) != 32:
            raise ValueError("Public key must be 32 bytes")
        self._known_witnesses[witness_id] = public_key

    def sign_state(self, state: SyncState, session_nonce: bytes) -> WitnessSignature | None:
        """
        Sign a sync state as a witness.

        Security: Creates domain-separated signature with nonce.
        """
        if self._private_key is None:
            return None

        if len(session_nonce) != 32:
            raise ValueError("Session nonce must be 32 bytes")

        timestamp_ms = int(time.time() * 1000)

        # Build message
        message = (
            b"FRONTIERSYNC_WITNESS_V1" +
            state.hash() +
            session_nonce +
            timestamp_ms.to_bytes(8, 'big')
        )

        # Sign
        signature = self._private_key.sign(message)
        public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        return WitnessSignature(
            witness_id=self._node_id,
            public_key=public_key,
            signature=signature,
            session_nonce=session_nonce,
            timestamp_ms=timestamp_ms,
        )

    async def validate_witness_signature(
        self,
        witness: WitnessSignature,
        state: SyncState
    ) -> tuple[bool, str]:
        """
        Validate a witness signature.

        Security checks:
            1. Witness is known
            2. Public key matches
            3. Signature is valid
            4. Timestamp is fresh
            5. Nonce hasn't been used
        """
        now_ms = int(time.time() * 1000)

        # Check if witness is known
        if witness.witness_id not in self._known_witnesses:
            return False, f"Unknown witness: {witness.witness_id}"

        # Check public key matches
        expected_key = self._known_witnesses[witness.witness_id]
        if witness.public_key != expected_key:
            return False, "Public key mismatch"

        # Check timestamp freshness
        age_ms = now_ms - witness.timestamp_ms
        if age_ms > FRESHNESS_WINDOW_MS:
            return False, f"Signature expired: {age_ms}ms old"
        if age_ms < -MAX_CLOCK_SKEW_MS:
            return False, f"Signature from future: {-age_ms}ms ahead"

        # Check nonce hasn't been used (replay prevention)
        is_fresh = await self._nonce_cache.check_and_add(
            witness.session_nonce,
            witness.timestamp_ms
        )
        if not is_fresh:
            return False, "Nonce replay detected"

        # Verify signature
        if not witness.verify(state):
            return False, "Invalid signature"

        return True, ""

    def detect_fork(
        self,
        state_a: CosignedSyncState,
        state_b: CosignedSyncState
    ) -> ForkProof | None:
        """
        Check if two cosigned states prove a fork.

        Returns ForkProof if fork detected, None otherwise.
        """
        # Must be at same size
        if state_a.state.size != state_b.state.size:
            return None

        # Must have different frontiers
        if state_a.state.frontier == state_b.state.frontier:
            return None

        # Find overlapping witnesses
        witnesses_a = {w.witness_id for w in state_a.witnesses}
        witnesses_b = {w.witness_id for w in state_b.witnesses}
        overlap = witnesses_a & witnesses_b

        if not overlap:
            return None

        return ForkProof(
            state_a=state_a,
            state_b=state_b,
            conflicting_witnesses=list(overlap),
        )


# =============================================================================
# FrontierSync Main Class
# =============================================================================

class FrontierSync:
    """
    Bandwidth-optimal log synchronization with witness cosigning.

    Security features:
        - IBLT for O(Δ) bandwidth (falls back to frontier on decode failure)
        - Witness cosigning for fork accountability
        - Rate limiting per peer
        - Bounded sessions
        - Nonce-based replay prevention
        - Input validation on all external data

    Usage:
        sync = FrontierSync(mmr, transport, config)
        result = await sync.sync_with_peer(peer_id)
    """

    def __init__(
        self,
        mmr: Any,  # TwigMMR or compatible
        transport: Any,  # P2P transport
        witness_coordinator: WitnessCoordinator | None = None,
        goal_id: str = "default",
        iblt_config: IBLTConfig | None = None,
    ):
        self._mmr = mmr
        self._transport = transport
        self._witness_coordinator = witness_coordinator
        self._goal_id = goal_id

        # IBLT configuration
        if iblt_config is None:
            iblt_config = IBLTConfig(
                num_cells=estimate_iblt_size(DEFAULT_EXPECTED_DIFF),
                hash_seed=secrets.token_bytes(32),
            )
        self._iblt_config = iblt_config

        # Session management
        self._active_sessions: dict[str, SyncSession] = {}
        self._session_lock = asyncio.Lock()

        # Security controls
        self._rate_limiter = SyncRateLimiter()
        self._nonce_cache = NonceCache()

    def get_sync_state(self) -> SyncState:
        """Get current sync state from MMR."""
        return SyncState(
            goal_id=self._goal_id,
            size=self._mmr.size if hasattr(self._mmr, 'size') else 0,
            frontier=self._mmr.frontier if hasattr(self._mmr, 'frontier') else [],
            version=self._mmr.version if hasattr(self._mmr, 'version') else 0,
        )

    async def _create_session(self, peer_id: str) -> SyncSession:
        """
        Create a new sync session with peer.

        Security: Bounded number of active sessions.
        """
        async with self._session_lock:
            # Clean expired sessions
            now_ms = int(time.time() * 1000)
            expired = [
                sid for sid, session in self._active_sessions.items()
                if session.is_expired(now_ms)
            ]
            for sid in expired:
                del self._active_sessions[sid]

            # Check session limit
            if len(self._active_sessions) >= MAX_ACTIVE_SESSIONS:
                raise RuntimeError(f"Max active sessions ({MAX_ACTIVE_SESSIONS}) exceeded")

            # Create session
            session = SyncSession(
                session_id=secrets.token_hex(16),
                peer_id=peer_id,
                goal_id=self._goal_id,
                direction=SyncDirection.NONE,
            )
            self._active_sessions[session.session_id] = session

            return session

    async def _close_session(self, session: SyncSession) -> None:
        """Close and remove a sync session."""
        async with self._session_lock:
            if session.session_id in self._active_sessions:
                del self._active_sessions[session.session_id]

    async def sync_with_peer(self, peer_id: str) -> SyncResult:
        """
        Synchronize log with a peer node.

        Algorithm:
            1. Rate limit check
            2. Create session
            3. Exchange sync states
            4. Try IBLT reconciliation first
            5. Fall back to frontier sync if IBLT fails
            6. Verify results
            7. Close session

        Security:
            - Rate limited
            - Bounded session duration
            - All data validated
        """
        # Security: Rate limit check
        allowed, reason = await self._rate_limiter.check_rate_limit(peer_id)
        if not allowed:
            return SyncResult(status=SyncStatus.RATE_LIMITED, error=reason)

        session = None
        try:
            # Create session
            session = await self._create_session(peer_id)

            # Get local state
            local_state = self.get_sync_state()
            session.local_state = local_state

            # Exchange states with peer
            peer_state = await self._exchange_states(peer_id, local_state)
            session.peer_state = peer_state

            # Determine direction
            if local_state.size == peer_state.size:
                if local_state.frontier == peer_state.frontier:
                    return SyncResult(status=SyncStatus.ALREADY_SYNCED)
                else:
                    # Same size, different frontier = FORK
                    return await self._handle_fork(session)

            if local_state.size < peer_state.size:
                session.direction = SyncDirection.PULL
                return await self._pull_from_peer(session)
            else:
                session.direction = SyncDirection.PUSH
                return await self._push_to_peer(session)

        except asyncio.TimeoutError:
            return SyncResult(status=SyncStatus.TIMEOUT, error="Sync timed out")
        except Exception as e:
            logger.exception(f"Sync error with {peer_id}")
            return SyncResult(status=SyncStatus.ERROR, error=str(e))
        finally:
            if session:
                await self._close_session(session)

    async def sync_with_iblt(self, peer_id: str) -> SyncResult:
        """
        Bandwidth-optimal sync using IBLT reconciliation.

        Security: Falls back to frontier sync if IBLT decode fails.
        """
        # Security: Rate limit check
        allowed, reason = await self._rate_limiter.check_rate_limit(peer_id)
        if not allowed:
            return SyncResult(status=SyncStatus.RATE_LIMITED, error=reason)

        session = None
        try:
            session = await self._create_session(peer_id)

            # Build local IBLT from entry hashes
            local_iblt = await self._build_iblt_from_log()

            auth_key = None
            if hasattr(self._transport, 'get_session_key'):
                try:
                    auth_key = self._transport.get_session_key(peer_id)
                except Exception:
                    auth_key = None
            if auth_key is not None and (not isinstance(auth_key, (bytes, bytearray)) or len(auth_key) != 32):
                auth_key = None

            local_iblt_data = local_iblt.serialize(auth_key=auth_key) if auth_key else local_iblt.serialize()

            # Exchange IBLTs with peer
            peer_iblt_data = await self._exchange_iblts(peer_id, local_iblt_data)
            try:
                peer_iblt = IBLT.deserialize(peer_iblt_data, self._iblt_config, auth_key=auth_key)
            except ValueError as e:
                if auth_key is not None and 'HMAC verification failed' in str(e):
                    return SyncResult(status=SyncStatus.ERROR, error=str(e))
                raise

            # Compute and decode difference
            diff_iblt = local_iblt.subtract(peer_iblt)
            only_local, only_peer, success = diff_iblt.decode()

            if not success:
                # IBLT decode failed, fall back to frontier sync
                logger.info("IBLT decode failed, falling back to frontier sync")
                return await self.sync_with_peer(peer_id)

            # Request entries we're missing
            if only_peer:
                missing_entries = await self._request_entries_by_hash(
                    peer_id,
                    list(only_peer)
                )
                for entry in missing_entries:
                    # Security: Validate entry hash matches
                    entry_hash = hashlib.sha256(entry).digest()
                    if entry_hash in only_peer:
                        self._mmr.append(entry)
                    else:
                        logger.warning("Received entry with unexpected hash")

                session.entries_received = len(missing_entries)

            # Send entries peer is missing
            if only_local:
                local_entries = await self._get_entries_by_hash(list(only_local))
                await self._send_entries(peer_id, local_entries)
                session.entries_sent = len(local_entries)

            # Calculate bandwidth savings
            full_sync_bytes = (len(only_peer) + len(only_local)) * 100  # Estimate
            iblt_bytes = len(local_iblt_data) * 2  # Sent + received
            saved = max(0, full_sync_bytes - iblt_bytes)

            return SyncResult(
                status=SyncStatus.SUCCESS,
                method="iblt",
                entries_synced=session.entries_received,
                entries_sent=session.entries_sent,
                bandwidth_saved_bytes=saved,
            )

        except Exception:
            logger.exception(f"IBLT sync error with {peer_id}")
            # Fall back to frontier sync
            return await self.sync_with_peer(peer_id)
        finally:
            if session:
                await self._close_session(session)

    async def _build_iblt_from_log(self) -> IBLT:
        """Build IBLT from current log entries."""
        iblt = IBLT(self._iblt_config)

        # Get all entry hashes from MMR
        if hasattr(self._mmr, 'get_all_entry_hashes'):
            entry_hashes = self._mmr.get_all_entry_hashes()
            for h in entry_hashes:
                iblt.insert(h)

        return iblt

    async def _exchange_states(self, peer_id: str, local_state: SyncState) -> SyncState:
        """Exchange sync states with peer."""
        # This would use the transport layer
        # Placeholder for actual implementation
        raise NotImplementedError("Subclass must implement _exchange_states")

    async def _exchange_iblts(self, peer_id: str, local_iblt_data: bytes) -> bytes:
        """Exchange IBLT data with peer."""
        raise NotImplementedError("Subclass must implement _exchange_iblts")

    async def _request_entries_by_hash(
        self,
        peer_id: str,
        hashes: list[bytes]
    ) -> list[bytes]:
        """Request specific entries by hash from peer."""
        raise NotImplementedError("Subclass must implement _request_entries_by_hash")

    async def _get_entries_by_hash(self, hashes: list[bytes]) -> list[bytes]:
        """Get local entries by hash."""
        raise NotImplementedError("Subclass must implement _get_entries_by_hash")

    async def _send_entries(self, peer_id: str, entries: list[bytes]) -> None:
        """Send entries to peer."""
        raise NotImplementedError("Subclass must implement _send_entries")

    async def _pull_from_peer(self, session: SyncSession) -> SyncResult:
        """Pull missing entries from peer."""
        raise NotImplementedError("Subclass must implement _pull_from_peer")

    async def _push_to_peer(self, session: SyncSession) -> SyncResult:
        """Push entries to peer that's behind."""
        raise NotImplementedError("Subclass must implement _push_to_peer")

    async def _handle_fork(self, session: SyncSession) -> SyncResult:
        """Handle detected fork between nodes."""
        if self._witness_coordinator is None:
            return SyncResult(
                status=SyncStatus.FORK_DETECTED,
                error="Fork detected but no witness coordinator configured",
            )

        # Get cosigned states for fork proof
        # This would involve requesting cosigned states from both sides
        # and constructing a ForkProof
        return SyncResult(
            status=SyncStatus.FORK_DETECTED,
            error="Fork detected - manual resolution required",
        )
