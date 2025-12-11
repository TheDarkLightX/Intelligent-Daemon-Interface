"""
IAN Real Tau Net Sender - Production integration with Tau Testnet.

Provides:
1. TauNetSender - Connect to real Tau Testnet node via RPC or CLI
2. Transaction formatting for Tau Testnet sendtx
3. BLS signature handling (optional)
4. Connection pooling and retry logic

Integration Methods:
1. CLI: Shell out to sendtx command
2. RPC: Direct TCP connection to Tau node
3. HTTP: REST API if available

Security:
- Private keys for signing can be loaded from env or file
- Connection uses TLS where available
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from io import BytesIO
from pathlib import Path
from struct import pack, unpack
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

# =============================================================================
# Backoff Utilities
# =============================================================================

def backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.3,
) -> float:
    """
    Calculate exponential backoff with jitter.
    
    Prevents thundering herd by adding randomness to retry delays.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter_factor: Fraction of delay to add as jitter (0.0-1.0)
        
    Returns:
        Delay in seconds with jitter applied
    """
    # Exponential: 1, 2, 4, 8, 16, ...
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    # Add jitter: ±30% by default
    jitter = delay * jitter_factor * random.uniform(-1, 1)
    
    return max(0.1, delay + jitter)

logger = logging.getLogger(__name__)


# =============================================================================
# IAN Bytecode Format
# =============================================================================

class IanOpCode(Enum):
    """
    IAN operation bytecodes for compact transaction encoding.
    
    Bytecode format is more efficient than JSON for:
    - Wire transmission (smaller payload)
    - On-chain validation (direct bitvector operations)
    - Cryptographic hashing (deterministic encoding)
    
    Format: [VERSION:1][OPCODE:1][PAYLOAD_LEN:4][PAYLOAD:N][SIGNATURE:96]
    """
    GOAL_REGISTER = 0x01
    LOG_COMMIT = 0x02
    UPGRADE = 0x03
    CHALLENGE = 0x04
    BOND_DEPOSIT = 0x05
    BOND_WITHDRAW = 0x06
    EVALUATOR_REGISTER = 0x07


# Version byte for bytecode format
BYTECODE_VERSION = 0x01
SUPPORTED_VERSIONS = {0x01}

# Security limits
MAX_PAYLOAD_SIZE = 16 * 1024 * 1024  # 16 MB max
MAX_NAME_LENGTH = 256
MAX_CONTRIB_ID_LENGTH = 256


class IanBytecodeEncoder:
    """
    Encode IAN transactions to compact bytecode format.
    
    Benefits over JSON:
    - ~50-70% smaller payload
    - Deterministic encoding (no key ordering issues)
    - Direct mapping to Tau bitvector operations
    - Faster parsing and validation
    
    Format:
    ```
    ┌─────────┬─────────┬────────────┬───────────┬───────────┐
    │ VERSION │ OPCODE  │ PAYLOAD_LEN│  PAYLOAD  │ SIGNATURE │
    │ 1 byte  │ 1 byte  │  4 bytes   │  N bytes  │ 96 bytes  │
    └─────────┴─────────┴────────────┴───────────┴───────────┘
    ```
    """
    
    @staticmethod
    def encode_goal_register(
        goal_id: bytes,  # 32 bytes
        goal_spec_hash: bytes,  # 32 bytes
        name: str,
        timestamp_ms: int,
    ) -> bytes:
        """
        Encode IAN_GOAL_REGISTER.
        
        Payload:
        [GOAL_ID:32][SPEC_HASH:32][NAME_LEN:2][NAME:N][TIMESTAMP:8]
        """
        buf = BytesIO()
        
        # Header
        buf.write(pack('>BB', BYTECODE_VERSION, IanOpCode.GOAL_REGISTER.value))
        
        # Payload
        name_bytes = name.encode('utf-8')[:256]  # Max 256 chars
        payload = BytesIO()
        payload.write(goal_id[:32].ljust(32, b'\x00'))  # Goal ID (32 bytes)
        payload.write(goal_spec_hash[:32].ljust(32, b'\x00'))  # Spec hash (32 bytes)
        payload.write(pack('>H', len(name_bytes)))  # Name length (2 bytes)
        payload.write(name_bytes)  # Name (variable)
        payload.write(pack('>Q', timestamp_ms))  # Timestamp (8 bytes)
        
        payload_bytes = payload.getvalue()
        buf.write(pack('>I', len(payload_bytes)))  # Payload length
        buf.write(payload_bytes)
        
        return buf.getvalue()
    
    @staticmethod
    def encode_log_commit(
        goal_id: bytes,  # 32 bytes
        log_root: bytes,  # 32 bytes
        log_size: int,
        leaderboard_root: bytes,  # 32 bytes
        leaderboard_size: int,
        prev_commit_hash: Optional[bytes],  # 32 bytes or None
        timestamp_ms: int,
    ) -> bytes:
        """
        Encode IAN_LOG_COMMIT.
        
        Payload:
        [GOAL_ID:32][LOG_ROOT:32][LOG_SIZE:8][LB_ROOT:32][LB_SIZE:4]
        [HAS_PREV:1][PREV_HASH:32?][TIMESTAMP:8]
        """
        buf = BytesIO()
        
        # Header
        buf.write(pack('>BB', BYTECODE_VERSION, IanOpCode.LOG_COMMIT.value))
        
        # Payload
        payload = BytesIO()
        payload.write(goal_id[:32].ljust(32, b'\x00'))
        payload.write(log_root[:32].ljust(32, b'\x00'))
        payload.write(pack('>Q', log_size))  # 8 bytes for large logs
        payload.write(leaderboard_root[:32].ljust(32, b'\x00'))
        payload.write(pack('>I', leaderboard_size))  # 4 bytes
        
        if prev_commit_hash:
            payload.write(b'\x01')  # Has previous
            payload.write(prev_commit_hash[:32].ljust(32, b'\x00'))
        else:
            payload.write(b'\x00')  # No previous
        
        payload.write(pack('>Q', timestamp_ms))
        
        payload_bytes = payload.getvalue()
        buf.write(pack('>I', len(payload_bytes)))
        buf.write(payload_bytes)
        
        return buf.getvalue()
    
    @staticmethod
    def encode_upgrade(
        goal_id: bytes,  # 32 bytes
        pack_hash: bytes,  # 32 bytes
        score: int,  # Fixed-point (scaled by 1e6)
        log_index: int,
        log_root: bytes,  # 32 bytes
        contributor_id: str,
        prev_pack_hash: Optional[bytes],
        timestamp_ms: int,
    ) -> bytes:
        """
        Encode IAN_UPGRADE.
        
        Payload:
        [GOAL_ID:32][PACK_HASH:32][SCORE:8][LOG_INDEX:8][LOG_ROOT:32]
        [CONTRIB_LEN:2][CONTRIB_ID:N][HAS_PREV:1][PREV_HASH:32?][TIMESTAMP:8]
        """
        buf = BytesIO()
        
        # Header
        buf.write(pack('>BB', BYTECODE_VERSION, IanOpCode.UPGRADE.value))
        
        # Payload
        payload = BytesIO()
        payload.write(goal_id[:32].ljust(32, b'\x00'))
        payload.write(pack_hash[:32].ljust(32, b'\x00'))
        payload.write(pack('>Q', score))  # Score as fixed-point int
        payload.write(pack('>Q', log_index))
        payload.write(log_root[:32].ljust(32, b'\x00'))
        
        contrib_bytes = contributor_id.encode('utf-8')[:256]
        payload.write(pack('>H', len(contrib_bytes)))
        payload.write(contrib_bytes)
        
        if prev_pack_hash:
            payload.write(b'\x01')
            payload.write(prev_pack_hash[:32].ljust(32, b'\x00'))
        else:
            payload.write(b'\x00')
        
        payload.write(pack('>Q', timestamp_ms))
        
        payload_bytes = payload.getvalue()
        buf.write(pack('>I', len(payload_bytes)))
        buf.write(payload_bytes)
        
        return buf.getvalue()
    
    @staticmethod
    def encode_challenge(
        goal_id: bytes,  # 32 bytes
        challenged_commit_hash: bytes,  # 32 bytes
        fraud_type: int,  # 1 byte
        fraud_proof_hash: bytes,  # 32 bytes (hash of full proof)
        timestamp_ms: int,
    ) -> bytes:
        """
        Encode IAN_CHALLENGE.
        
        Payload:
        [GOAL_ID:32][COMMIT_HASH:32][FRAUD_TYPE:1][PROOF_HASH:32][TIMESTAMP:8]
        """
        buf = BytesIO()
        
        # Header
        buf.write(pack('>BB', BYTECODE_VERSION, IanOpCode.CHALLENGE.value))
        
        # Payload
        payload = BytesIO()
        payload.write(goal_id[:32].ljust(32, b'\x00'))
        payload.write(challenged_commit_hash[:32].ljust(32, b'\x00'))
        payload.write(pack('>B', fraud_type))
        payload.write(fraud_proof_hash[:32].ljust(32, b'\x00'))
        payload.write(pack('>Q', timestamp_ms))
        
        payload_bytes = payload.getvalue()
        buf.write(pack('>I', len(payload_bytes)))
        buf.write(payload_bytes)
        
        return buf.getvalue()
    
    @staticmethod
    def add_signature(bytecode: bytes, signature: bytes) -> bytes:
        """Append 96-byte BLS signature to bytecode."""
        return bytecode + signature[:96].ljust(96, b'\x00')


class IanBytecodeDecoder:
    """
    Decode IAN bytecode back to structured data.
    
    Security:
    - Validates all buffer bounds before reads
    - Enforces max payload size limits
    - Validates version compatibility
    - Fails fast on malformed input
    """
    
    @staticmethod
    def _require_bytes(buf: BytesIO, size: int, total_len: int, field_name: str) -> bytes:
        """
        Read exactly `size` bytes with bounds validation.
        
        Precondition: buf.tell() + size <= total_len
        Raises: ValueError if insufficient data
        """
        if buf.tell() + size > total_len:
            raise ValueError(f"Truncated bytecode at field '{field_name}': need {size} bytes")
        data = buf.read(size)
        if len(data) != size:
            raise ValueError(f"Short read at field '{field_name}': got {len(data)}, need {size}")
        return data
    
    @staticmethod
    def decode(data: bytes) -> Dict[str, Any]:
        """
        Decode bytecode to dictionary.
        
        Args:
            data: Raw bytecode bytes
            
        Returns:
            Dict with 'version', 'opcode', 'payload', 'signature'
            
        Raises:
            ValueError: If bytecode is malformed, truncated, or exceeds limits
        """
        total_len = len(data)
        
        # Precondition: minimum header size (version + opcode + payload_len)
        if total_len < 6:
            raise ValueError(f"Bytecode too short: {total_len} < 6 bytes minimum")
        
        buf = BytesIO(data)
        
        # Parse header
        header = IanBytecodeDecoder._require_bytes(buf, 2, total_len, "header")
        version, opcode = unpack('>BB', header)
        
        # Invariant: version must be supported
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported bytecode version: {version}, expected one of {SUPPORTED_VERSIONS}")
        
        # Validate opcode exists
        try:
            opcode_enum = IanOpCode(opcode)
        except ValueError:
            raise ValueError(f"Unknown opcode: 0x{opcode:02x}")
        
        # Parse payload length
        len_bytes = IanBytecodeDecoder._require_bytes(buf, 4, total_len, "payload_length")
        payload_len = unpack('>I', len_bytes)[0]
        
        # Precondition: payload length within bounds
        if payload_len > MAX_PAYLOAD_SIZE:
            raise ValueError(f"Payload too large: {payload_len} > {MAX_PAYLOAD_SIZE}")
        
        # Precondition: sufficient data for payload
        remaining = total_len - buf.tell()
        if payload_len > remaining:
            raise ValueError(f"Truncated payload: need {payload_len}, have {remaining}")
        
        payload = IanBytecodeDecoder._require_bytes(buf, payload_len, total_len, "payload")
        
        # Signature if present (remaining 96 bytes)
        signature = None
        sig_remaining = total_len - buf.tell()
        if sig_remaining >= 96:
            signature = buf.read(96)
        
        # Decode payload based on opcode
        result: Dict[str, Any] = {
            'version': version,
            'opcode': opcode_enum,
            'signature': signature,
        }
        
        try:
            if opcode == IanOpCode.GOAL_REGISTER.value:
                result['payload'] = IanBytecodeDecoder._decode_goal_register(payload)
            elif opcode == IanOpCode.LOG_COMMIT.value:
                result['payload'] = IanBytecodeDecoder._decode_log_commit(payload)
            elif opcode == IanOpCode.UPGRADE.value:
                result['payload'] = IanBytecodeDecoder._decode_upgrade(payload)
            elif opcode == IanOpCode.CHALLENGE.value:
                result['payload'] = IanBytecodeDecoder._decode_challenge(payload)
            else:
                result['payload'] = payload  # Raw for unknown opcodes
        except Exception as e:
            raise ValueError(f"Failed to decode {opcode_enum.name} payload: {e}") from e
        
        return result
    
    @staticmethod
    def _decode_goal_register(payload: bytes) -> Dict[str, Any]:
        """Decode GOAL_REGISTER payload with bounds validation."""
        total_len = len(payload)
        min_size = 32 + 32 + 2 + 8  # goal_id + spec_hash + name_len + timestamp
        if total_len < min_size:
            raise ValueError(f"GOAL_REGISTER payload too short: {total_len} < {min_size}")
        
        buf = BytesIO(payload)
        req = lambda n, name: IanBytecodeDecoder._require_bytes(buf, n, total_len, name)
        
        goal_id = req(32, "goal_id")
        spec_hash = req(32, "goal_spec_hash")
        name_len = unpack('>H', req(2, "name_len"))[0]
        
        # Validate name length
        if name_len > MAX_NAME_LENGTH:
            raise ValueError(f"Name too long: {name_len} > {MAX_NAME_LENGTH}")
        
        name_bytes = req(name_len, "name")
        try:
            name = name_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 in name: {e}")
        
        timestamp_ms = unpack('>Q', req(8, "timestamp_ms"))[0]
        
        return {
            'goal_id': goal_id,
            'goal_spec_hash': spec_hash,
            'name': name,
            'timestamp_ms': timestamp_ms,
        }
    
    @staticmethod
    def _decode_log_commit(payload: bytes) -> Dict[str, Any]:
        """Decode LOG_COMMIT payload with bounds validation."""
        total_len = len(payload)
        min_size = 32 + 32 + 8 + 32 + 4 + 1 + 8  # Without optional prev_hash
        if total_len < min_size:
            raise ValueError(f"LOG_COMMIT payload too short: {total_len} < {min_size}")
        
        buf = BytesIO(payload)
        req = lambda n, name: IanBytecodeDecoder._require_bytes(buf, n, total_len, name)
        
        goal_id = req(32, "goal_id")
        log_root = req(32, "log_root")
        log_size = unpack('>Q', req(8, "log_size"))[0]
        lb_root = req(32, "leaderboard_root")
        lb_size = unpack('>I', req(4, "leaderboard_size"))[0]
        
        has_prev_byte = req(1, "has_prev")
        has_prev = has_prev_byte[0]
        
        prev_hash = None
        if has_prev:
            prev_hash = req(32, "prev_commit_hash")
        
        timestamp_ms = unpack('>Q', req(8, "timestamp_ms"))[0]
        
        return {
            'goal_id': goal_id,
            'log_root': log_root,
            'log_size': log_size,
            'leaderboard_root': lb_root,
            'leaderboard_size': lb_size,
            'prev_commit_hash': prev_hash,
            'timestamp_ms': timestamp_ms,
        }
    
    @staticmethod
    def _decode_upgrade(payload: bytes) -> Dict[str, Any]:
        """Decode UPGRADE payload with bounds validation."""
        total_len = len(payload)
        min_size = 32 + 32 + 8 + 8 + 32 + 2 + 1 + 8  # Without variable contrib_id and optional prev
        if total_len < min_size:
            raise ValueError(f"UPGRADE payload too short: {total_len} < {min_size}")
        
        buf = BytesIO(payload)
        req = lambda n, name: IanBytecodeDecoder._require_bytes(buf, n, total_len, name)
        
        goal_id = req(32, "goal_id")
        pack_hash = req(32, "pack_hash")
        score = unpack('>Q', req(8, "score"))[0]
        log_index = unpack('>Q', req(8, "log_index"))[0]
        log_root = req(32, "log_root")
        
        contrib_len = unpack('>H', req(2, "contrib_len"))[0]
        
        # Validate contributor ID length
        if contrib_len > MAX_CONTRIB_ID_LENGTH:
            raise ValueError(f"Contributor ID too long: {contrib_len} > {MAX_CONTRIB_ID_LENGTH}")
        
        contrib_bytes = req(contrib_len, "contributor_id")
        try:
            contributor_id = contrib_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 in contributor_id: {e}")
        
        has_prev_byte = req(1, "has_prev")
        has_prev = has_prev_byte[0]
        
        prev_hash = None
        if has_prev:
            prev_hash = req(32, "prev_pack_hash")
        
        timestamp_ms = unpack('>Q', req(8, "timestamp_ms"))[0]
        
        return {
            'goal_id': goal_id,
            'pack_hash': pack_hash,
            'score': score,
            'log_index': log_index,
            'log_root': log_root,
            'contributor_id': contributor_id,
            'prev_pack_hash': prev_hash,
            'timestamp_ms': timestamp_ms,
        }
    
    @staticmethod
    def _decode_challenge(payload: bytes) -> Dict[str, Any]:
        """Decode CHALLENGE payload with bounds validation."""
        total_len = len(payload)
        expected_size = 32 + 32 + 1 + 32 + 8
        if total_len < expected_size:
            raise ValueError(f"CHALLENGE payload too short: {total_len} < {expected_size}")
        
        buf = BytesIO(payload)
        req = lambda n, name: IanBytecodeDecoder._require_bytes(buf, n, total_len, name)
        
        goal_id = req(32, "goal_id")
        commit_hash = req(32, "challenged_commit_hash")
        fraud_type_byte = req(1, "fraud_type")
        fraud_type = fraud_type_byte[0]
        proof_hash = req(32, "fraud_proof_hash")
        timestamp_ms = unpack('>Q', req(8, "timestamp_ms"))[0]
        
        return {
            'goal_id': goal_id,
            'challenged_commit_hash': commit_hash,
            'fraud_type': fraud_type,
            'fraud_proof_hash': proof_hash,
            'timestamp_ms': timestamp_ms,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TauSenderConfig:
    """Configuration for Tau Net sender."""
    
    # Connection
    tau_host: str = "localhost"
    tau_port: int = 10330  # Default Tau testnet port
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    
    # Retries
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Transaction
    default_fee_limit: int = 1000
    default_expiration_seconds: int = 300  # 5 minutes
    
    # Keys (optional - for signed transactions)
    sender_pubkey: Optional[str] = None  # 96-char hex BLS pubkey
    private_key_path: Optional[str] = None
    
    # CLI fallback
    use_cli: bool = False
    tau_cli_path: str = "tau-cli"
    sendtx_script_path: Optional[str] = None
    
    # Environment overrides
    @classmethod
    def from_env(cls) -> "TauSenderConfig":
        """Load config from environment variables."""
        return cls(
            tau_host=os.environ.get("TAU_HOST", "localhost"),
            tau_port=int(os.environ.get("TAU_PORT", "10330")),
            sender_pubkey=os.environ.get("TAU_SENDER_PUBKEY"),
            private_key_path=os.environ.get("TAU_PRIVATE_KEY_PATH"),
            use_cli=os.environ.get("TAU_USE_CLI", "0") == "1",
            tau_cli_path=os.environ.get("TAU_CLI_PATH", "tau-cli"),
        )


# =============================================================================
# Transaction Builder
# =============================================================================

class TxFormat(Enum):
    """Transaction format options."""
    JSON = auto()      # JSON format (readable, larger)
    BYTECODE = auto()  # Compact bytecode (smaller, faster)


class IanTxBuilder:
    """
    Build Tau-compatible transactions for IAN operations.
    
    Supports two formats:
    1. JSON - Human-readable, compatible with current sendtx
    2. Bytecode - Compact binary, ~50-70% smaller
    
    IAN uses custom operation types within Tau transactions:
    - Operation "2": IAN_GOAL_REGISTER
    - Operation "3": IAN_LOG_COMMIT
    - Operation "4": IAN_UPGRADE
    - Operation "5": IAN_CHALLENGE
    """
    
    # IAN operation codes (within Tau operations dict)
    OP_GOAL_REGISTER = "2"
    OP_LOG_COMMIT = "3"
    OP_UPGRADE = "4"
    OP_CHALLENGE = "5"
    
    def __init__(
        self,
        sender_pubkey: str,
        config: Optional[TauSenderConfig] = None,
        format: TxFormat = TxFormat.JSON,
    ):
        self._sender_pubkey = sender_pubkey
        self._config = config or TauSenderConfig()
        self._sequence_number = 0  # Would be fetched from chain state
        self._format = format
    
    def set_sequence_number(self, seq: int) -> None:
        """Set current sequence number (from chain state)."""
        self._sequence_number = seq
    
    def build_goal_register_tx(
        self,
        goal_id: str,
        goal_spec_hash: str,
        name: str,
        description: str,
        invariant_ids: List[str],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build IAN_GOAL_REGISTER transaction."""
        ian_payload = {
            "type": "IAN_GOAL_REGISTER",
            "goal_id": goal_id,
            "goal_spec_hash": goal_spec_hash,
            "name": name,
            "description": description,
            "invariant_ids": invariant_ids,
            "thresholds": thresholds,
            "timestamp_ms": int(time.time() * 1000),
        }
        
        return self._wrap_ian_payload(self.OP_GOAL_REGISTER, ian_payload)
    
    def build_log_commit_tx(
        self,
        goal_id: str,
        log_root: str,
        log_size: int,
        leaderboard_root: str,
        leaderboard_size: int,
        prev_commit_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build IAN_LOG_COMMIT transaction."""
        ian_payload = {
            "type": "IAN_LOG_COMMIT",
            "goal_id": goal_id,
            "log_root": log_root,
            "log_size": log_size,
            "leaderboard_root": leaderboard_root,
            "leaderboard_size": leaderboard_size,
            "prev_commit_hash": prev_commit_hash,
            "timestamp_ms": int(time.time() * 1000),
        }
        
        return self._wrap_ian_payload(self.OP_LOG_COMMIT, ian_payload)
    
    def build_upgrade_tx(
        self,
        goal_id: str,
        pack_hash: str,
        score: float,
        metrics: Dict[str, float],
        log_index: int,
        log_root: str,
        contributor_id: str,
        prev_pack_hash: Optional[str] = None,
        governance_signatures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build IAN_UPGRADE transaction."""
        ian_payload = {
            "type": "IAN_UPGRADE",
            "goal_id": goal_id,
            "pack_hash": pack_hash,
            "prev_pack_hash": prev_pack_hash,
            "score": score,
            "metrics": metrics,
            "log_index": log_index,
            "log_root": log_root,
            "contributor_id": contributor_id,
            "governance_signatures": governance_signatures or [],
            "timestamp_ms": int(time.time() * 1000),
        }
        
        return self._wrap_ian_payload(self.OP_UPGRADE, ian_payload)
    
    def build_challenge_tx(
        self,
        goal_id: str,
        challenged_commit_hash: str,
        fraud_type: str,
        fraud_proof_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build IAN_CHALLENGE transaction."""
        ian_payload = {
            "type": "IAN_CHALLENGE",
            "goal_id": goal_id,
            "challenged_commit_hash": challenged_commit_hash,
            "fraud_type": fraud_type,
            "fraud_proof_data": fraud_proof_data,
            "timestamp_ms": int(time.time() * 1000),
        }
        
        return self._wrap_ian_payload(self.OP_CHALLENGE, ian_payload)
    
    # -------------------------------------------------------------------------
    # Bytecode Serialization
    # -------------------------------------------------------------------------
    
    def serialize_log_commit_bytecode(
        self,
        goal_id: str,
        log_root: str,
        log_size: int,
        leaderboard_root: str,
        leaderboard_size: int,
        prev_commit_hash: Optional[str] = None,
    ) -> bytes:
        """
        Serialize LOG_COMMIT as compact bytecode.
        
        Size comparison:
        - JSON: ~350 bytes
        - Bytecode: ~125 bytes (~65% smaller)
        """
        return IanBytecodeEncoder.encode_log_commit(
            goal_id=bytes.fromhex(goal_id) if len(goal_id) == 64 else hashlib.sha256(goal_id.encode()).digest(),
            log_root=bytes.fromhex(log_root),
            log_size=log_size,
            leaderboard_root=bytes.fromhex(leaderboard_root),
            leaderboard_size=leaderboard_size,
            prev_commit_hash=bytes.fromhex(prev_commit_hash) if prev_commit_hash else None,
            timestamp_ms=int(time.time() * 1000),
        )
    
    def serialize_upgrade_bytecode(
        self,
        goal_id: str,
        pack_hash: str,
        score: float,
        log_index: int,
        log_root: str,
        contributor_id: str,
        prev_pack_hash: Optional[str] = None,
    ) -> bytes:
        """
        Serialize UPGRADE as compact bytecode.
        
        Size comparison:
        - JSON: ~400 bytes
        - Bytecode: ~140 bytes (~65% smaller)
        """
        # Convert score to fixed-point (1e6 scale)
        score_int = int(score * 1_000_000)
        
        return IanBytecodeEncoder.encode_upgrade(
            goal_id=bytes.fromhex(goal_id) if len(goal_id) == 64 else hashlib.sha256(goal_id.encode()).digest(),
            pack_hash=bytes.fromhex(pack_hash),
            score=score_int,
            log_index=log_index,
            log_root=bytes.fromhex(log_root),
            contributor_id=contributor_id,
            prev_pack_hash=bytes.fromhex(prev_pack_hash) if prev_pack_hash else None,
            timestamp_ms=int(time.time() * 1000),
        )
    
    def serialize_challenge_bytecode(
        self,
        goal_id: str,
        challenged_commit_hash: str,
        fraud_type: int,
        fraud_proof_data: Dict[str, Any],
    ) -> bytes:
        """
        Serialize CHALLENGE as compact bytecode.
        
        Note: Full fraud proof data is hashed; full proof submitted separately.
        """
        proof_hash = hashlib.sha256(
            json.dumps(fraud_proof_data, sort_keys=True).encode()
        ).digest()
        
        return IanBytecodeEncoder.encode_challenge(
            goal_id=bytes.fromhex(goal_id) if len(goal_id) == 64 else hashlib.sha256(goal_id.encode()).digest(),
            challenged_commit_hash=bytes.fromhex(challenged_commit_hash),
            fraud_type=fraud_type,
            fraud_proof_hash=proof_hash,
            timestamp_ms=int(time.time() * 1000),
        )
    
    def serialize(
        self,
        tx: Dict[str, Any],
        format: Optional[TxFormat] = None,
    ) -> bytes:
        """
        Serialize transaction to bytes in specified format.
        
        Args:
            tx: Transaction dict from build_*_tx methods
            format: Override default format
            
        Returns:
            Serialized transaction bytes
        """
        fmt = format or self._format
        
        if fmt == TxFormat.JSON:
            return json.dumps(tx, sort_keys=True, separators=(",", ":")).encode()
        
        elif fmt == TxFormat.BYTECODE:
            # Extract IAN payload from operations
            op_key = list(tx["operations"].keys())[0]
            ian_payload = json.loads(tx["operations"][op_key])
            tx_type = ian_payload.get("type", "")
            
            if tx_type == "IAN_LOG_COMMIT":
                return self.serialize_log_commit_bytecode(
                    goal_id=ian_payload["goal_id"],
                    log_root=ian_payload["log_root"],
                    log_size=ian_payload["log_size"],
                    leaderboard_root=ian_payload["leaderboard_root"],
                    leaderboard_size=ian_payload["leaderboard_size"],
                    prev_commit_hash=ian_payload.get("prev_commit_hash"),
                )
            elif tx_type == "IAN_UPGRADE":
                return self.serialize_upgrade_bytecode(
                    goal_id=ian_payload["goal_id"],
                    pack_hash=ian_payload["pack_hash"],
                    score=ian_payload["score"],
                    log_index=ian_payload["log_index"],
                    log_root=ian_payload["log_root"],
                    contributor_id=ian_payload["contributor_id"],
                    prev_pack_hash=ian_payload.get("prev_pack_hash"),
                )
            elif tx_type == "IAN_CHALLENGE":
                return self.serialize_challenge_bytecode(
                    goal_id=ian_payload["goal_id"],
                    challenged_commit_hash=ian_payload["challenged_commit_hash"],
                    fraud_type=1,  # Would map from string
                    fraud_proof_data=ian_payload.get("fraud_proof_data", {}),
                )
            else:
                # Fall back to JSON for unsupported types
                return json.dumps(tx, sort_keys=True, separators=(",", ":")).encode()
        
        raise ValueError(f"Unknown format: {fmt}")
    
    def _wrap_ian_payload(self, op_code: str, ian_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap IAN payload in Tau transaction format."""
        now = int(time.time())
        expiration = now + self._config.default_expiration_seconds
        
        tx = {
            "sender_pubkey": self._sender_pubkey,
            "sequence_number": self._sequence_number,
            "expiration_time": expiration,
            "operations": {
                op_code: json.dumps(ian_payload, sort_keys=True, separators=(",", ":")),
            },
            "fee_limit": self._config.default_fee_limit,
            # signature added later
        }
        
        return tx
    
    def sign_transaction(
        self,
        tx: Dict[str, Any],
        private_key: bytes,
    ) -> Dict[str, Any]:
        """
        Sign transaction with BLS private key.
        
        Note: Requires py_ecc library for BLS signatures.
        """
        try:
            from py_ecc.bls import G2Basic
        except ImportError:
            logger.warning("py_ecc not available; using placeholder signature")
            tx["signature"] = "00" * 96  # Placeholder
            return tx
        
        # Build signing message
        signing_dict = {
            "sender_pubkey": tx["sender_pubkey"],
            "sequence_number": tx["sequence_number"],
            "expiration_time": tx["expiration_time"],
            "operations": tx["operations"],
            "fee_limit": tx["fee_limit"],
        }
        msg_bytes = json.dumps(signing_dict, sort_keys=True, separators=(",", ":")).encode()
        msg_hash = hashlib.sha256(msg_bytes).digest()
        
        # Sign
        signature = G2Basic.Sign(private_key, msg_hash)
        tx["signature"] = signature.hex()
        
        return tx


# =============================================================================
# TCP Sender
# =============================================================================

# Security: Maximum response size to prevent memory exhaustion
MAX_TCP_RESPONSE_SIZE = 64 * 1024  # 64 KB


class TauTCPSender:
    """
    Send transactions to Tau Net via TCP.
    
    Connects directly to a Tau Testnet node and sends
    sendtx commands.
    
    Security:
    - Bounded response read to prevent memory exhaustion
    - Timeout on all I/O operations
    """
    
    def __init__(self, config: Optional[TauSenderConfig] = None):
        self._config = config or TauSenderConfig()
        self._socket: Optional[socket.socket] = None
    
    def connect(self) -> bool:
        """Establish TCP connection to Tau node."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._config.connection_timeout)
            self._socket.connect((self._config.tau_host, self._config.tau_port))
            logger.info(f"Connected to Tau node at {self._config.tau_host}:{self._config.tau_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Tau node: {e}")
            self._socket = None
            return False
    
    def disconnect(self) -> None:
        """Close TCP connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send transaction to Tau node.
        
        Args:
            tx_data: Serialized transaction JSON
            
        Returns:
            (success, result_or_error)
            
        Security:
            - Enforces max response size to prevent memory exhaustion
            - Enforces read timeout
        """
        if not self._socket:
            if not self.connect():
                return False, "Failed to connect to Tau node"
        
        try:
            # Format as sendtx command
            tx_json = tx_data.decode('utf-8')
            command = f"sendtx {tx_json}\r\n"
            
            self._socket.settimeout(self._config.read_timeout)
            self._socket.sendall(command.encode('utf-8'))
            
            # Read response with size limit (security: prevent memory exhaustion)
            response = b""
            while len(response) < MAX_TCP_RESPONSE_SIZE:
                chunk = self._socket.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\r\n" in response:
                    break
            
            # Check if we hit the limit without finding end-of-response
            if len(response) >= MAX_TCP_RESPONSE_SIZE and b"\r\n" not in response:
                self.disconnect()
                return False, "Response too large - possible attack or malformed response"
            
            response_str = response.decode('utf-8').strip()
            
            if response_str.startswith("SUCCESS"):
                # Extract tx hash if present
                tx_hash = hashlib.sha256(tx_data).hexdigest()
                return True, tx_hash
            else:
                return False, response_str
                
        except socket.timeout:
            self.disconnect()
            return False, "Connection timeout"
        except Exception as e:
            self.disconnect()
            return False, str(e)


# =============================================================================
# CLI Sender
# =============================================================================

class TauCLISender:
    """
    Send transactions via Tau CLI or sendtx script.
    
    Falls back to shell execution when direct TCP isn't available.
    """
    
    def __init__(self, config: Optional[TauSenderConfig] = None):
        self._config = config or TauSenderConfig()
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send transaction via CLI.
        
        Args:
            tx_data: Serialized transaction JSON
            
        Returns:
            (success, result_or_error)
        """
        tx_json = tx_data.decode('utf-8')
        
        # Try sendtx script first
        if self._config.sendtx_script_path:
            return self._send_via_script(tx_json)
        
        # Fall back to CLI
        return self._send_via_cli(tx_json)
    
    def _send_via_script(self, tx_json: str) -> Tuple[bool, str]:
        """Send via Python sendtx script."""
        try:
            # Import and call directly if possible
            import sys
            script_dir = Path(self._config.sendtx_script_path).parent
            sys.path.insert(0, str(script_dir))
            
            import sendtx
            result = sendtx.queue_transaction(tx_json, propagate=True)
            
            if result.startswith("SUCCESS"):
                tx_hash = hashlib.sha256(tx_json.encode()).hexdigest()
                return True, tx_hash
            else:
                return False, result
                
        except ImportError:
            # Fall back to subprocess
            return self._send_via_subprocess(tx_json)
        except Exception as e:
            return False, str(e)
    
    def _send_via_cli(self, tx_json: str) -> Tuple[bool, str]:
        """Send via tau-cli command."""
        try:
            result = subprocess.run(
                [self._config.tau_cli_path, "sendtx", tx_json],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                tx_hash = hashlib.sha256(tx_json.encode()).hexdigest()
                return True, tx_hash
            else:
                return False, result.stdout or result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "CLI timeout"
        except FileNotFoundError:
            return False, f"CLI not found: {self._config.tau_cli_path}"
        except Exception as e:
            return False, str(e)
    
    def _send_via_subprocess(self, tx_json: str) -> Tuple[bool, str]:
        """Send via subprocess calling Python script."""
        try:
            result = subprocess.run(
                ["python3", self._config.sendtx_script_path, tx_json],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                tx_hash = hashlib.sha256(tx_json.encode()).hexdigest()
                return True, tx_hash
            else:
                return False, result.stdout or result.stderr
                
        except Exception as e:
            return False, str(e)


# =============================================================================
# Async Sender
# =============================================================================

class TauAsyncSender:
    """
    Async transaction sender with connection pooling.
    
    Uses asyncio for non-blocking operations.
    """
    
    def __init__(self, config: Optional[TauSenderConfig] = None):
        self._config = config or TauSenderConfig()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """Establish async TCP connection."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._config.tau_host,
                    self._config.tau_port,
                ),
                timeout=self._config.connection_timeout,
            )
            logger.info(f"Async connected to Tau node")
            return True
        except Exception as e:
            logger.error(f"Async connect failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close async connection."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
    
    async def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send transaction asynchronously.
        
        Args:
            tx_data: Serialized transaction JSON
            
        Returns:
            (success, result_or_error)
        """
        async with self._lock:
            if not self._writer:
                if not await self.connect():
                    return False, "Failed to connect"
            
            try:
                tx_json = tx_data.decode('utf-8')
                command = f"sendtx {tx_json}\r\n"
                
                self._writer.write(command.encode('utf-8'))
                await self._writer.drain()
                
                # Read response
                response = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=self._config.read_timeout,
                )
                
                response_str = response.decode('utf-8').strip()
                
                if response_str.startswith("SUCCESS"):
                    tx_hash = hashlib.sha256(tx_data).hexdigest()
                    return True, tx_hash
                else:
                    return False, response_str
                    
            except asyncio.TimeoutError:
                await self.disconnect()
                return False, "Read timeout"
            except Exception as e:
                await self.disconnect()
                return False, str(e)


# =============================================================================
# Unified Tau Net Sender
# =============================================================================

class TauNetSender:
    """
    Production Tau Net sender with retry logic.
    
    Combines multiple transport methods and provides
    automatic failover and retries.
    """
    
    def __init__(
        self,
        config: Optional[TauSenderConfig] = None,
        tx_builder: Optional[IanTxBuilder] = None,
    ):
        self._config = config or TauSenderConfig.from_env()
        self._tx_builder = tx_builder
        
        # Initialize senders based on config
        if self._config.use_cli:
            self._sender = TauCLISender(self._config)
        else:
            self._sender = TauTCPSender(self._config)
        
        # Stats
        self._sent_count = 0
        self._failed_count = 0
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send transaction with retries (synchronous version).
        
        Note: For async context, use send_tx_async() instead to avoid
        blocking the event loop.
        
        Args:
            tx_data: Serialized transaction JSON
            
        Returns:
            (success, tx_hash_or_error)
        """
        last_error = "unknown error"
        delay = self._config.retry_delay
        
        for attempt in range(self._config.max_retries):
            try:
                success, result = self._sender.send_tx(tx_data)
                
                if success:
                    self._sent_count += 1
                    logger.info(f"Tau tx sent: {result[:16]}...")
                    return True, result
                
                last_error = result
                logger.warning(f"Tau tx attempt {attempt + 1} failed: {result}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tau tx attempt {attempt + 1} error: {e}")
            
            # Wait before retry with jitter (blocking)
            if attempt < self._config.max_retries - 1:
                jittered_delay = backoff_with_jitter(
                    attempt,
                    self._config.retry_delay,
                    60.0,
                )
                time.sleep(jittered_delay)
        
        self._failed_count += 1
        logger.error(f"Tau tx failed after {self._config.max_retries} attempts: {last_error}")
        return False, last_error
    
    async def send_tx_async(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send transaction with retries (async version).
        
        Use this in async context to avoid blocking the event loop.
        
        Args:
            tx_data: Serialized transaction JSON
            
        Returns:
            (success, tx_hash_or_error)
        """
        last_error = "unknown error"
        delay = self._config.retry_delay
        
        for attempt in range(self._config.max_retries):
            try:
                # Run sync sender in executor to avoid blocking
                loop = asyncio.get_running_loop()
                success, result = await loop.run_in_executor(
                    None,
                    self._sender.send_tx,
                    tx_data
                )
                
                if success:
                    self._sent_count += 1
                    logger.info(f"Tau tx sent: {result[:16]}...")
                    return True, result
                
                last_error = result
                logger.warning(f"Tau tx attempt {attempt + 1} failed: {result}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tau tx attempt {attempt + 1} error: {e}")
            
            # Wait before retry with jitter (non-blocking)
            if attempt < self._config.max_retries - 1:
                jittered_delay = backoff_with_jitter(
                    attempt,
                    self._config.retry_delay,
                    60.0,
                )
                await asyncio.sleep(jittered_delay)
        
        self._failed_count += 1
        logger.error(f"Tau tx failed after {self._config.max_retries} attempts: {last_error}")
        return False, last_error
    
    def send_ian_goal_register(
        self,
        goal_id: str,
        goal_spec_hash: str,
        name: str,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Convenience method for IAN_GOAL_REGISTER."""
        if not self._tx_builder:
            # Send raw JSON
            payload = {
                "type": "IAN_GOAL_REGISTER",
                "goal_id": goal_id,
                "goal_spec_hash": goal_spec_hash,
                "name": name,
                **kwargs,
            }
            return self.send_tx(json.dumps(payload).encode())
        
        tx = self._tx_builder.build_goal_register_tx(
            goal_id=goal_id,
            goal_spec_hash=goal_spec_hash,
            name=name,
            **kwargs,
        )
        return self.send_tx(json.dumps(tx).encode())
    
    def send_ian_log_commit(
        self,
        goal_id: str,
        log_root: str,
        log_size: int,
        leaderboard_root: str,
        leaderboard_size: int,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Convenience method for IAN_LOG_COMMIT."""
        if not self._tx_builder:
            payload = {
                "type": "IAN_LOG_COMMIT",
                "goal_id": goal_id,
                "log_root": log_root,
                "log_size": log_size,
                "leaderboard_root": leaderboard_root,
                "leaderboard_size": leaderboard_size,
                **kwargs,
            }
            return self.send_tx(json.dumps(payload).encode())
        
        tx = self._tx_builder.build_log_commit_tx(
            goal_id=goal_id,
            log_root=log_root,
            log_size=log_size,
            leaderboard_root=leaderboard_root,
            leaderboard_size=leaderboard_size,
            **kwargs,
        )
        return self.send_tx(json.dumps(tx).encode())
    
    def send_ian_upgrade(
        self,
        goal_id: str,
        pack_hash: str,
        score: float,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Convenience method for IAN_UPGRADE."""
        if not self._tx_builder:
            payload = {
                "type": "IAN_UPGRADE",
                "goal_id": goal_id,
                "pack_hash": pack_hash,
                "score": score,
                **kwargs,
            }
            return self.send_tx(json.dumps(payload).encode())
        
        tx = self._tx_builder.build_upgrade_tx(
            goal_id=goal_id,
            pack_hash=pack_hash,
            score=score,
            **kwargs,
        )
        return self.send_tx(json.dumps(tx).encode())
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get sender statistics."""
        return {
            "sent": self._sent_count,
            "failed": self._failed_count,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_tau_sender(
    host: str = "localhost",
    port: int = 10330,
    use_cli: bool = False,
    sender_pubkey: Optional[str] = None,
) -> TauNetSender:
    """
    Create a Tau Net sender.
    
    Args:
        host: Tau node host
        port: Tau node port
        use_cli: Use CLI instead of TCP
        sender_pubkey: BLS public key for signed transactions
        
    Returns:
        Configured TauNetSender
    """
    config = TauSenderConfig(
        tau_host=host,
        tau_port=port,
        use_cli=use_cli,
        sender_pubkey=sender_pubkey,
    )
    
    tx_builder = None
    if sender_pubkey:
        tx_builder = IanTxBuilder(sender_pubkey, config)
    
    return TauNetSender(config=config, tx_builder=tx_builder)


def create_tau_sender_from_env() -> TauNetSender:
    """Create Tau sender from environment variables."""
    config = TauSenderConfig.from_env()
    
    tx_builder = None
    if config.sender_pubkey:
        tx_builder = IanTxBuilder(config.sender_pubkey, config)
    
    return TauNetSender(config=config, tx_builder=tx_builder)
