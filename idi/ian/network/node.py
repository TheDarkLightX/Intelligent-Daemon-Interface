"""
IAN Node Identity - Node identification and key management.

Provides:
1. Cryptographic node identity (Ed25519 keys)
2. Node capabilities advertisement
3. Signature generation/verification

Security:
- Private keys never leave the node
- All messages are signed
- Node IDs are derived from public keys
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .protocol import Message

# Ed25519 cryptography is a HARD REQUIREMENT - no fallback
# This ensures all environments use real cryptographic signatures
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True  # Always True now - kept for backwards compatibility
except ImportError as e:
    raise ImportError(
        "SECURITY ERROR: The 'cryptography' library is required for Ed25519 signatures. "
        "Install with: pip install cryptography\n"
        "This is a hard requirement for all environments (dev, test, prod)."
    ) from e


# =============================================================================
# Node Identity
# =============================================================================

_KEYRING_REF_PREFIX = "keyring://"
_DEFAULT_KEYRING_SERVICE = "idi.ian"


def _atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    """
    Atomically write text to a file with restrictive permissions.

    Security goals:
    - Avoid transient world/group-readable perms (create with mode=0o600).
    - Avoid partially-written identity files (write temp + fsync + rename).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )

    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))

        # Best-effort fsync on directory for durability (POSIX).
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class NodeCapabilities:
    """
    Node capabilities advertisement.
    
    Tells other nodes what this node can do.
    """
    # Core capabilities
    accepts_contributions: bool = True
    serves_leaderboard: bool = True
    serves_log_proofs: bool = True
    
    # Supported goals
    goal_ids: List[str] = field(default_factory=list)
    
    # Resource limits
    max_contribution_size: int = 10 * 1024 * 1024  # 10 MB
    max_proof_size: int = 1 * 1024 * 1024  # 1 MB
    
    # Version
    protocol_version: str = "1.0"
    software_version: str = "0.1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCapabilities":
        return cls(**data)


@dataclass
class NodeInfo:
    """
    Public node information shared with peers.
    """
    node_id: str
    public_key: bytes
    addresses: List[str]  # e.g., ["tcp://1.2.3.4:9000", "ws://1.2.3.4:9001"]
    capabilities: NodeCapabilities
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "public_key": base64.b64encode(self.public_key).decode(),
            "addresses": self.addresses,
            "capabilities": self.capabilities.to_dict(),
            "timestamp": self.timestamp,
            "signature": base64.b64encode(self.signature).decode() if self.signature else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            node_id=data["node_id"],
            public_key=base64.b64decode(data["public_key"]),
            addresses=data["addresses"],
            capabilities=NodeCapabilities.from_dict(data["capabilities"]),
            timestamp=data["timestamp"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
        )
    
    def signing_payload(self) -> bytes:
        """Get the payload to sign (excludes signature)."""
        data = {
            "node_id": self.node_id,
            "public_key": base64.b64encode(self.public_key).decode(),
            "addresses": self.addresses,
            "capabilities": self.capabilities.to_dict(),
            "timestamp": self.timestamp,
        }
        return json.dumps(data, sort_keys=True).encode()


class NodeIdentity:
    """
    Node identity with cryptographic keys.
    
    Manages:
    - Ed25519 key pair generation/loading
    - Node ID derivation (hash of public key)
    - Message signing
    - Signature verification
    
    Invariants:
    - Private key is never exposed
    - Node ID is deterministic from public key
    """
    
    def __init__(
        self,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
    ):
        """
        Initialize node identity.
        
        Args:
            private_key: 32-byte Ed25519 private key (generates new if None)
            public_key: 32-byte Ed25519 public key (derived from private if None)
        """
        if private_key:
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
            self._public_key = self._private_key.public_key()
        elif public_key:
            self._private_key = None
            self._public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        else:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        
        self._public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        # Node ID = first 20 bytes of SHA-256(public_key), hex encoded
        self._node_id = hashlib.sha256(self._public_key_bytes).hexdigest()[:40]
    
    @property
    def node_id(self) -> str:
        """Get node ID (40-char hex string)."""
        return self._node_id
    
    @property
    def public_key(self) -> bytes:
        """Get public key bytes."""
        return self._public_key_bytes
    
    def has_private_key(self) -> bool:
        """Check if this identity has a private key (can sign)."""
        return self._private_key is not None
    
    def sign(self, data: bytes) -> bytes:
        """
        Sign data with private key.
        
        Args:
            data: Data to sign
            
        Returns:
            64-byte Ed25519 signature
            
        Raises:
            ValueError: If no private key available
        """
        if not self.has_private_key():
            raise ValueError("Cannot sign without private key")
        
        return self._private_key.sign(data)
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self._public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
    
    def sign_message(self, message: "Message") -> None:
        if not self.has_private_key():
            raise ValueError("Cannot sign message without private key")
        if message.sender_id != self.node_id:
            raise ValueError("Message sender_id must match identity.node_id")
        message.signature = self.sign(message.signing_payload())
    
    @classmethod
    def verify_with_public_key(cls, public_key: bytes, data: bytes, signature: bytes) -> bool:
        """Verify signature using a public key."""
        try:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
            pk.verify(signature, data)
            return True
        except (InvalidSignature, Exception):
            return False
    
    def create_node_info(
        self,
        addresses: List[str],
        capabilities: Optional[NodeCapabilities] = None,
    ) -> NodeInfo:
        """
        Create signed node info.
        
        Args:
            addresses: List of addresses this node is reachable at
            capabilities: Node capabilities (uses defaults if None)
            
        Returns:
            Signed NodeInfo
        """
        info = NodeInfo(
            node_id=self.node_id,
            public_key=self.public_key,
            addresses=addresses,
            capabilities=capabilities or NodeCapabilities(),
        )
        
        if self.has_private_key():
            info.signature = self.sign(info.signing_payload())
        
        return info
    
    def save(self, path: Path) -> None:
        """
        Save identity to file.
        
        Warning: This saves the private key! Protect this file.
        """
        if not self.has_private_key():
            raise ValueError("Cannot save identity without private key")

        private_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        
        # Save as JSON with base64-encoded key
        data = {
            "node_id": self.node_id,
            "private_key": base64.b64encode(private_bytes).decode(),
            "public_key": base64.b64encode(self.public_key).decode(),
        }

        _atomic_write_text(path, json.dumps(data, indent=2), mode=0o600)
    
    @classmethod
    def load(cls, path: Path) -> "NodeIdentity":
        """Load identity from file."""
        data = json.loads(path.read_text())
        private_key_b64 = data.get("private_key")
        if not isinstance(private_key_b64, str) or not private_key_b64:
            raise ValueError("Invalid identity file: missing private_key")
        private_key = base64.b64decode(private_key_b64, validate=True)
        if len(private_key) != 32:
            raise ValueError(f"Invalid identity file: private_key must be 32 bytes, got {len(private_key)}")

        identity = cls(private_key=private_key)

        # Best-effort integrity checks: do not trust stored node_id/public_key unless they match.
        stored_node_id = data.get("node_id")
        if isinstance(stored_node_id, str) and stored_node_id:
            if stored_node_id != identity.node_id:
                raise ValueError("Identity file integrity error: stored node_id does not match derived node_id")

        stored_public_key_b64 = data.get("public_key")
        if isinstance(stored_public_key_b64, str) and stored_public_key_b64:
            stored_public_key = base64.b64decode(stored_public_key_b64, validate=True)
            if stored_public_key != identity.public_key:
                raise ValueError(
                    "Identity file integrity error: stored public_key does not match derived public_key"
                )

        return identity

    @staticmethod
    def _parse_keyring_ref(ref: str) -> Tuple[str, str]:
        if not ref.startswith(_KEYRING_REF_PREFIX):
            raise ValueError(f"Not a keyring ref: {ref!r}")
        parsed = urlparse(ref)
        service = parsed.netloc or _DEFAULT_KEYRING_SERVICE
        account = (parsed.path or "").lstrip("/")
        if not account:
            raise ValueError(
                f"Invalid keyring ref (missing account): {ref!r}. "
                f"Expected {_KEYRING_REF_PREFIX}<service>/<account>"
            )
        return service, account

    def save_to_keyring(self, *, service: str = _DEFAULT_KEYRING_SERVICE, account: Optional[str] = None) -> None:
        """
        Save the private key into the OS keyring (optional dependency).

        Requires `keyring` to be installed and a suitable backend to be available.
        """
        if not self.has_private_key():
            raise ValueError("Cannot save identity without private key")
        if not account:
            account = self.node_id

        try:
            keyring = importlib.import_module("keyring")
        except ImportError as e:
            raise ImportError(
                "Optional dependency missing: 'keyring'. "
                "Install with: pip install 'idi[security]'"
            ) from e

        private_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        secret = base64.b64encode(private_bytes).decode("ascii")
        keyring.set_password(service, account, secret)

    @classmethod
    def load_from_keyring(cls, *, service: str = _DEFAULT_KEYRING_SERVICE, account: str) -> "NodeIdentity":
        """Load the private key from the OS keyring (optional dependency)."""
        if not account:
            raise ValueError("account is required")

        try:
            keyring = importlib.import_module("keyring")
        except ImportError as e:
            raise ImportError(
                "Optional dependency missing: 'keyring'. "
                "Install with: pip install 'idi[security]'"
            ) from e

        secret = keyring.get_password(service, account)
        if secret is None:
            raise FileNotFoundError(f"No keyring entry for service={service!r} account={account!r}")

        try:
            private_bytes = base64.b64decode(secret, validate=True)
        except Exception as e:
            raise ValueError("Invalid base64 private key in keyring") from e
        if len(private_bytes) != 32:
            raise ValueError(f"Invalid private key length in keyring: expected 32 bytes, got {len(private_bytes)}")

        return cls(private_key=private_bytes)

    def save_to_ref(self, ref: str | Path) -> None:
        """
        Save identity using either:
        - file path (default)
        - keyring ref: keyring://<service>/<account>
        """
        if isinstance(ref, Path):
            self.save(ref)
            return
        if ref.startswith(_KEYRING_REF_PREFIX):
            service, account = self._parse_keyring_ref(ref)
            self.save_to_keyring(service=service, account=account)
            return
        self.save(Path(ref))

    @classmethod
    def load_from_ref(cls, ref: str | Path) -> "NodeIdentity":
        """
        Load identity using either:
        - file path (default)
        - keyring ref: keyring://<service>/<account>
        """
        if isinstance(ref, Path):
            return cls.load(ref)
        if ref.startswith(_KEYRING_REF_PREFIX):
            service, account = cls._parse_keyring_ref(ref)
            return cls.load_from_keyring(service=service, account=account)
        return cls.load(Path(ref))
    
    @classmethod
    def generate(cls) -> "NodeIdentity":
        """Generate a new random identity."""
        return cls()


def verify_message_signature(info: NodeInfo, message: "Message") -> bool:
    if message.signature is None:
        return False
    if message.sender_id != info.node_id:
        return False
    return NodeIdentity.verify_with_public_key(
        info.public_key,
        message.signing_payload(),
        message.signature,
    )
