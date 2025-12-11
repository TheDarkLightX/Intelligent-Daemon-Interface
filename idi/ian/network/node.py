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
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import Message

# Use cryptography library if available, otherwise fallback to hashlib-based signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    import warnings
    warnings.warn(
        "cryptography library not available. Node identity will use INSECURE fallback. "
        "Install 'cryptography' for production use.",
        UserWarning,
        stacklevel=2,
    )


# =============================================================================
# Node Identity
# =============================================================================

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
        if HAS_CRYPTO:
            self._init_with_crypto(private_key, public_key)
        else:
            self._init_fallback(private_key, public_key)
    
    def _init_with_crypto(self, private_key: Optional[bytes], public_key: Optional[bytes]):
        """Initialize using cryptography library."""
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
    
    def _init_fallback(self, private_key: Optional[bytes], public_key: Optional[bytes]):
        """Fallback initialization without cryptography library."""
        if private_key:
            self._private_key_bytes = private_key
        else:
            self._private_key_bytes = secrets.token_bytes(32)
        
        # Derive "public key" as hash of private key (not real Ed25519)
        self._public_key_bytes = hashlib.sha256(self._private_key_bytes).digest()
        self._private_key = self._private_key_bytes
        self._public_key = self._public_key_bytes
        
        # Node ID
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
        
        if HAS_CRYPTO:
            return self._private_key.sign(data)
        else:
            # Fallback: HMAC-SHA256 (not a real signature, for testing only)
            import hmac
            return hmac.new(self._private_key_bytes, data, hashlib.sha256).digest() * 2
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if valid, False otherwise
        """
        if HAS_CRYPTO:
            try:
                self._public_key.verify(signature, data)
                return True
            except InvalidSignature:
                return False
        else:
            # Fallback verification
            import hmac
            expected = hmac.new(self._private_key_bytes, data, hashlib.sha256).digest() * 2
            return hmac.compare_digest(expected, signature)
    
    def sign_message(self, message: "Message") -> None:
        if not self.has_private_key():
            raise ValueError("Cannot sign message without private key")
        if message.sender_id != self.node_id:
            raise ValueError("Message sender_id must match identity.node_id")
        message.signature = self.sign(message.signing_payload())
    
    @classmethod
    def verify_with_public_key(cls, public_key: bytes, data: bytes, signature: bytes) -> bool:
        """Verify signature using a public key."""
        if HAS_CRYPTO:
            try:
                pk = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
                pk.verify(signature, data)
                return True
            except (InvalidSignature, Exception):
                return False
        else:
            # Fallback: cannot verify without cryptography support; treat as invalid
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
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_CRYPTO:
            private_bytes = self._private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        else:
            private_bytes = self._private_key_bytes
        
        # Save as JSON with base64-encoded key
        data = {
            "node_id": self.node_id,
            "private_key": base64.b64encode(private_bytes).decode(),
            "public_key": base64.b64encode(self.public_key).decode(),
        }
        
        path.write_text(json.dumps(data, indent=2))
        os.chmod(path, 0o600)  # Owner read/write only
    
    @classmethod
    def load(cls, path: Path) -> "NodeIdentity":
        """Load identity from file."""
        data = json.loads(path.read_text())
        private_key = base64.b64decode(data["private_key"])
        return cls(private_key=private_key)
    
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
