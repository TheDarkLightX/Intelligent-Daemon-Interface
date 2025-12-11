"""
IAN TLS Support - Secure P2P connections with mutual TLS.

This module provides TLS/mTLS support for P2P connections:
1. Certificate generation for node identity
2. TLS context creation for server and client
3. Certificate validation and pinning
4. Secure handshake with peer verification

Usage:
    # Generate node certificate
    tls_config = TLSConfig.generate(node_id="my_node")
    
    # Create TLS context for server
    server_ctx = tls_config.create_server_context()
    
    # Create TLS context for client
    client_ctx = tls_config.create_client_context()
    
    # Use with asyncio streams
    reader, writer = await asyncio.open_connection(
        host, port, ssl=client_ctx
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import ssl
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Check for cryptography library
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519
    from cryptography.x509.oid import NameOID
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography library not available, TLS features limited")


# =============================================================================
# TLS Configuration
# =============================================================================

@dataclass
class TLSConfig:
    """
    TLS configuration for P2P connections.
    
    Stores certificate, private key, and trusted CA certificates.
    Supports both file paths and PEM-encoded strings.
    """
    
    # Node identity
    node_id: str = ""
    
    # Certificate paths (or PEM strings)
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    
    # PEM-encoded data (alternative to paths)
    cert_pem: Optional[str] = None
    key_pem: Optional[str] = None
    ca_pem: Optional[str] = None
    
    # TLS settings
    verify_peer: bool = True
    require_client_cert: bool = True  # mTLS
    min_version: str = "TLSv1_3"
    cipher_suites: List[str] = field(default_factory=list)
    
    # Certificate pinning
    pinned_certs: Set[str] = field(default_factory=set)  # SHA256 fingerprints
    
    # Validity
    cert_valid_days: int = 365
    
    @classmethod
    def generate(
        cls,
        node_id: str,
        output_dir: Optional[str] = None,
        valid_days: int = 365,
    ) -> "TLSConfig":
        """
        Generate new self-signed certificate for node.
        
        Args:
            node_id: Node identifier
            output_dir: Directory to save cert/key files (optional)
            valid_days: Certificate validity in days
            
        Returns:
            TLSConfig with generated certificate
        """
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library required for cert generation")
        
        # Generate private key (Ed25519 for compact signatures)
        private_key = ed25519.Ed25519PrivateKey.generate()
        
        # Create subject
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IAN Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"node-{node_id[:16]}"),
            x509.NameAttribute(NameOID.SERIAL_NUMBER, node_id),
        ])
        
        # Build certificate
        now = datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=valid_days))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(f"node-{node_id[:16]}.ian.local"),
                ]),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )
            .sign(private_key, None)  # Ed25519 doesn't use hash
        )
        
        # Serialize to PEM
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        
        config = cls(
            node_id=node_id,
            cert_pem=cert_pem,
            key_pem=key_pem,
            cert_valid_days=valid_days,
        )
        
        # Save to files if output_dir specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            cert_file = output_path / f"{node_id[:16]}_cert.pem"
            key_file = output_path / f"{node_id[:16]}_key.pem"
            
            cert_file.write_text(cert_pem)
            key_file.write_text(key_pem)
            os.chmod(key_file, 0o600)  # Restrict key permissions
            
            config.cert_path = str(cert_file)
            config.key_path = str(key_file)
            
            logger.info(f"Generated TLS certificate: {cert_file}")
        
        return config
    
    @classmethod
    def load(cls, cert_path: str, key_path: str, ca_path: Optional[str] = None) -> "TLSConfig":
        """Load TLS config from files."""
        config = cls(
            cert_path=cert_path,
            key_path=key_path,
            ca_path=ca_path,
        )
        
        # Extract node_id from certificate
        if HAS_CRYPTO:
            cert_pem = Path(cert_path).read_bytes()
            cert = x509.load_pem_x509_certificate(cert_pem)
            
            for attr in cert.subject:
                if attr.oid == NameOID.SERIAL_NUMBER:
                    config.node_id = attr.value
                    break
        
        return config
    
    def create_server_context(self) -> ssl.SSLContext:
        """
        Create SSL context for server (accepting connections).
        
        Returns:
            Configured SSLContext for server use
        """
        # Create context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Set minimum version
        ctx.minimum_version = getattr(ssl.TLSVersion, self.min_version, ssl.TLSVersion.TLSv1_3)
        
        # Load certificate and key
        self._load_cert_key(ctx)
        
        # Client certificate verification (mTLS)
        if self.require_client_cert:
            ctx.verify_mode = ssl.CERT_REQUIRED
            self._load_ca(ctx)
        else:
            ctx.verify_mode = ssl.CERT_OPTIONAL
        
        # Cipher suites
        if self.cipher_suites:
            ctx.set_ciphers(":".join(self.cipher_suites))
        
        return ctx
    
    def create_client_context(self, verify_hostname: bool = False) -> ssl.SSLContext:
        """
        Create SSL context for client (initiating connections).
        
        Args:
            verify_hostname: Whether to verify server hostname
            
        Returns:
            Configured SSLContext for client use
        """
        # Create context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        # Set minimum version
        ctx.minimum_version = getattr(ssl.TLSVersion, self.min_version, ssl.TLSVersion.TLSv1_3)
        
        # Load certificate and key for mTLS
        self._load_cert_key(ctx)
        
        # Server verification
        if self.verify_peer:
            ctx.verify_mode = ssl.CERT_REQUIRED
            self._load_ca(ctx)
        else:
            ctx.verify_mode = ssl.CERT_NONE
        
        ctx.check_hostname = verify_hostname
        
        # Cipher suites
        if self.cipher_suites:
            ctx.set_ciphers(":".join(self.cipher_suites))
        
        return ctx
    
    def _load_cert_key(self, ctx: ssl.SSLContext) -> None:
        """Load certificate and key into context."""
        if self.cert_path and self.key_path:
            ctx.load_cert_chain(self.cert_path, self.key_path)
        elif self.cert_pem and self.key_pem:
            # Write to temp files (ssl module requires files)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as cert_file:
                cert_file.write(self.cert_pem)
                cert_path = cert_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as key_file:
                key_file.write(self.key_pem)
                key_path = key_file.name
            
            try:
                ctx.load_cert_chain(cert_path, key_path)
            finally:
                os.unlink(cert_path)
                os.unlink(key_path)
    
    def _load_ca(self, ctx: ssl.SSLContext) -> None:
        """Load CA certificates into context."""
        if self.ca_path:
            ctx.load_verify_locations(self.ca_path)
        elif self.ca_pem:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as ca_file:
                ca_file.write(self.ca_pem)
                ca_path = ca_file.name
            
            try:
                ctx.load_verify_locations(ca_path)
            finally:
                os.unlink(ca_path)
        else:
            # Use self-signed cert as CA
            if self.cert_path:
                ctx.load_verify_locations(self.cert_path)
            elif self.cert_pem:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as cert_file:
                    cert_file.write(self.cert_pem)
                    cert_path = cert_file.name
                
                try:
                    ctx.load_verify_locations(cert_path)
                finally:
                    os.unlink(cert_path)
    
    def get_fingerprint(self) -> str:
        """
        Get SHA256 fingerprint of certificate.
        
        Returns:
            Hex-encoded fingerprint
        """
        if not HAS_CRYPTO:
            return ""
        
        cert_pem = self.cert_pem
        if not cert_pem and self.cert_path:
            cert_pem = Path(self.cert_path).read_text()
        
        if not cert_pem:
            return ""
        
        cert = x509.load_pem_x509_certificate(cert_pem.encode())
        fingerprint = cert.fingerprint(hashes.SHA256())
        return fingerprint.hex()
    
    def add_pinned_cert(self, fingerprint: str) -> None:
        """Add a pinned certificate fingerprint."""
        self.pinned_certs.add(fingerprint.lower())
    
    def verify_pinned(self, peer_cert: bytes) -> bool:
        """
        Verify peer certificate against pinned certificates.
        
        Args:
            peer_cert: DER-encoded peer certificate
            
        Returns:
            True if certificate is pinned or no pins configured
        """
        if not self.pinned_certs:
            return True
        
        fingerprint = hashlib.sha256(peer_cert).hexdigest()
        return fingerprint.lower() in self.pinned_certs


# =============================================================================
# TLS Transport
# =============================================================================

class TLSTransport:
    """
    TLS-secured TCP transport for P2P connections.
    
    Provides encrypted and authenticated communication.
    """
    
    def __init__(self, config: TLSConfig):
        self._config = config
        self._server: Optional[asyncio.Server] = None
        self._connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
    
    async def start_server(
        self,
        host: str,
        port: int,
        on_connect: callable,
    ) -> None:
        """
        Start TLS server.
        
        Args:
            host: Bind address
            port: Bind port
            on_connect: Callback for new connections
        """
        ssl_ctx = self._config.create_server_context()
        
        self._server = await asyncio.start_server(
            on_connect,
            host,
            port,
            ssl=ssl_ctx,
        )
        
        logger.info(f"TLS server listening on {host}:{port}")
    
    async def stop_server(self) -> None:
        """Stop TLS server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
    
    async def connect(
        self,
        host: str,
        port: int,
        timeout: float = 10.0,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """
        Connect to TLS server.
        
        Args:
            host: Server host
            port: Server port
            timeout: Connection timeout
            
        Returns:
            Tuple of (reader, writer)
        """
        ssl_ctx = self._config.create_client_context()
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=timeout,
        )
        
        # Verify peer certificate if pinning enabled
        if self._config.pinned_certs:
            ssl_object = writer.get_extra_info('ssl_object')
            if ssl_object:
                peer_cert = ssl_object.getpeercert(binary_form=True)
                if peer_cert and not self._config.verify_pinned(peer_cert):
                    writer.close()
                    await writer.wait_closed()
                    raise ssl.SSLError("Peer certificate not in pinned set")
        
        return reader, writer
    
    def get_peer_node_id(self, writer: asyncio.StreamWriter) -> Optional[str]:
        """
        Extract node ID from peer's TLS certificate.
        
        Args:
            writer: Stream writer with TLS connection
            
        Returns:
            Node ID or None
        """
        if not HAS_CRYPTO:
            return None
        
        ssl_object = writer.get_extra_info('ssl_object')
        if not ssl_object:
            return None
        
        peer_cert_der = ssl_object.getpeercert(binary_form=True)
        if not peer_cert_der:
            return None
        
        try:
            cert = x509.load_der_x509_certificate(peer_cert_der)
            for attr in cert.subject:
                if attr.oid == NameOID.SERIAL_NUMBER:
                    return attr.value
        except Exception as e:
            logger.warning(f"Failed to extract node ID from cert: {e}")
        
        return None


# =============================================================================
# Certificate Authority (for network-wide trust)
# =============================================================================

class CertificateAuthority:
    """
    Simple CA for issuing node certificates.
    
    For production, use a proper PKI or external CA.
    """
    
    def __init__(
        self,
        ca_cert_path: Optional[str] = None,
        ca_key_path: Optional[str] = None,
    ):
        self._ca_cert = None
        self._ca_key = None
        
        if ca_cert_path and ca_key_path:
            self._load(ca_cert_path, ca_key_path)
    
    def _load(self, cert_path: str, key_path: str) -> None:
        """Load CA certificate and key."""
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library required")
        
        cert_pem = Path(cert_path).read_bytes()
        key_pem = Path(key_path).read_bytes()
        
        self._ca_cert = x509.load_pem_x509_certificate(cert_pem)
        self._ca_key = serialization.load_pem_private_key(key_pem, password=None)
    
    @classmethod
    def generate(cls, output_dir: str, valid_days: int = 3650) -> "CertificateAuthority":
        """
        Generate new CA certificate.
        
        Args:
            output_dir: Directory to save CA files
            valid_days: CA validity in days
            
        Returns:
            CertificateAuthority instance
        """
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library required")
        
        # Generate CA key
        ca_key = ec.generate_private_key(ec.SECP256R1())
        
        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IAN Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, "IAN Root CA"),
        ])
        
        now = datetime.utcnow()
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=valid_days))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=1),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        
        # Save files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cert_file = output_path / "ca_cert.pem"
        key_file = output_path / "ca_key.pem"
        
        cert_file.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        key_file.write_bytes(ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        os.chmod(key_file, 0o600)
        
        logger.info(f"Generated CA certificate: {cert_file}")
        
        ca = cls()
        ca._ca_cert = ca_cert
        ca._ca_key = ca_key
        return ca
    
    def issue_certificate(
        self,
        node_id: str,
        valid_days: int = 365,
    ) -> Tuple[str, str]:
        """
        Issue certificate for a node.
        
        Args:
            node_id: Node identifier
            valid_days: Certificate validity
            
        Returns:
            Tuple of (cert_pem, key_pem)
        """
        if not HAS_CRYPTO or not self._ca_cert or not self._ca_key:
            raise RuntimeError("CA not initialized")
        
        # Generate node key
        node_key = ed25519.Ed25519PrivateKey.generate()
        
        # Create node certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IAN Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"node-{node_id[:16]}"),
            x509.NameAttribute(NameOID.SERIAL_NUMBER, node_id),
        ])
        
        now = datetime.utcnow()
        node_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca_cert.subject)
            .public_key(node_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=valid_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )
            .sign(self._ca_key, hashes.SHA256())
        )
        
        cert_pem = node_cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = node_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        
        return cert_pem, key_pem
    
    def get_ca_pem(self) -> str:
        """Get CA certificate in PEM format."""
        if not self._ca_cert:
            return ""
        return self._ca_cert.public_bytes(serialization.Encoding.PEM).decode()


# =============================================================================
# Utility Functions
# =============================================================================

def create_self_signed_tls_config(node_id: str) -> TLSConfig:
    """
    Create TLS config with self-signed certificate.
    
    Convenience function for development/testing.
    """
    return TLSConfig.generate(node_id)


def verify_peer_certificate(
    ssl_object,
    expected_node_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verify peer certificate.
    
    Args:
        ssl_object: SSL object from connection
        expected_node_id: Expected node ID (optional)
        
    Returns:
        Tuple of (is_valid, node_id or error)
    """
    if not ssl_object:
        return False, "No SSL connection"
    
    peer_cert = ssl_object.getpeercert()
    if not peer_cert:
        return False, "No peer certificate"
    
    # Check if certificate is still valid
    # (This is handled by SSL library, but we can add extra checks)
    
    # Extract node ID
    if HAS_CRYPTO:
        peer_cert_der = ssl_object.getpeercert(binary_form=True)
        if peer_cert_der:
            try:
                cert = x509.load_der_x509_certificate(peer_cert_der)
                for attr in cert.subject:
                    if attr.oid == NameOID.SERIAL_NUMBER:
                        node_id = attr.value
                        
                        if expected_node_id and node_id != expected_node_id:
                            return False, f"Node ID mismatch: {node_id}"
                        
                        return True, node_id
            except Exception as e:
                return False, f"Certificate parse error: {e}"
    
    return True, "unknown"
