"""
Tests for IAN Network Module.

Tests cover:
- Node identity and key management
- P2P protocol messages
- Peer discovery
- API handlers
"""

import base64
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from idi.ian.network.node import (
    NodeIdentity,
    NodeInfo,
    NodeCapabilities,
    verify_message_signature,
    HAS_CRYPTO,
)
from idi.ian.network.protocol import (
    Message,
    MessageType,
    ContributionAnnounce,
    ContributionRequest,
    ContributionResponse,
    StateRequest,
    StateResponse,
    PeerExchange,
    Ping,
    Pong,
)
from idi.ian.network.discovery import SeedNodeDiscovery, PeerState
from idi.ian.network.api import IANApiHandlers, ApiConfig, success_response, error_response


# =============================================================================
# Node Identity Tests
# =============================================================================

class TestNodeIdentity:
    """Tests for NodeIdentity."""
    
    def test_generate_identity(self):
        """Can generate a new identity."""
        identity = NodeIdentity.generate()
        
        assert identity.node_id is not None
        assert len(identity.node_id) == 40  # 20 bytes hex
        assert identity.public_key is not None
        assert identity.has_private_key()
    
    def test_unique_identities(self):
        """Each generated identity is unique."""
        id1 = NodeIdentity.generate()
        id2 = NodeIdentity.generate()
        
        assert id1.node_id != id2.node_id
        assert id1.public_key != id2.public_key
    
    def test_sign_and_verify(self):
        """Can sign and verify data."""
        identity = NodeIdentity.generate()
        
        data = b"test message"
        signature = identity.sign(data)
        
        assert signature is not None
        assert len(signature) == 64  # Ed25519 signature
        assert identity.verify(data, signature)
    
    def test_verify_wrong_data_fails(self):
        """Verification fails for wrong data."""
        identity = NodeIdentity.generate()
        
        data = b"test message"
        signature = identity.sign(data)
        
        wrong_data = b"wrong message"
        assert not identity.verify(wrong_data, signature)
    
    def test_save_and_load(self):
        """Can save and load identity."""
        identity = NodeIdentity.generate()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            identity.save(path)

            # Ensure the private key file is not group/world readable (POSIX).
            if os.name == "posix":
                assert (path.stat().st_mode & 0o777) == 0o600
            
            loaded = NodeIdentity.load(path)
            
            assert loaded.node_id == identity.node_id
            assert loaded.public_key == identity.public_key
    
    def test_create_node_info(self):
        """Can create signed node info."""
        identity = NodeIdentity.generate()
        
        info = identity.create_node_info(
            addresses=["tcp://127.0.0.1:9000"],
            capabilities=NodeCapabilities(goal_ids=["GOAL_1"]),
        )
        
        assert info.node_id == identity.node_id
        assert info.public_key == identity.public_key
        assert "tcp://127.0.0.1:9000" in info.addresses
        assert info.signature is not None


class TestNodeInfo:
    """Tests for NodeInfo."""
    
    def test_to_dict_and_back(self):
        """NodeInfo serializes and deserializes correctly."""
        identity = NodeIdentity.generate()
        info = identity.create_node_info(["tcp://localhost:9000"])
        
        data = info.to_dict()
        loaded = NodeInfo.from_dict(data)
        
        assert loaded.node_id == info.node_id
        assert loaded.public_key == info.public_key
        assert loaded.addresses == info.addresses
        assert loaded.timestamp == info.timestamp


class TestP2PSigning:
    """Tests for P2P message signing and verification."""
    
    def test_sign_and_verify_message(self):
        identity = NodeIdentity.generate()
        if not HAS_CRYPTO:
            pytest.skip("cryptography not available; public-key verification disabled")
        info = identity.create_node_info(["tcp://localhost:9000"])
        ping = Ping(sender_id=identity.node_id)
        identity.sign_message(ping)
        assert ping.signature is not None
        assert verify_message_signature(info, ping)
    
    def test_verify_message_signature_wrong_key_fails(self):
        if not HAS_CRYPTO:
            pytest.skip("cryptography not available; public-key verification disabled")
        sender = NodeIdentity.generate()
        receiver = NodeIdentity.generate()
        sender_info = sender.create_node_info(["tcp://sender:9000"])
        receiver_info = receiver.create_node_info(["tcp://receiver:9000"])
        ping = Ping(sender_id=sender.node_id)
        sender.sign_message(ping)
        assert not verify_message_signature(receiver_info, ping)


# =============================================================================
# Protocol Tests
# =============================================================================

class TestProtocol:
    """Tests for P2P protocol messages."""
    
    def test_contribution_announce(self):
        """ContributionAnnounce serializes correctly."""
        msg = ContributionAnnounce(
            sender_id="node123",
            goal_id="GOAL_1",
            contribution_hash="abc123",
            contributor_id="alice",
            score=0.95,
            log_index=42,
        )
        
        data = msg.to_dict()
        assert data["type"] == "contribution_announce"
        assert data["goal_id"] == "GOAL_1"
        assert data["score"] == 0.95
        
        # Round-trip
        loaded = Message.from_dict(data)
        assert isinstance(loaded, ContributionAnnounce)
        assert loaded.goal_id == "GOAL_1"
    
    def test_contribution_request(self):
        """ContributionRequest serializes correctly."""
        msg = ContributionRequest(
            sender_id="node123",
            contribution_hash="abc123",
        )
        
        data = msg.to_dict()
        loaded = Message.from_dict(data)
        
        assert isinstance(loaded, ContributionRequest)
        assert loaded.contribution_hash == "abc123"
    
    def test_state_request(self):
        """StateRequest serializes correctly."""
        msg = StateRequest(
            sender_id="node123",
            goal_id="GOAL_1",
            include_leaderboard=True,
            from_log_index=10,
        )
        
        data = msg.to_dict()
        loaded = Message.from_dict(data)
        
        assert isinstance(loaded, StateRequest)
        assert loaded.goal_id == "GOAL_1"
        assert loaded.from_log_index == 10
    
    def test_state_response(self):
        """StateResponse serializes correctly."""
        msg = StateResponse(
            sender_id="node123",
            goal_id="GOAL_1",
            log_root="abc123",
            log_size=100,
            leaderboard=[{"score": 0.9}],
        )
        
        data = msg.to_dict()
        loaded = Message.from_dict(data)
        
        assert isinstance(loaded, StateResponse)
        assert loaded.log_size == 100
    
    def test_peer_exchange(self):
        """PeerExchange serializes correctly."""
        msg = PeerExchange(
            sender_id="node123",
            peers=[{"node_id": "peer1", "addresses": ["tcp://..."]}],
        )
        
        data = msg.to_dict()
        loaded = Message.from_dict(data)
        
        assert isinstance(loaded, PeerExchange)
        assert len(loaded.peers) == 1
    
    def test_ping_pong(self):
        """Ping/Pong messages work."""
        ping = Ping(sender_id="node123")
        pong = Pong(sender_id="node456", ping_nonce=ping.nonce)
        
        assert ping.type == MessageType.PING
        assert pong.type == MessageType.PONG
        assert pong.ping_nonce == ping.nonce
    
    def test_wire_format(self):
        """Messages serialize to wire format."""
        msg = Ping(sender_id="node123")
        
        wire = msg.to_wire()
        assert len(wire) > 4  # At least length prefix
        
        # Parse back
        loaded = Message.from_wire(wire)
        assert isinstance(loaded, Ping)
        assert loaded.sender_id == "node123"
    
    def test_message_id_unique(self):
        """Message IDs are unique."""
        msg1 = Ping(sender_id="node123")
        msg2 = Ping(sender_id="node123")
        
        assert msg1.message_id() != msg2.message_id()


# =============================================================================
# Discovery Tests
# =============================================================================

class TestDiscovery:
    """Tests for peer discovery."""
    
    def test_seed_node_discovery_init(self):
        """Can initialize SeedNodeDiscovery."""
        identity = NodeIdentity.generate()
        
        discovery = SeedNodeDiscovery(
            identity=identity,
            seed_addresses=["tcp://seed1:9000", "tcp://seed2:9000"],
        )
        
        assert len(discovery.get_peers()) == 0
    
    def test_add_peer(self):
        """Can add a peer."""
        identity = NodeIdentity.generate()
        discovery = SeedNodeDiscovery(identity=identity, seed_addresses=[])
        
        peer_identity = NodeIdentity.generate()
        peer_info = peer_identity.create_node_info(["tcp://peer:9000"])
        
        discovery.add_peer(peer_info)
        
        peers = discovery.get_peers()
        assert len(peers) == 1
        assert peers[0].info.node_id == peer_identity.node_id
    
    def test_dont_add_self(self):
        """Don't add self as peer."""
        identity = NodeIdentity.generate()
        discovery = SeedNodeDiscovery(identity=identity, seed_addresses=[])
        
        self_info = identity.create_node_info(["tcp://self:9000"])
        discovery.add_peer(self_info)
        
        assert len(discovery.get_peers()) == 0
    
    def test_remove_peer(self):
        """Can remove a peer."""
        identity = NodeIdentity.generate()
        discovery = SeedNodeDiscovery(identity=identity, seed_addresses=[])
        
        peer_identity = NodeIdentity.generate()
        peer_info = peer_identity.create_node_info(["tcp://peer:9000"])
        
        discovery.add_peer(peer_info)
        discovery.remove_peer(peer_identity.node_id)
        
        assert len(discovery.get_peers()) == 0
    
    def test_max_peers_eviction(self):
        """Evicts peers when over limit."""
        identity = NodeIdentity.generate()
        discovery = SeedNodeDiscovery(
            identity=identity,
            seed_addresses=[],
            max_peers=3,
        )
        
        # Add 4 peers
        for i in range(4):
            peer = NodeIdentity.generate()
            info = peer.create_node_info([f"tcp://peer{i}:9000"])
            discovery.add_peer(info)
        
        # Should have max 3
        assert len(discovery.get_peers()) == 3
    
    def test_handle_peer_exchange(self):
        """Handle incoming peer exchange."""
        identity = NodeIdentity.generate()
        discovery = SeedNodeDiscovery(identity=identity, seed_addresses=[])
        
        # Create peer info and register peer as known
        peer = NodeIdentity.generate()
        peer_info = peer.create_node_info(["tcp://peer:9000"])
        discovery.add_peer(peer_info)
        
        # Create exchange message
        exchange = PeerExchange(
            sender_id=peer.node_id,
            peers=[peer_info.to_dict()],
        )
        if HAS_CRYPTO and peer.has_private_key():
            peer.sign_message(exchange)
        
        discovery.handle_peer_exchange(exchange)
        
        assert len(discovery.get_peers()) == 1


# =============================================================================
# API Tests
# =============================================================================

class TestApiHandlers:
    """Tests for API handlers."""
    
    @pytest.fixture
    def coordinator(self):
        from idi.ian import (
            IANCoordinator,
            CoordinatorConfig,
            GoalSpec,
            GoalID,
            EvaluationLimits,
            Thresholds,
        )
        
        goal_spec = GoalSpec(
            goal_id=GoalID("API_TEST"),
            name="API Test Goal",
            description="Testing API",
            eval_limits=EvaluationLimits(),
            thresholds=Thresholds(),
        )
        
        return IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=10),
        )
    
    @pytest.fixture
    def handlers(self, coordinator):
        config = ApiConfig(rate_limit_per_ip=1000)
        return IANApiHandlers(coordinator, config)
    
    def test_health(self, handlers):
        """Health endpoint works."""
        resp = handlers.handle_health()
        
        assert resp.success
        assert resp.data["status"] == "healthy"
    
    def test_status(self, handlers):
        """Status endpoint works."""
        resp = handlers.handle_status("API_TEST", "127.0.0.1")
        
        assert resp.success
        assert resp.data["goal_id"] == "API_TEST"
        assert "log_size" in resp.data
    
    def test_leaderboard(self, handlers):
        """Leaderboard endpoint works."""
        resp = handlers.handle_leaderboard("API_TEST", limit=10, ip="127.0.0.1")
        
        assert resp.success
        assert resp.data["goal_id"] == "API_TEST"
        assert "entries" in resp.data
    
    def test_policy(self, handlers):
        """Policy endpoint works."""
        resp = handlers.handle_policy("API_TEST", "127.0.0.1")
        
        assert resp.success
        assert resp.data["goal_id"] == "API_TEST"
    
    def test_contribute(self, handlers):
        """Contribute endpoint works."""
        body = {
            "goal_id": "API_TEST",
            "agent_pack": {
                "version": "1.0",
                "parameters": base64.b64encode(b"test_params").decode(),
            },
            "contributor_id": "test_user",
            "seed": 12345,
        }
        
        resp = handlers.handle_contribute(body, "127.0.0.1")
        
        assert resp.success
        assert "accepted" in resp.data
    
    def test_rate_limiting(self, handlers):
        """Rate limiting works."""
        # Create handler with low limit
        from idi.ian import IANCoordinator, CoordinatorConfig, GoalSpec, GoalID, EvaluationLimits, Thresholds
        
        goal_spec = GoalSpec(
            goal_id=GoalID("RATE_TEST"),
            name="Rate Test",
            description="Test",
            eval_limits=EvaluationLimits(),
            thresholds=Thresholds(),
        )
        
        coord = IANCoordinator(goal_spec=goal_spec, config=CoordinatorConfig())
        config = ApiConfig(rate_limit_per_ip=2)
        handlers = IANApiHandlers(coord, config)
        
        # First two should succeed
        handlers.handle_status("RATE_TEST", "10.0.0.1")
        handlers.handle_status("RATE_TEST", "10.0.0.1")
        
        # Third should be rate limited
        resp = handlers.handle_status("RATE_TEST", "10.0.0.1")
        
        assert not resp.success
        assert "Rate limited" in resp.error
    
    def test_api_key_required(self, coordinator):
        """API key validation works."""
        import asyncio
        config = ApiConfig(api_key="secret123")
        handlers = IANApiHandlers(coordinator, config)
        
        body = {
            "goal_id": "API_TEST",
            "agent_pack": {"version": "1.0", "parameters": base64.b64encode(b"test").decode()},
            "contributor_id": "test",
            "seed": 1,
        }
        
        # Without key - should fail
        resp = handlers.handle_contribute(body, "127.0.0.1", api_key=None)
        assert not resp.success
        assert "API key" in resp.error
        
        # With wrong key - should fail
        resp = handlers.handle_contribute(body, "127.0.0.1", api_key="wrong")
        assert not resp.success
        
        # With correct key - should succeed
        resp = handlers.handle_contribute(body, "127.0.0.1", api_key="secret123")
        assert resp.success
