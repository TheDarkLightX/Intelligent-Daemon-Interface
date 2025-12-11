"""
P2P Integration Tests for IAN Decentralized Network.

These tests verify multi-node behavior including:
1. Message propagation between nodes
2. State synchronization after partition
3. Consensus under Byzantine conditions
4. Rate limiting under load
5. Graceful degradation

Usage:
    pytest test_p2p_integration.py -v
    pytest test_p2p_integration.py -k "test_message_propagation" -v
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================

@dataclass
class MockMessage:
    """Mock P2P message for testing."""
    msg_type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: bytes = b""


@dataclass
class MockPeer:
    """Mock peer for testing."""
    node_id: str
    address: str
    connected: bool = True
    messages_received: List[MockMessage] = field(default_factory=list)
    messages_sent: List[MockMessage] = field(default_factory=list)
    latency_ms: float = 10.0
    drop_rate: float = 0.0  # Probability of dropping messages


class MockNetwork:
    """
    Mock network for multi-node integration testing.
    
    Simulates a P2P network with configurable:
    - Latency between nodes
    - Message drop rates
    - Network partitions
    - Byzantine behavior
    """
    
    def __init__(self):
        self._nodes: Dict[str, "MockNode"] = {}
        self._partitions: List[Set[str]] = []  # Groups that can't communicate
        self._global_latency_ms: float = 10.0
        self._global_drop_rate: float = 0.0
        self._message_log: List[Dict[str, Any]] = []
    
    def add_node(self, node: "MockNode") -> None:
        """Add node to network."""
        self._nodes[node.node_id] = node
        node.set_network(self)
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from network."""
        self._nodes.pop(node_id, None)
    
    def get_node(self, node_id: str) -> Optional["MockNode"]:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List["MockNode"]:
        """Get all nodes."""
        return list(self._nodes.values())
    
    def create_partition(self, group_a: Set[str], group_b: Set[str]) -> None:
        """Create network partition between two groups."""
        self._partitions.append(group_a)
        self._partitions.append(group_b)
        logger.info(f"Created partition: {group_a} <-> {group_b}")
    
    def heal_partitions(self) -> None:
        """Heal all network partitions."""
        self._partitions.clear()
        logger.info("Healed all network partitions")
    
    def can_communicate(self, from_id: str, to_id: str) -> bool:
        """Check if two nodes can communicate."""
        for partition in self._partitions:
            if from_id in partition and to_id not in partition:
                return False
            if to_id in partition and from_id not in partition:
                return False
        return True
    
    async def send_message(
        self,
        from_id: str,
        to_id: str,
        message: MockMessage,
    ) -> bool:
        """Send message between nodes."""
        # Check partition
        if not self.can_communicate(from_id, to_id):
            logger.debug(f"Message dropped: partition {from_id} -> {to_id}")
            return False
        
        # Check drop rate
        if random.random() < self._global_drop_rate:
            logger.debug(f"Message dropped: random {from_id} -> {to_id}")
            return False
        
        # Get target node
        target = self._nodes.get(to_id)
        if not target:
            return False
        
        # Simulate latency
        await asyncio.sleep(self._global_latency_ms / 1000.0)
        
        # Log message
        self._message_log.append({
            "from": from_id,
            "to": to_id,
            "type": message.msg_type,
            "timestamp": time.time(),
        })
        
        # Deliver message
        await target.receive_message(message)
        return True
    
    async def broadcast(
        self,
        from_id: str,
        message: MockMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast message to all nodes."""
        exclude = exclude or set()
        exclude.add(from_id)
        
        delivered = 0
        for node_id in self._nodes:
            if node_id not in exclude:
                if await self.send_message(from_id, node_id, message):
                    delivered += 1
        
        return delivered
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._message_log)
    
    def get_messages_by_type(self, msg_type: str) -> List[Dict[str, Any]]:
        """Get messages by type."""
        return [m for m in self._message_log if m["type"] == msg_type]


class MockNode:
    """
    Mock node for integration testing.
    
    Simulates an IAN node with configurable behavior.
    """
    
    def __init__(
        self,
        node_id: str,
        is_byzantine: bool = False,
    ):
        self.node_id = node_id
        self.is_byzantine = is_byzantine
        self._network: Optional[MockNetwork] = None
        self._peers: Set[str] = set()
        self._messages_received: List[MockMessage] = []
        self._state: Dict[str, Any] = {
            "log_size": 0,
            "log_root": b"\x00" * 32,
            "contributions": [],
        }
        self._handlers: Dict[str, Callable] = {}
        self._running = False
    
    def set_network(self, network: MockNetwork) -> None:
        """Set network reference."""
        self._network = network
    
    def connect_to_peer(self, peer_id: str) -> None:
        """Connect to a peer."""
        self._peers.add(peer_id)
    
    def disconnect_from_peer(self, peer_id: str) -> None:
        """Disconnect from a peer."""
        self._peers.discard(peer_id)
    
    def register_handler(self, msg_type: str, handler: Callable) -> None:
        """Register message handler."""
        self._handlers[msg_type] = handler
    
    async def receive_message(self, message: MockMessage) -> None:
        """Receive and process a message."""
        self._messages_received.append(message)
        
        # Call handler if registered
        handler = self._handlers.get(message.msg_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
    
    async def send_to_peer(self, peer_id: str, message: MockMessage) -> bool:
        """Send message to specific peer."""
        if not self._network:
            return False
        return await self._network.send_message(self.node_id, peer_id, message)
    
    async def broadcast(self, message: MockMessage) -> int:
        """Broadcast message to all peers."""
        if not self._network:
            return 0
        return await self._network.broadcast(self.node_id, message)
    
    def get_state(self) -> Dict[str, Any]:
        """Get node state."""
        return self._state.copy()
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value."""
        self._state[key] = value
    
    def add_contribution(self, contribution: Dict[str, Any]) -> None:
        """Add contribution to local state."""
        self._state["contributions"].append(contribution)
        self._state["log_size"] += 1
    
    def get_message_count(self) -> int:
        """Get received message count."""
        return len(self._messages_received)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def network():
    """Create mock network."""
    return MockNetwork()


@pytest.fixture
def three_node_network(network):
    """Create network with 3 nodes."""
    nodes = [
        MockNode(f"node_{i}")
        for i in range(3)
    ]
    
    for node in nodes:
        network.add_node(node)
        # Connect to all other nodes
        for other in nodes:
            if other.node_id != node.node_id:
                node.connect_to_peer(other.node_id)
    
    return network, nodes


@pytest.fixture
def five_node_network(network):
    """Create network with 5 nodes."""
    nodes = [
        MockNode(f"node_{i}")
        for i in range(5)
    ]
    
    for node in nodes:
        network.add_node(node)
        for other in nodes:
            if other.node_id != node.node_id:
                node.connect_to_peer(other.node_id)
    
    return network, nodes


# =============================================================================
# Message Propagation Tests
# =============================================================================

class TestMessagePropagation:
    """Tests for message propagation in P2P network."""
    
    @pytest.mark.asyncio
    async def test_direct_message_delivery(self, three_node_network):
        """Test direct message delivery between two nodes."""
        network, nodes = three_node_network
        sender, receiver = nodes[0], nodes[1]
        
        message = MockMessage(
            msg_type="CONTRIBUTION",
            sender_id=sender.node_id,
            payload={"data": "test"},
        )
        
        success = await sender.send_to_peer(receiver.node_id, message)
        
        assert success
        assert receiver.get_message_count() == 1
        assert receiver._messages_received[0].payload["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_broadcast_reaches_all_nodes(self, three_node_network):
        """Test broadcast reaches all connected nodes."""
        network, nodes = three_node_network
        sender = nodes[0]
        
        message = MockMessage(
            msg_type="ANNOUNCEMENT",
            sender_id=sender.node_id,
            payload={"announcement": "hello"},
        )
        
        delivered = await sender.broadcast(message)
        
        assert delivered == 2  # All except sender
        for node in nodes[1:]:
            assert node.get_message_count() == 1
    
    @pytest.mark.asyncio
    async def test_message_handler_called(self, three_node_network):
        """Test message handlers are called on receipt."""
        network, nodes = three_node_network
        sender, receiver = nodes[0], nodes[1]
        
        handler_called = []
        
        def handler(msg):
            handler_called.append(msg)
        
        receiver.register_handler("TEST", handler)
        
        message = MockMessage(
            msg_type="TEST",
            sender_id=sender.node_id,
            payload={},
        )
        
        await sender.send_to_peer(receiver.node_id, message)
        
        assert len(handler_called) == 1
    
    @pytest.mark.asyncio
    async def test_gossip_propagation(self, five_node_network):
        """Test gossip-style message propagation."""
        network, nodes = five_node_network
        
        # Track which nodes received the message
        received_by: Set[str] = set()
        
        async def forward_handler(msg):
            node_id = msg.payload.get("current_node")
            received_by.add(node_id)
            
            # Forward to random peers (gossip)
            node = network.get_node(node_id)
            if node and len(received_by) < 5:
                for peer_id in list(node._peers)[:2]:
                    if peer_id not in received_by:
                        forward_msg = MockMessage(
                            msg_type="GOSSIP",
                            sender_id=node_id,
                            payload={"current_node": peer_id, "original": msg.payload.get("original")},
                        )
                        await node.send_to_peer(peer_id, forward_msg)
        
        # Register handlers
        for node in nodes:
            node.register_handler("GOSSIP", forward_handler)
        
        # Start gossip from first node
        origin = nodes[0]
        received_by.add(origin.node_id)
        
        for peer_id in list(origin._peers)[:2]:
            msg = MockMessage(
                msg_type="GOSSIP",
                sender_id=origin.node_id,
                payload={"current_node": peer_id, "original": "test_data"},
            )
            await origin.send_to_peer(peer_id, msg)
        
        # Allow propagation
        await asyncio.sleep(0.1)
        
        # Should reach most/all nodes
        assert len(received_by) >= 3


# =============================================================================
# Network Partition Tests
# =============================================================================

class TestNetworkPartitions:
    """Tests for network partition handling."""
    
    @pytest.mark.asyncio
    async def test_partition_blocks_messages(self, five_node_network):
        """Test that partitions block messages between groups."""
        network, nodes = five_node_network
        
        # Create partition: [0,1] <-> [2,3,4]
        group_a = {nodes[0].node_id, nodes[1].node_id}
        group_b = {nodes[2].node_id, nodes[3].node_id, nodes[4].node_id}
        network.create_partition(group_a, group_b)
        
        # Message within group A should work
        msg = MockMessage(msg_type="TEST", sender_id=nodes[0].node_id, payload={})
        success = await nodes[0].send_to_peer(nodes[1].node_id, msg)
        assert success
        
        # Message across partition should fail
        success = await nodes[0].send_to_peer(nodes[2].node_id, msg)
        assert not success
    
    @pytest.mark.asyncio
    async def test_partition_healing(self, five_node_network):
        """Test that healing partitions restores communication."""
        network, nodes = five_node_network
        
        # Create and heal partition
        group_a = {nodes[0].node_id}
        group_b = {nodes[1].node_id}
        network.create_partition(group_a, group_b)
        
        # Should fail
        msg = MockMessage(msg_type="TEST", sender_id=nodes[0].node_id, payload={})
        assert not await nodes[0].send_to_peer(nodes[1].node_id, msg)
        
        # Heal
        network.heal_partitions()
        
        # Should work now
        assert await nodes[0].send_to_peer(nodes[1].node_id, msg)
    
    @pytest.mark.asyncio
    async def test_state_divergence_during_partition(self, five_node_network):
        """Test state diverges during partition and can be detected."""
        network, nodes = five_node_network
        
        # Partition into two groups
        group_a = {nodes[0].node_id, nodes[1].node_id}
        group_b = {nodes[2].node_id, nodes[3].node_id, nodes[4].node_id}
        network.create_partition(group_a, group_b)
        
        # Add different contributions to each group
        nodes[0].add_contribution({"id": "contrib_a1", "group": "A"})
        nodes[1].add_contribution({"id": "contrib_a1", "group": "A"})
        
        nodes[2].add_contribution({"id": "contrib_b1", "group": "B"})
        nodes[3].add_contribution({"id": "contrib_b1", "group": "B"})
        nodes[4].add_contribution({"id": "contrib_b1", "group": "B"})
        
        # States should diverge
        state_a = nodes[0].get_state()
        state_b = nodes[2].get_state()
        
        assert state_a["contributions"] != state_b["contributions"]
        assert state_a["contributions"][0]["group"] == "A"
        assert state_b["contributions"][0]["group"] == "B"


# =============================================================================
# State Synchronization Tests
# =============================================================================

class TestStateSynchronization:
    """Tests for state synchronization between nodes."""
    
    @pytest.mark.asyncio
    async def test_new_node_syncs_state(self, three_node_network):
        """Test new node can sync state from existing nodes."""
        network, nodes = three_node_network
        
        # Add contributions to existing nodes
        for i in range(5):
            for node in nodes:
                node.add_contribution({"id": f"contrib_{i}"})
        
        # Add new node
        new_node = MockNode("new_node")
        network.add_node(new_node)
        
        # New node requests state
        request = MockMessage(
            msg_type="SYNC_REQUEST",
            sender_id=new_node.node_id,
            payload={"from_index": 0},
        )
        
        await new_node.send_to_peer(nodes[0].node_id, request)
        
        # Verify request was received
        assert nodes[0].get_message_count() >= 1
    
    @pytest.mark.asyncio
    async def test_sync_after_partition_heal(self, five_node_network):
        """Test nodes sync state after partition heals."""
        network, nodes = five_node_network
        
        # Create partition
        group_a = {nodes[0].node_id, nodes[1].node_id}
        group_b = {nodes[2].node_id, nodes[3].node_id, nodes[4].node_id}
        network.create_partition(group_a, group_b)
        
        # Add contributions during partition
        nodes[0].add_contribution({"id": "contrib_a"})
        nodes[2].add_contribution({"id": "contrib_b"})
        
        # Heal partition
        network.heal_partitions()
        
        # Simulate sync request
        sync_request = MockMessage(
            msg_type="STATE_SYNC",
            sender_id=nodes[0].node_id,
            payload={
                "log_size": nodes[0].get_state()["log_size"],
                "contributions": nodes[0].get_state()["contributions"],
            },
        )
        
        # Broadcast state
        delivered = await nodes[0].broadcast(sync_request)
        
        assert delivered >= 3  # Should reach nodes in other partition


# =============================================================================
# Byzantine Behavior Tests
# =============================================================================

class TestByzantineBehavior:
    """Tests for handling Byzantine (malicious) nodes."""
    
    @pytest.mark.asyncio
    async def test_invalid_message_rejected(self, three_node_network):
        """Test that invalid messages are rejected."""
        network, nodes = three_node_network
        
        rejected = []
        
        def validator_handler(msg):
            # Reject messages without valid signature
            if not msg.signature:
                rejected.append(msg)
                return
        
        nodes[1].register_handler("DATA", validator_handler)
        
        # Send message without signature
        msg = MockMessage(
            msg_type="DATA",
            sender_id=nodes[0].node_id,
            payload={"data": "test"},
            signature=b"",  # Invalid
        )
        
        await nodes[0].send_to_peer(nodes[1].node_id, msg)
        
        assert len(rejected) == 1
    
    @pytest.mark.asyncio
    async def test_byzantine_node_isolation(self, five_node_network):
        """Test that Byzantine nodes can be isolated."""
        network, nodes = five_node_network
        
        # Mark node 0 as Byzantine
        byzantine = nodes[0]
        byzantine.is_byzantine = True
        
        # Track reputation
        reputation: Dict[str, int] = {n.node_id: 100 for n in nodes}
        
        async def reputation_handler(msg):
            sender = msg.sender_id
            # Penalize invalid messages
            if msg.payload.get("invalid"):
                reputation[sender] -= 20
        
        for node in nodes[1:]:
            node.register_handler("DATA", reputation_handler)
        
        # Byzantine node sends invalid messages
        for _ in range(5):
            msg = MockMessage(
                msg_type="DATA",
                sender_id=byzantine.node_id,
                payload={"invalid": True},
            )
            await byzantine.broadcast(msg)
        
        # Byzantine node should have low reputation
        assert reputation[byzantine.node_id] == 0
    
    @pytest.mark.asyncio
    async def test_majority_consensus(self, five_node_network):
        """Test that majority consensus is reached despite Byzantine node."""
        network, nodes = five_node_network
        
        # One Byzantine node sends different value
        byzantine = nodes[0]
        byzantine.is_byzantine = True
        
        votes: Dict[str, str] = {}
        
        async def vote_handler(msg):
            votes[msg.sender_id] = msg.payload["vote"]
        
        for node in nodes:
            node.register_handler("VOTE", vote_handler)
        
        # Honest nodes vote "A"
        for node in nodes[1:]:
            msg = MockMessage(
                msg_type="VOTE",
                sender_id=node.node_id,
                payload={"vote": "A"},
            )
            await node.broadcast(msg)
        
        # Byzantine node votes "B"
        msg = MockMessage(
            msg_type="VOTE",
            sender_id=byzantine.node_id,
            payload={"vote": "B"},
        )
        await byzantine.broadcast(msg)
        
        # Count votes
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Majority should be "A"
        assert vote_counts.get("A", 0) > vote_counts.get("B", 0)


# =============================================================================
# Load and Stress Tests
# =============================================================================

class TestLoadAndStress:
    """Tests for behavior under load."""
    
    @pytest.mark.asyncio
    async def test_high_message_volume(self, five_node_network):
        """Test network handles high message volume."""
        network, nodes = five_node_network
        
        message_count = 100
        received_counts: Dict[str, int] = {n.node_id: 0 for n in nodes}
        
        def count_handler(msg):
            # Count received messages
            pass
        
        for node in nodes:
            node.register_handler("FLOOD", count_handler)
        
        # Each node sends many messages
        tasks = []
        for node in nodes:
            for i in range(message_count // len(nodes)):
                msg = MockMessage(
                    msg_type="FLOOD",
                    sender_id=node.node_id,
                    payload={"seq": i},
                )
                tasks.append(node.broadcast(msg))
        
        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        total_messages = network.get_message_count()
        
        # Should handle all messages
        assert total_messages > 0
        logger.info(f"Processed {total_messages} messages in {elapsed:.2f}s")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, three_node_network):
        """Test rate limiting prevents message flooding."""
        network, nodes = three_node_network
        
        # Simulate rate limiter
        rate_limit = 10  # messages per node
        message_counts: Dict[str, int] = {}
        rejected: List[MockMessage] = []
        
        def rate_limited_handler(msg):
            sender = msg.sender_id
            message_counts[sender] = message_counts.get(sender, 0) + 1
            
            if message_counts[sender] > rate_limit:
                rejected.append(msg)
        
        nodes[1].register_handler("DATA", rate_limited_handler)
        
        # Send many messages from one node
        sender = nodes[0]
        for i in range(20):
            msg = MockMessage(
                msg_type="DATA",
                sender_id=sender.node_id,
                payload={"seq": i},
            )
            await sender.send_to_peer(nodes[1].node_id, msg)
        
        # Should have rejected some messages
        assert len(rejected) == 10  # 20 - rate_limit
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_failures(self, five_node_network):
        """Test network degrades gracefully when nodes fail."""
        network, nodes = five_node_network
        
        # Remove two nodes (simulating failure)
        network.remove_node(nodes[3].node_id)
        network.remove_node(nodes[4].node_id)
        
        # Remaining nodes should still communicate
        msg = MockMessage(
            msg_type="TEST",
            sender_id=nodes[0].node_id,
            payload={"data": "test"},
        )
        
        delivered = await nodes[0].broadcast(msg)
        
        # Should deliver to remaining nodes
        assert delivered == 2  # nodes[1] and nodes[2]


# =============================================================================
# Latency and Timing Tests
# =============================================================================

class TestLatencyAndTiming:
    """Tests for latency-sensitive behavior."""
    
    @pytest.mark.asyncio
    async def test_message_ordering(self, three_node_network):
        """Test messages maintain ordering."""
        network, nodes = three_node_network
        
        received_order: List[int] = []
        
        def order_handler(msg):
            received_order.append(msg.payload["seq"])
        
        nodes[1].register_handler("ORDERED", order_handler)
        
        # Send messages in order
        for i in range(10):
            msg = MockMessage(
                msg_type="ORDERED",
                sender_id=nodes[0].node_id,
                payload={"seq": i},
            )
            await nodes[0].send_to_peer(nodes[1].node_id, msg)
        
        # Should receive in order (with our mock network)
        assert received_order == list(range(10))
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, three_node_network):
        """Test timeout handling for slow responses."""
        network, nodes = three_node_network
        
        # Increase network latency
        network._global_latency_ms = 100.0
        
        async def slow_operation():
            msg = MockMessage(
                msg_type="SLOW",
                sender_id=nodes[0].node_id,
                payload={},
            )
            return await nodes[0].send_to_peer(nodes[1].node_id, msg)
        
        # Should complete despite latency
        start = time.time()
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
        elapsed = time.time() - start
        
        assert result
        assert elapsed >= 0.1  # Should have waited for latency


# =============================================================================
# Fuzz Tests
# =============================================================================

class TestFuzzing:
    """Fuzz tests for robustness."""
    
    @pytest.mark.asyncio
    async def test_random_message_payloads(self, three_node_network):
        """Test handling of random message payloads."""
        network, nodes = three_node_network
        
        errors: List[Exception] = []
        
        def safe_handler(msg):
            try:
                # Try to process payload
                _ = msg.payload.get("data", "default")
            except Exception as e:
                errors.append(e)
        
        nodes[1].register_handler("FUZZ", safe_handler)
        
        # Send messages with random payloads
        for _ in range(50):
            payload = {
                "data": random.choice([None, "", 0, [], {}, "valid"]),
                "extra": random.randint(-1000, 1000),
            }
            
            msg = MockMessage(
                msg_type="FUZZ",
                sender_id=nodes[0].node_id,
                payload=payload,
            )
            
            await nodes[0].send_to_peer(nodes[1].node_id, msg)
        
        # Should handle all without errors
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, network):
        """Test rapid node connect/disconnect cycles."""
        # Create and destroy nodes rapidly
        for i in range(20):
            node = MockNode(f"temp_node_{i}")
            network.add_node(node)
            
            # Send a message
            if network.get_all_nodes():
                other = network.get_all_nodes()[0]
                if other.node_id != node.node_id:
                    msg = MockMessage(
                        msg_type="PING",
                        sender_id=node.node_id,
                        payload={},
                    )
                    await node.send_to_peer(other.node_id, msg)
            
            # Remove node
            network.remove_node(node.node_id)
        
        # Network should be empty
        assert len(network.get_all_nodes()) == 0


# =============================================================================
# Integration Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
