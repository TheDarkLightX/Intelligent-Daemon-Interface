"""
Tests for IAN decentralized L2 components.

Tests:
1. Contribution ordering and mempool
2. Consensus coordinator
3. Fraud proof generation and verification
4. Economic security (bonding, slashing)
5. Evaluation quorum
"""

import asyncio
import hashlib
import json
import pytest
import time
from dataclasses import asdict

from idi.ian.models import (
    GoalID, GoalSpec, AgentPack, Contribution, Metrics,
    EvaluationLimits, Thresholds,
)
from idi.ian.coordinator import IANCoordinator


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def goal_spec():
    """Create test goal spec."""
    return GoalSpec(
        goal_id=GoalID("TEST_GOAL_001"),
        name="Test Goal",
        description="A test goal",
        eval_limits=EvaluationLimits(
            max_episodes=10,
            max_steps_per_episode=100,
            timeout_seconds=60,
            max_memory_mb=256,
        ),
        thresholds=Thresholds(
            min_reward=0.0,
            max_risk=1.0,
            max_complexity=1.0,
        ),
    )


@pytest.fixture
def coordinator(goal_spec):
    """Create test coordinator."""
    return IANCoordinator(goal_spec)


def make_contribution(goal_id, contributor_id="test_contributor", seed=None):
    """Create test contribution."""
    if seed is None:
        seed = int(time.time() * 1000)
    
    return Contribution(
        goal_id=goal_id,
        agent_pack=AgentPack(
            version="1.0",
            parameters=f"params_{seed}".encode(),
        ),
        contributor_id=contributor_id,
        seed=seed,
    )


# =============================================================================
# Ordering Tests
# =============================================================================

class TestOrdering:
    """Test contribution ordering."""
    
    def test_ordering_key_comparison(self):
        """Test ordering key comparison."""
        from idi.ian.network.ordering import OrderingKey
        
        # Earlier timestamp wins
        key1 = OrderingKey(timestamp_ms=1000, pack_hash=b'\x00' * 32)
        key2 = OrderingKey(timestamp_ms=2000, pack_hash=b'\x00' * 32)
        
        assert key1 < key2
        assert not key2 < key1
    
    def test_ordering_key_tiebreak(self):
        """Test ordering key tiebreak by hash."""
        from idi.ian.network.ordering import OrderingKey
        
        # Same timestamp, hash tiebreak
        key1 = OrderingKey(timestamp_ms=1000, pack_hash=b'\x00' * 32)
        key2 = OrderingKey(timestamp_ms=1000, pack_hash=b'\xff' * 32)
        
        assert key1 < key2  # \x00 < \xff
    
    @pytest.mark.asyncio
    async def test_mempool_ordering(self, goal_spec):
        """Test mempool maintains order."""
        from idi.ian.network.ordering import ContributionMempool
        
        mempool = ContributionMempool(goal_id=str(goal_spec.goal_id))
        
        # Add contributions with different timestamps
        contrib3 = make_contribution(goal_spec.goal_id, seed=3000)
        contrib1 = make_contribution(goal_spec.goal_id, seed=1000)
        contrib2 = make_contribution(goal_spec.goal_id, seed=2000)
        
        # Add in wrong order
        await mempool.add(contrib3)
        await mempool.add(contrib1)
        await mempool.add(contrib2)
        
        # Pop should return in correct order
        entry1 = await mempool.pop_next()
        entry2 = await mempool.pop_next()
        entry3 = await mempool.pop_next()
        
        assert entry1.key.timestamp_ms == 1000
        assert entry2.key.timestamp_ms == 2000
        assert entry3.key.timestamp_ms == 3000
    
    @pytest.mark.asyncio
    async def test_mempool_dedup(self, goal_spec):
        """Test mempool rejects duplicates."""
        from idi.ian.network.ordering import ContributionMempool
        
        mempool = ContributionMempool(goal_id=str(goal_spec.goal_id))
        
        contrib = make_contribution(goal_spec.goal_id, seed=1000)
        
        success1, _ = await mempool.add(contrib)
        success2, reason = await mempool.add(contrib)
        
        assert success1
        assert not success2
        assert "duplicate" in reason
    
    @pytest.mark.asyncio
    async def test_mempool_bounded_size(self, goal_spec):
        """Test mempool respects max size."""
        from idi.ian.network.ordering import ContributionMempool
        
        max_size = 5
        mempool = ContributionMempool(
            goal_id=str(goal_spec.goal_id),
            max_size=max_size,
        )
        
        # Add more than max size
        for i in range(max_size + 3):
            contrib = make_contribution(goal_spec.goal_id, seed=i * 1000)
            await mempool.add(contrib)
        
        assert mempool.size <= max_size
    
    def test_ordering_proof_validity(self):
        """Test ordering proof verification."""
        from idi.ian.network.ordering import OrderingProof, OrderingProofEntry, OrderingKey
        
        # Create valid ordering proof
        entries = [
            OrderingProofEntry(
                key=OrderingKey(timestamp_ms=i * 1000, pack_hash=bytes([i]) * 32),
                contribution_hash=bytes([i]) * 32,
            )
            for i in range(5)
        ]
        
        proof = OrderingProof(
            goal_id="TEST_GOAL",
            timestamp_ms=int(time.time() * 1000),
            entries=entries,
            mempool_size=5,
        )
        
        valid, reason = proof.verify_ordering()
        assert valid, reason
    
    def test_ordering_proof_invalid(self):
        """Test detection of invalid ordering."""
        from idi.ian.network.ordering import OrderingProof, OrderingProofEntry, OrderingKey
        
        # Create out-of-order proof
        entries = [
            OrderingProofEntry(
                key=OrderingKey(timestamp_ms=2000, pack_hash=b'\x00' * 32),
                contribution_hash=b'\x00' * 32,
            ),
            OrderingProofEntry(
                key=OrderingKey(timestamp_ms=1000, pack_hash=b'\x01' * 32),  # Earlier!
                contribution_hash=b'\x01' * 32,
            ),
        ]
        
        proof = OrderingProof(
            goal_id="TEST_GOAL",
            timestamp_ms=int(time.time() * 1000),
            entries=entries,
            mempool_size=2,
        )
        
        valid, reason = proof.verify_ordering()
        assert not valid
        assert "out of order" in reason
    
    def test_ordering_key_deterministic_when_seed_zero(self, goal_spec):
        """Regression: OrderingKey must be deterministic when seed==0.
        
        Previously, seed==0 caused a wall-clock fallback which broke
        multi-node consensus (different nodes computed different keys).
        Now seed==0 falls back to _U64_MAX, ensuring all nodes agree.
        """
        from idi.ian.network.ordering import OrderingKey, _U64_MAX
        
        # Create contribution with seed=0 (no explicit timestamp)
        contrib = make_contribution(goal_spec.goal_id, seed=0)
        
        # Generate key twice - must be identical (deterministic)
        key1 = OrderingKey.from_contribution(contrib)
        key2 = OrderingKey.from_contribution(contrib)
        
        assert key1 == key2, "OrderingKey must be deterministic"
        assert key1.timestamp_ms == _U64_MAX, "seed==0 must fall back to _U64_MAX"
    
    @pytest.mark.asyncio
    async def test_mempool_ordering_with_seed_zero(self, goal_spec):
        """Regression: Contributions with seed=0 should sort after explicit seeds.
        
        This ensures that contributions without explicit ordering hints
        are processed last, giving priority to timestamped contributions.
        """
        from idi.ian.network.ordering import ContributionMempool, _U64_MAX
        
        mempool = ContributionMempool(goal_id=str(goal_spec.goal_id))
        
        # Add contribution with seed=0 (deterministic fallback)
        contrib_no_seed = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=AgentPack(version="1.0", parameters=b"no_seed_params"),
            contributor_id="alice",
            seed=0,
        )
        
        # Add contribution with explicit seed
        contrib_with_seed = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=AgentPack(version="1.0", parameters=b"with_seed_params"),
            contributor_id="bob",
            seed=1000,
        )
        
        # Add in reverse order
        await mempool.add(contrib_no_seed)
        await mempool.add(contrib_with_seed)
        
        # Pop should return explicit-seed first (lower timestamp wins)
        entry1 = await mempool.pop_next()
        entry2 = await mempool.pop_next()
        
        assert entry1.key.timestamp_ms == 1000, "Explicit seed should come first"
        assert entry2.key.timestamp_ms == _U64_MAX, "seed=0 should fallback to _U64_MAX"


# =============================================================================
# Consensus Tests
# =============================================================================

class TestConsensus:
    """Test consensus coordinator."""
    
    @pytest.mark.asyncio
    async def test_consensus_coordinator_init(self, coordinator):
        """Test consensus coordinator initialization."""
        from idi.ian.network.consensus import ConsensusCoordinator, ConsensusState
        
        consensus = ConsensusCoordinator(
            coordinator=coordinator,
            node_id="test_node_001",
        )
        
        assert consensus.consensus_state == ConsensusState.ISOLATED
        assert consensus.mempool_size == 0
    
    @pytest.mark.asyncio
    async def test_submit_contribution(self, goal_spec, coordinator):
        """Test contribution submission."""
        from idi.ian.network.consensus import ConsensusCoordinator
        
        consensus = ConsensusCoordinator(
            coordinator=coordinator,
            node_id="test_node_001",
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        success, reason = await consensus.submit_contribution(contrib)
        
        assert success, reason
        assert consensus.mempool_size == 1
    
    def test_peer_state_matching(self):
        """Test peer state comparison."""
        from idi.ian.network.consensus import PeerStateSnapshot
        
        state1 = PeerStateSnapshot(
            node_id="node1",
            goal_id="GOAL",
            log_root=b'\x00' * 32,
            log_size=100,
            leaderboard_root=b'\x01' * 32,
            active_policy_hash=None,
            timestamp_ms=int(time.time() * 1000),
        )
        
        state2 = PeerStateSnapshot(
            node_id="node2",
            goal_id="GOAL",
            log_root=b'\x00' * 32,
            log_size=100,
            leaderboard_root=b'\x01' * 32,
            active_policy_hash=None,
            timestamp_ms=int(time.time() * 1000),
        )
        
        assert state1.matches(state2)
        
        # Different root
        state3 = PeerStateSnapshot(
            node_id="node3",
            goal_id="GOAL",
            log_root=b'\xff' * 32,  # Different!
            log_size=100,
            leaderboard_root=b'\x01' * 32,
            active_policy_hash=None,
            timestamp_ms=int(time.time() * 1000),
        )
        
        assert not state1.matches(state3)


# =============================================================================
# Fraud Proof Tests
# =============================================================================

class TestFraudProofs:
    """Test fraud proof generation and verification."""
    
    def test_invalid_log_root_proof(self, coordinator):
        """Test invalid log root fraud proof."""
        from idi.ian.network.fraud import InvalidLogRootProof, FraudProofVerifier
        
        # Create proof with different roots and no Merkle proofs needed for basic check
        proof = InvalidLogRootProof(
            goal_id=str(coordinator.goal_spec.goal_id),
            challenged_commit_hash=b'\xab' * 32,
            claimed_root=b'\x00' * 32,
            actual_leaves=[],  # Empty - just testing root mismatch
            leaf_indices=[],
            merkle_proofs=[],
            computed_root=b'\xff' * 32,  # Different from claimed
        )
        
        verifier = FraudProofVerifier()
        valid, reason = verifier.verify(proof)
        
        # Should be valid fraud proof (roots differ)
        assert valid, reason
    
    def test_invalid_log_root_no_fraud(self):
        """Test that matching roots don't constitute fraud."""
        from idi.ian.network.fraud import InvalidLogRootProof, FraudProofVerifier
        
        # Roots match - no fraud
        proof = InvalidLogRootProof(
            goal_id="TEST_GOAL",
            challenged_commit_hash=b'\xab' * 32,
            claimed_root=b'\x00' * 32,
            computed_root=b'\x00' * 32,  # Same as claimed
        )
        
        verifier = FraudProofVerifier()
        valid, reason = verifier.verify(proof)
        
        assert not valid
        assert "no fraud" in reason
    
    def test_wrong_ordering_proof(self, goal_spec):
        """Test wrong ordering fraud proof."""
        from idi.ian.network.fraud import WrongOrderingProof, FraudProofVerifier
        
        # Contribution A has earlier timestamp but higher log index
        proof = WrongOrderingProof(
            goal_id=str(goal_spec.goal_id),
            challenged_commit_hash=b'\xab' * 32,
            contribution_a={"goal_id": str(goal_spec.goal_id)},
            contribution_b={"goal_id": str(goal_spec.goal_id)},
            log_index_a=10,  # Higher index
            log_index_b=5,   # Lower index
            ordering_key_a={"timestamp_ms": 1000, "pack_hash": "00" * 32},  # Earlier
            ordering_key_b={"timestamp_ms": 2000, "pack_hash": "ff" * 32},  # Later
        )
        
        verifier = FraudProofVerifier()
        valid, reason = verifier.verify(proof)
        
        assert valid, reason
        assert "wrong ordering" in reason
    
    def test_fraud_proof_serialization(self):
        """Test fraud proof serialization."""
        from idi.ian.network.fraud import InvalidLogRootProof
        
        proof = InvalidLogRootProof(
            goal_id="TEST_GOAL",
            challenged_commit_hash=b'\xab' * 32,
            claimed_root=b'\x00' * 32,
            computed_root=b'\xff' * 32,
        )
        
        # Serialize
        data = proof.to_dict()
        
        assert data["fraud_type"] == "invalid_log_root"
        assert data["goal_id"] == "TEST_GOAL"
        assert data["claimed_root"] == "00" * 32
        assert data["computed_root"] == "ff" * 32


# =============================================================================
# Economic Security Tests
# =============================================================================

class TestEconomics:
    """Test economic security."""
    
    def test_bond_registration(self):
        """Test committer bond registration."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig
        
        config = EconomicConfig(min_committer_bond=100)
        manager = EconomicManager(node_id="test_node", config=config)
        
        success, reason = manager.register_bond(
            committer_id="committer_001",
            goal_id="GOAL_001",
            amount=1000,
        )
        
        assert success, reason
        
        bond = manager.get_bond("committer_001", "GOAL_001")
        assert bond is not None
        assert bond.amount == 1000
    
    def test_bond_below_minimum(self):
        """Test bond below minimum is rejected."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig
        
        config = EconomicConfig(min_committer_bond=1000)
        manager = EconomicManager(node_id="test_node", config=config)
        
        success, reason = manager.register_bond(
            committer_id="committer_001",
            goal_id="GOAL_001",
            amount=500,  # Below minimum
        )
        
        assert not success
        assert "minimum" in reason
    
    def test_can_commit_authorization(self):
        """Test commit authorization check."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig
        
        config = EconomicConfig(min_committer_bond=100)
        manager = EconomicManager(node_id="test_node", config=config)
        
        # No bond - cannot commit
        can, reason = manager.can_commit("committer_001", "GOAL_001")
        assert not can
        assert "no bond" in reason
        
        # Register bond
        manager.register_bond("committer_001", "GOAL_001", 1000)
        
        # Now can commit
        can, reason = manager.can_commit("committer_001", "GOAL_001")
        assert can
    
    def test_slash_calculation(self):
        """Test slash amount calculation."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig
        
        config = EconomicConfig(
            min_committer_bond=100,
            slash_percentage=0.5,
        )
        manager = EconomicManager(node_id="test_node", config=config)
        
        manager.register_bond("committer_001", "GOAL_001", 1000)
        
        # First slash: 50% of 1000 = 500
        slash_amount = manager.calculate_slash_amount("committer_001", "GOAL_001")
        assert slash_amount == 500
    
    def test_slash_execution(self):
        """Test slash execution."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig, BondStatus
        
        config = EconomicConfig(
            min_committer_bond=100,
            slash_percentage=0.5,
            challenger_reward_percentage=0.25,
        )
        manager = EconomicManager(node_id="test_node", config=config)
        
        manager.register_bond("committer_001", "GOAL_001", 1000)
        
        event = manager.execute_slash(
            committer_id="committer_001",
            goal_id="GOAL_001",
            commit_hash=b'\xab' * 32,
            fraud_type="invalid_log_root",
            challenger_id="challenger_001",
        )
        
        assert event is not None
        assert event.amount_slashed == 500
        assert event.challenger_reward == 125  # 25% of 500
        
        bond = manager.get_bond("committer_001", "GOAL_001")
        assert bond.slash_count == 1
        assert bond.total_slashed == 500
    
    def test_slash_escalation(self):
        """Test slash escalation for repeat offenders."""
        from idi.ian.network.economics import EconomicManager, EconomicConfig
        
        config = EconomicConfig(
            min_committer_bond=100,
            slash_percentage=0.2,  # 20%
            slash_escalation=2.0,  # Double each time
            max_slash_percentage=1.0,
        )
        manager = EconomicManager(node_id="test_node", config=config)
        
        manager.register_bond("committer_001", "GOAL_001", 1000)
        
        # First slash: 20% of 1000 = 200
        event1 = manager.execute_slash(
            "committer_001", "GOAL_001", b'\x01' * 32, "fraud1", "challenger"
        )
        assert event1.amount_slashed == 200
        
        # Second slash: 40% of 800 = 320
        event2 = manager.execute_slash(
            "committer_001", "GOAL_001", b'\x02' * 32, "fraud2", "challenger"
        )
        assert event2.amount_slashed == 320


# =============================================================================
# Evaluation Quorum Tests
# =============================================================================

class TestEvaluationQuorum:
    """Test distributed evaluation quorum."""
    
    def test_evaluator_registration(self):
        """Test evaluator registration."""
        from idi.ian.network.evaluation import (
            EvaluationQuorumManager, EvaluatorInfo, EvaluatorStatus,
            EvaluationQuorumConfig,
        )
        
        config = EvaluationQuorumConfig(evaluator_stake=100)
        manager = EvaluationQuorumManager(node_id="test_node", config=config)
        
        info = EvaluatorInfo(
            node_id="evaluator_001",
            address="tcp://localhost:9001",
            stake_amount=1000,
            supported_goal_ids=["GOAL_001"],
        )
        
        success, reason = manager.register_evaluator(info)
        
        assert success, reason
    
    def test_evaluator_stake_requirement(self):
        """Test evaluator stake requirement."""
        from idi.ian.network.evaluation import (
            EvaluationQuorumManager, EvaluatorInfo, EvaluationQuorumConfig,
        )
        
        config = EvaluationQuorumConfig(evaluator_stake=1000)
        manager = EvaluationQuorumManager(node_id="test_node", config=config)
        
        info = EvaluatorInfo(
            node_id="evaluator_001",
            address="tcp://localhost:9001",
            stake_amount=500,  # Below requirement
        )
        
        success, reason = manager.register_evaluator(info)
        
        assert not success
        assert "insufficient stake" in reason
    
    def test_evaluator_selection(self):
        """Test evaluator selection by reputation."""
        from idi.ian.network.evaluation import (
            EvaluationQuorumManager, EvaluatorInfo, EvaluationQuorumConfig,
        )
        
        config = EvaluationQuorumConfig(evaluator_stake=100)
        manager = EvaluationQuorumManager(node_id="test_node", config=config)
        
        # Register evaluators with different reputations
        for i in range(5):
            info = EvaluatorInfo(
                node_id=f"evaluator_{i:03d}",
                address=f"tcp://localhost:{9000 + i}",
                stake_amount=1000,
                evaluations_completed=i * 10,  # Higher = better reputation
                evaluations_failed=1,
            )
            manager.register_evaluator(info)
        
        # Select top 3
        selected = manager.get_evaluators("GOAL_001", 3)
        
        assert len(selected) == 3
        # Should be sorted by reputation (highest first)
        assert selected[0].evaluations_completed >= selected[1].evaluations_completed
    
    def test_metric_agreement(self):
        """Test metric agreement checking."""
        from idi.ian.network.evaluation import (
            EvaluationQuorumManager, EvaluationResponse, EvaluationQuorumConfig,
        )
        
        config = EvaluationQuorumConfig(
            reward_tolerance=0.05,
            risk_tolerance=0.05,
            complexity_tolerance=0.05,
        )
        manager = EvaluationQuorumManager(node_id="test_node", config=config)
        
        # Responses that agree (within 5%)
        responses = [
            EvaluationResponse(
                request_id="req1",
                evaluator_id=f"eval_{i}",
                success=True,
                metrics={"reward": 0.50 + i * 0.01, "risk": 0.30, "complexity": 0.40},
            )
            for i in range(3)
        ]
        
        agreeing, disagreeing, consensus = manager._check_agreement(responses)
        
        assert len(agreeing) == 3
        assert len(disagreeing) == 0
        assert 0.49 <= consensus["reward"] <= 0.52


# =============================================================================
# Integration Tests
# =============================================================================

class TestDecentralizedNode:
    """Test decentralized node integration."""
    
    def test_node_creation(self, goal_spec):
        """Test node creation."""
        from idi.ian.network.decentralized_node import (
            DecentralizedNode, DecentralizedNodeConfig,
        )
        from idi.ian.network.node import NodeIdentity
        
        identity = NodeIdentity.generate()
        config = DecentralizedNodeConfig(
            accept_contributions=True,
            commit_to_tau=False,  # Disable for testing
        )
        
        node = DecentralizedNode(
            goal_spec=goal_spec,
            identity=identity,
            config=config,
        )
        
        assert node.node_id == identity.node_id
        assert node.goal_id == str(goal_spec.goal_id)
    
    def test_node_info(self, goal_spec):
        """Test node info generation."""
        from idi.ian.network.decentralized_node import DecentralizedNode
        from idi.ian.network.node import NodeIdentity
        
        identity = NodeIdentity.generate()
        node = DecentralizedNode(
            goal_spec=goal_spec,
            identity=identity,
        )
        
        info = node.get_node_info()
        
        assert info.node_id == identity.node_id
        assert str(goal_spec.goal_id) in info.capabilities.goal_ids
    
    @pytest.mark.asyncio
    async def test_node_lifecycle(self, goal_spec):
        """Test node start/stop lifecycle."""
        from idi.ian.network.decentralized_node import DecentralizedNode
        from idi.ian.network.node import NodeIdentity
        
        identity = NodeIdentity.generate()
        node = DecentralizedNode(
            goal_spec=goal_spec,
            identity=identity,
        )
        
        assert not node.is_running
        
        await node.start()
        assert node.is_running
        
        await node.stop()
        assert not node.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
