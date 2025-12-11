"""
Tests for Phase 4: Tau Net Integration.

Tests cover:
- Transaction types (IAN_GOAL_REGISTER, IAN_LOG_COMMIT, IAN_UPGRADE)
- Wire format serialization/deserialization
- TauBridge functionality
- TauIntegratedCoordinator
"""

import hashlib
import json
import time
import pytest

from idi.ian.models import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    ContributionMeta,
    Metrics,
    Thresholds,
    EvaluationLimits,
)
from idi.ian.tau_bridge import (
    IanTxType,
    IanGoalRegisterTx,
    IanLogCommitTx,
    IanUpgradeTx,
    IanTauState,
    TauBridge,
    TauBridgeConfig,
    MockTauSender,
    TauIntegratedCoordinator,
    parse_ian_tx,
    create_tau_integrated_coordinator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_goal_spec():
    """Create a sample GoalSpec for testing."""
    return GoalSpec(
        goal_id=GoalID("TEST_TRADING_GOAL"),
        name="Test Trading Goal",
        description="A test goal for trading agents",
        invariant_ids=["I1", "I2", "I3"],
        eval_harness_id="backtest",
        eval_limits=EvaluationLimits(
            max_episodes=100,
            max_steps_per_episode=1000,
            timeout_seconds=60,
            max_memory_mb=512,
        ),
        thresholds=Thresholds(
            min_reward=0.5,
            max_risk=0.3,
            max_complexity=0.5,
        ),
        ranking_weights={"reward": 1.0, "risk": -0.5},
    )


@pytest.fixture
def sample_contribution_meta():
    """Create a sample ContributionMeta for testing."""
    return ContributionMeta(
        pack_hash=hashlib.sha256(b"test_pack").digest(),
        metrics=Metrics(
            reward=0.85,
            risk=0.15,
            complexity=0.2,
            episodes_run=100,
            steps_run=10000,
        ),
        score=0.75,
        contributor_id="test_contributor",
        timestamp_ms=int(time.time() * 1000),
        log_index=42,
    )


# =============================================================================
# Transaction Type Tests
# =============================================================================

class TestIanGoalRegisterTx:
    """Tests for IAN_GOAL_REGISTER transaction."""
    
    def test_from_goal_spec(self, sample_goal_spec):
        """Test creating registration tx from GoalSpec."""
        tx = IanGoalRegisterTx.from_goal_spec(sample_goal_spec)
        
        assert tx.goal_id == sample_goal_spec.goal_id
        assert tx.name == sample_goal_spec.name
        assert tx.description == sample_goal_spec.description
        assert len(tx.goal_spec_hash) == 32
        assert tx.timestamp_ms > 0
    
    def test_wire_roundtrip(self, sample_goal_spec):
        """Test serialization/deserialization."""
        tx = IanGoalRegisterTx.from_goal_spec(sample_goal_spec)
        
        wire = tx.to_wire()
        assert isinstance(wire, bytes)
        
        parsed = IanGoalRegisterTx.from_wire(wire)
        assert parsed.goal_id == tx.goal_id
        assert parsed.goal_spec_hash == tx.goal_spec_hash
        assert parsed.name == tx.name
        assert parsed.timestamp_ms == tx.timestamp_ms
    
    def test_tx_hash_deterministic(self, sample_goal_spec):
        """Test that tx_hash is deterministic."""
        tx1 = IanGoalRegisterTx.from_goal_spec(sample_goal_spec, timestamp_ms=12345)
        tx2 = IanGoalRegisterTx.from_goal_spec(sample_goal_spec, timestamp_ms=12345)
        
        assert tx1.tx_hash == tx2.tx_hash
    
    def test_wire_format_structure(self, sample_goal_spec):
        """Test wire format has expected fields."""
        tx = IanGoalRegisterTx.from_goal_spec(sample_goal_spec)
        wire = tx.to_wire()
        
        obj = json.loads(wire.decode("utf-8"))
        assert obj["type"] == "IAN_GOAL_REGISTER"
        assert "goal_id" in obj
        assert "goal_spec_hash" in obj
        assert "timestamp_ms" in obj


class TestIanLogCommitTx:
    """Tests for IAN_LOG_COMMIT transaction."""
    
    def test_creation(self):
        """Test creating log commit tx."""
        tx = IanLogCommitTx(
            goal_id=GoalID("TEST_GOAL"),
            log_root=b'\x01' * 32,
            log_size=100,
            leaderboard_root=b'\x02' * 32,
            leaderboard_size=10,
            prev_commit_hash=None,
            timestamp_ms=int(time.time() * 1000),
        )
        
        assert tx.log_size == 100
        assert tx.leaderboard_size == 10
    
    def test_wire_roundtrip(self):
        """Test serialization/deserialization."""
        tx = IanLogCommitTx(
            goal_id=GoalID("TEST_GOAL"),
            log_root=b'\xab' * 32,
            log_size=50,
            leaderboard_root=b'\xcd' * 32,
            leaderboard_size=5,
            prev_commit_hash=b'\xef' * 32,
            timestamp_ms=123456789,
        )
        
        wire = tx.to_wire()
        parsed = IanLogCommitTx.from_wire(wire)
        
        assert parsed.goal_id == tx.goal_id
        assert parsed.log_root == tx.log_root
        assert parsed.log_size == tx.log_size
        assert parsed.prev_commit_hash == tx.prev_commit_hash
    
    def test_chain_integrity(self):
        """Test commit chain linking."""
        tx1 = IanLogCommitTx(
            goal_id=GoalID("TEST_GOAL"),
            log_root=b'\x01' * 32,
            log_size=10,
            leaderboard_root=b'\x02' * 32,
            leaderboard_size=5,
            prev_commit_hash=None,
            timestamp_ms=1000,
        )
        
        tx2 = IanLogCommitTx(
            goal_id=GoalID("TEST_GOAL"),
            log_root=b'\x03' * 32,
            log_size=20,
            leaderboard_root=b'\x04' * 32,
            leaderboard_size=8,
            prev_commit_hash=tx1.tx_hash,  # Chain to previous
            timestamp_ms=2000,
        )
        
        assert tx2.prev_commit_hash == tx1.tx_hash


class TestIanUpgradeTx:
    """Tests for IAN_UPGRADE transaction."""
    
    def test_from_contribution_meta(self, sample_contribution_meta):
        """Test creating upgrade tx from ContributionMeta."""
        goal_id = GoalID("TEST_GOAL")
        log_root = b'\xaa' * 32
        
        tx = IanUpgradeTx.from_contribution_meta(
            meta=sample_contribution_meta,
            goal_id=goal_id,
            log_root=log_root,
        )
        
        assert tx.goal_id == goal_id
        assert tx.pack_hash == sample_contribution_meta.pack_hash
        assert tx.score == sample_contribution_meta.score
        assert tx.log_index == sample_contribution_meta.log_index
    
    def test_wire_roundtrip(self, sample_contribution_meta):
        """Test serialization/deserialization."""
        tx = IanUpgradeTx.from_contribution_meta(
            meta=sample_contribution_meta,
            goal_id=GoalID("TEST_GOAL"),
            log_root=b'\xbb' * 32,
            prev_pack_hash=b'\xcc' * 32,
            timestamp_ms=999999,
        )
        
        wire = tx.to_wire()
        parsed = IanUpgradeTx.from_wire(wire)
        
        assert parsed.pack_hash == tx.pack_hash
        assert parsed.prev_pack_hash == tx.prev_pack_hash
        assert parsed.score == tx.score
        assert parsed.cooldown_ok == tx.cooldown_ok
    
    def test_governance_signatures(self, sample_contribution_meta):
        """Test governance signature handling."""
        tx = IanUpgradeTx(
            goal_id=GoalID("TEST_GOAL"),
            pack_hash=sample_contribution_meta.pack_hash,
            prev_pack_hash=None,
            score=0.9,
            metrics={"reward": 0.9},
            log_index=1,
            log_root=b'\x00' * 32,
            contributor_id="test",
            timestamp_ms=1000,
            governance_signatures=[b'\x11' * 64, b'\x22' * 64],
        )
        
        wire = tx.to_wire()
        parsed = IanUpgradeTx.from_wire(wire)
        
        assert len(parsed.governance_signatures) == 2


class TestParseIanTx:
    """Tests for transaction parsing."""
    
    def test_parse_goal_register(self, sample_goal_spec):
        """Test parsing goal register tx."""
        tx = IanGoalRegisterTx.from_goal_spec(sample_goal_spec)
        wire = tx.to_wire()
        
        parsed = parse_ian_tx(wire)
        assert isinstance(parsed, IanGoalRegisterTx)
    
    def test_parse_log_commit(self):
        """Test parsing log commit tx."""
        tx = IanLogCommitTx(
            goal_id=GoalID("TEST"),
            log_root=b'\x00' * 32,
            log_size=0,
            leaderboard_root=b'\x00' * 32,
            leaderboard_size=0,
            prev_commit_hash=None,
            timestamp_ms=0,
        )
        wire = tx.to_wire()
        
        parsed = parse_ian_tx(wire)
        assert isinstance(parsed, IanLogCommitTx)
    
    def test_parse_upgrade(self, sample_contribution_meta):
        """Test parsing upgrade tx."""
        tx = IanUpgradeTx.from_contribution_meta(
            meta=sample_contribution_meta,
            goal_id=GoalID("TEST"),
            log_root=b'\x00' * 32,
        )
        wire = tx.to_wire()
        
        parsed = parse_ian_tx(wire)
        assert isinstance(parsed, IanUpgradeTx)
    
    def test_parse_unknown_type(self):
        """Test parsing unknown tx type raises error."""
        wire = json.dumps({"type": "UNKNOWN_TX"}).encode()
        
        with pytest.raises(ValueError, match="Unknown IAN transaction type"):
            parse_ian_tx(wire)


# =============================================================================
# TauBridge Tests
# =============================================================================

class TestIanTauState:
    """Tests for IanTauState."""
    
    def test_default_state(self):
        """Test default state initialization."""
        state = IanTauState(goal_id=GoalID("TEST"))
        
        assert not state.registered
        assert state.active_policy_hash is None
        assert state.upgrade_count == 0
    
    def test_serialization_roundtrip(self):
        """Test state serialization."""
        state = IanTauState(
            goal_id=GoalID("TEST"),
            registered=True,
            active_policy_hash=b'\xaa' * 32,
            log_root=b'\xbb' * 32,
            upgrade_count=5,
        )
        
        data = state.to_dict()
        restored = IanTauState.from_dict(data)
        
        assert restored.goal_id == state.goal_id
        assert restored.registered == state.registered
        assert restored.active_policy_hash == state.active_policy_hash
        assert restored.upgrade_count == state.upgrade_count


class TestMockTauSender:
    """Tests for MockTauSender."""
    
    def test_send_tx_stores_and_returns_success(self):
        """Test mock sender stores transactions."""
        sender = MockTauSender()
        
        tx_data = b"test_tx_data"
        success, result = sender.send_tx(tx_data)
        
        assert success
        assert len(result) == 64  # hex hash
        assert tx_data in sender.sent_txs


class TestTauBridge:
    """Tests for TauBridge."""
    
    def test_initialization(self):
        """Test bridge initialization."""
        bridge = TauBridge()
        
        assert bridge.sender is not None
        assert bridge.config is not None
    
    def test_get_state_creates_new(self):
        """Test get_state creates state for new goal."""
        bridge = TauBridge()
        goal_id = GoalID("NEW_GOAL")
        
        state = bridge.get_state(goal_id)
        
        assert state.goal_id == goal_id
        assert not state.registered
    
    def test_register_goal(self, sample_goal_spec):
        """Test goal registration."""
        sender = MockTauSender()
        bridge = TauBridge(sender=sender)
        
        success, result = bridge.register_goal(sample_goal_spec)
        
        assert success
        assert len(sender.sent_txs) == 1
        
        state = bridge.get_state(sample_goal_spec.goal_id)
        assert state.registered
    
    def test_commit_log(self, sample_goal_spec):
        """Test log commit."""
        sender = MockTauSender()
        bridge = TauBridge(sender=sender)
        
        # Register first
        bridge.register_goal(sample_goal_spec)
        
        # Commit
        success, result = bridge.commit_log(
            goal_id=sample_goal_spec.goal_id,
            log_root=b'\xaa' * 32,
            log_size=100,
            leaderboard_root=b'\xbb' * 32,
            leaderboard_size=10,
        )
        
        assert success
        assert len(sender.sent_txs) == 2  # register + commit
        
        state = bridge.get_state(sample_goal_spec.goal_id)
        assert state.log_root == b'\xaa' * 32
    
    def test_upgrade_policy(self, sample_goal_spec, sample_contribution_meta):
        """Test policy upgrade."""
        sender = MockTauSender()
        config = TauBridgeConfig(upgrade_cooldown_seconds=0)  # No cooldown
        bridge = TauBridge(sender=sender, config=config)
        
        # Register first
        bridge.register_goal(sample_goal_spec)
        
        # Upgrade
        success, result = bridge.upgrade_policy(
            goal_id=sample_goal_spec.goal_id,
            new_policy=sample_contribution_meta,
            log_root=b'\xcc' * 32,
        )
        
        assert success
        
        state = bridge.get_state(sample_goal_spec.goal_id)
        assert state.active_policy_hash == sample_contribution_meta.pack_hash
        assert state.upgrade_count == 1
    
    def test_should_commit_time_based(self, sample_goal_spec):
        """Test time-based commit trigger."""
        config = TauBridgeConfig(commit_interval_seconds=0)  # Immediate
        bridge = TauBridge(config=config)
        
        bridge.register_goal(sample_goal_spec)
        
        assert bridge.should_commit(sample_goal_spec.goal_id)
    
    def test_should_commit_count_based(self, sample_goal_spec):
        """Test count-based commit trigger."""
        config = TauBridgeConfig(
            commit_interval_seconds=10000,  # Long interval
            commit_threshold_contributions=5,
        )
        bridge = TauBridge(config=config)
        
        bridge.register_goal(sample_goal_spec)
        
        # Not enough contributions
        for _ in range(4):
            bridge.record_contribution(sample_goal_spec.goal_id)
        assert not bridge.should_commit(sample_goal_spec.goal_id)
        
        # Threshold reached
        bridge.record_contribution(sample_goal_spec.goal_id)
        assert bridge.should_commit(sample_goal_spec.goal_id)
    
    def test_serialization_roundtrip(self, sample_goal_spec):
        """Test bridge state serialization."""
        bridge = TauBridge()
        bridge.register_goal(sample_goal_spec)
        
        data = bridge.to_dict()
        restored = TauBridge.from_dict(data)
        
        state = restored.get_state(sample_goal_spec.goal_id)
        assert state.registered


# =============================================================================
# TauIntegratedCoordinator Tests
# =============================================================================

class TestTauIntegratedCoordinator:
    """Tests for TauIntegratedCoordinator."""
    
    def test_process_contribution_triggers_commit(self, sample_goal_spec):
        """Test that processing contributions can trigger commits."""
        from idi.ian.coordinator import IANCoordinator, CoordinatorConfig
        
        config = CoordinatorConfig(leaderboard_capacity=10)
        coordinator = IANCoordinator(goal_spec=sample_goal_spec, config=config)
        
        bridge_config = TauBridgeConfig(
            commit_threshold_contributions=1,  # Commit after each
        )
        sender = MockTauSender()
        bridge = TauBridge(sender=sender, config=bridge_config)
        
        integrated = TauIntegratedCoordinator(coordinator=coordinator, bridge=bridge)
        
        # Register goal
        integrated.register_on_tau()
        assert len(sender.sent_txs) == 1
        
        # Create and process a contribution
        pack = AgentPack(version="1.0", parameters=b"test_params")
        contrib = Contribution(
            goal_id=sample_goal_spec.goal_id,
            agent_pack=pack,
            proofs={},
            contributor_id="test_user",
            seed=42,
        )
        
        result = integrated.process_contribution(contrib)
        
        # Should have triggered commit (threshold=1)
        if result.accepted:
            assert len(sender.sent_txs) >= 2  # register + commit
    
    def test_force_commit(self, sample_goal_spec):
        """Test forcing a commit."""
        from idi.ian.coordinator import IANCoordinator
        
        coordinator = IANCoordinator(goal_spec=sample_goal_spec)
        sender = MockTauSender()
        bridge = TauBridge(sender=sender)
        
        integrated = TauIntegratedCoordinator(coordinator=coordinator, bridge=bridge)
        integrated.register_on_tau()
        
        success, _ = integrated.force_commit()
        
        assert success
        assert len(sender.sent_txs) == 2  # register + commit


class TestCreateTauIntegratedCoordinator:
    """Tests for factory function."""
    
    def test_create_with_defaults(self, sample_goal_spec):
        """Test creating integrated coordinator with defaults."""
        integrated = create_tau_integrated_coordinator(sample_goal_spec)
        
        assert integrated.coordinator is not None
        assert integrated.bridge is not None
    
    def test_create_with_custom_config(self, sample_goal_spec):
        """Test creating with custom configuration."""
        sender = MockTauSender()
        config = TauBridgeConfig(commit_interval_seconds=60)
        
        integrated = create_tau_integrated_coordinator(
            sample_goal_spec,
            tau_sender=sender,
            bridge_config=config,
        )
        
        assert integrated.bridge.sender is sender
        assert integrated.bridge.config.commit_interval_seconds == 60
