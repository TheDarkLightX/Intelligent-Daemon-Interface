"""
Tests for IAN Phase 1: Core Data Structures.

Tests cover:
- Models (GoalID, Metrics, Thresholds, AgentPack, etc.)
- MerkleMountainRange (append, root, proofs)
- Leaderboard (top-K, eviction, ties)
- ParetoFrontier (domination, non-dominated set)
- DedupService (Bloom filter + index)
"""

import hashlib
import os
import pytest
import time

# Import IAN modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from idi.ian.models import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    ContributionMeta,
    Metrics,
    Thresholds,
    EvaluationLimits,
    GoalState,
)
from idi.ian.mmr import MerkleMountainRange, MMRProof
from idi.ian.leaderboard import Leaderboard, ParetoFrontier
from idi.ian.dedup import BloomFilter, DedupIndex, DedupService


# =============================================================================
# Model Tests
# =============================================================================

class TestGoalID:
    """Tests for GoalID validation and serialization."""
    
    def test_valid_goal_id(self):
        gid = GoalID("OWNERLESS_VC_AGENT")
        assert str(gid) == "OWNERLESS_VC_AGENT"
        assert gid.to_bytes() == b"OWNERLESS_VC_AGENT"
    
    def test_empty_goal_id_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            GoalID("")
    
    def test_long_goal_id_raises(self):
        with pytest.raises(ValueError, match="too long"):
            GoalID("a" * 65)
    
    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            GoalID("goal-with-dash")
    
    def test_goal_id_hashable(self):
        gid1 = GoalID("TEST")
        gid2 = GoalID("TEST")
        assert hash(gid1) == hash(gid2)
        assert gid1 == gid2
        
        s = {gid1, gid2}
        assert len(s) == 1


class TestMetrics:
    """Tests for Metrics data class."""
    
    def test_create_metrics(self):
        m = Metrics(reward=0.95, risk=0.1, complexity=0.5)
        assert m.reward == 0.95
        assert m.risk == 0.1
        assert m.complexity == 0.5
    
    def test_metrics_optional_fields(self):
        m = Metrics(reward=1.0, risk=0.0, complexity=0.0, sharpe_ratio=2.5, max_drawdown=0.1)
        assert m.sharpe_ratio == 2.5
        assert m.max_drawdown == 0.1
    
    def test_metrics_to_dict(self):
        m = Metrics(reward=0.5, risk=0.2, complexity=0.3)
        d = m.to_dict()
        assert d["reward"] == 0.5
        assert d["risk"] == 0.2
    
    def test_metrics_from_dict(self):
        d = {"reward": 0.5, "risk": 0.2, "complexity": 0.3}
        m = Metrics.from_dict(d)
        assert m.reward == 0.5


class TestThresholds:
    """Tests for Thresholds checker."""
    
    def test_default_thresholds_pass(self):
        t = Thresholds()
        m = Metrics(reward=0.5, risk=0.5, complexity=0.5)
        passed, reason = t.check(m)
        assert passed
        assert reason == "passed"
    
    def test_low_reward_fails(self):
        t = Thresholds(min_reward=0.6)
        m = Metrics(reward=0.5, risk=0.1, complexity=0.1)
        passed, reason = t.check(m)
        assert not passed
        assert "reward" in reason
    
    def test_high_risk_fails(self):
        t = Thresholds(max_risk=0.2)
        m = Metrics(reward=0.9, risk=0.5, complexity=0.1)
        passed, reason = t.check(m)
        assert not passed
        assert "risk" in reason


class TestAgentPack:
    """Tests for AgentPack."""
    
    def test_create_agent_pack(self):
        pack = AgentPack(version="1.0", parameters=b"test_params")
        assert pack.version == "1.0"
        assert len(pack.pack_hash) == 32
    
    def test_empty_params_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            AgentPack(version="1.0", parameters=b"")
    
    def test_pack_hash_deterministic(self):
        pack1 = AgentPack(version="1.0", parameters=b"params", metadata={"a": 1})
        pack2 = AgentPack(version="1.0", parameters=b"params", metadata={"a": 1})
        assert pack1.pack_hash == pack2.pack_hash
    
    def test_different_params_different_hash(self):
        pack1 = AgentPack(version="1.0", parameters=b"params1")
        pack2 = AgentPack(version="1.0", parameters=b"params2")
        assert pack1.pack_hash != pack2.pack_hash


class TestContributionMeta:
    """Tests for ContributionMeta."""
    
    def test_create_contribution_meta(self):
        m = Metrics(reward=0.9, risk=0.1, complexity=0.2)
        meta = ContributionMeta(
            pack_hash=b"\x00" * 32,
            metrics=m,
            score=0.85,
            contributor_id="alice",
            timestamp_ms=1234567890,
        )
        assert meta.score == 0.85
    
    def test_invalid_hash_length_raises(self):
        m = Metrics(reward=0.9, risk=0.1, complexity=0.2)
        with pytest.raises(ValueError, match="must be 32 bytes"):
            ContributionMeta(
                pack_hash=b"\x00" * 16,
                metrics=m,
                score=0.5,
                contributor_id="bob",
                timestamp_ms=123,
            )
    
    def test_serialization_roundtrip(self):
        m = Metrics(reward=0.9, risk=0.1, complexity=0.2)
        meta = ContributionMeta(
            pack_hash=b"\xab" * 32,
            metrics=m,
            score=0.85,
            contributor_id="alice",
            timestamp_ms=1234567890,
        )
        d = meta.to_dict()
        meta2 = ContributionMeta.from_dict(d)
        assert meta.pack_hash == meta2.pack_hash
        assert meta.score == meta2.score


# =============================================================================
# MMR Tests
# =============================================================================

class TestMerkleMountainRange:
    """Tests for Merkle Mountain Range."""
    
    def test_empty_mmr(self):
        mmr = MerkleMountainRange()
        assert mmr.size == 0
        assert mmr.get_root() == b"\x00" * 32
    
    def test_single_append(self):
        mmr = MerkleMountainRange()
        idx = mmr.append(b"leaf1")
        assert idx == 0
        assert mmr.size == 1
        assert mmr.get_root() != b"\x00" * 32
    
    def test_multiple_appends(self):
        mmr = MerkleMountainRange()
        for i in range(10):
            idx = mmr.append(f"leaf{i}".encode())
            assert idx == i
        assert mmr.size == 10
    
    def test_root_changes_on_append(self):
        mmr = MerkleMountainRange()
        mmr.append(b"leaf1")
        root1 = mmr.get_root()
        mmr.append(b"leaf2")
        root2 = mmr.get_root()
        assert root1 != root2
    
    def test_deterministic_root(self):
        mmr1 = MerkleMountainRange()
        mmr2 = MerkleMountainRange()
        
        for data in [b"a", b"b", b"c"]:
            mmr1.append(data)
            mmr2.append(data)
        
        assert mmr1.get_root() == mmr2.get_root()
    
    def test_serialization_roundtrip(self):
        mmr = MerkleMountainRange()
        for i in range(5):
            mmr.append(f"leaf{i}".encode())
        
        d = mmr.to_dict()
        mmr2 = MerkleMountainRange.from_dict(d)
        
        assert mmr.size == mmr2.size
        assert mmr.get_root() == mmr2.get_root()
    
    def test_large_mmr(self):
        """Verify O(log N) behavior with many entries."""
        mmr = MerkleMountainRange()
        for i in range(1000):
            mmr.append(f"leaf{i}".encode())
        
        assert mmr.size == 1000
        # Root should be computed quickly
        root = mmr.get_root()
        assert len(root) == 32


# =============================================================================
# Leaderboard Tests
# =============================================================================

class TestLeaderboard:
    """Tests for top-K Leaderboard."""
    
    def _make_meta(self, score: float, timestamp: int = None, suffix: str = "") -> ContributionMeta:
        """Helper to create test ContributionMeta."""
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        # Create unique hash based on score + suffix
        h = hashlib.sha256(f"{score}_{suffix}".encode()).digest()
        return ContributionMeta(
            pack_hash=h,
            metrics=Metrics(reward=score, risk=0.1, complexity=0.1),
            score=score,
            contributor_id="test",
            timestamp_ms=timestamp,
        )
    
    def test_empty_leaderboard(self):
        lb = Leaderboard(capacity=10)
        assert len(lb) == 0
        assert lb.top_k() == []
        assert lb.worst_score() is None
    
    def test_add_single(self):
        lb = Leaderboard(capacity=10)
        meta = self._make_meta(0.9)
        added = lb.add(meta)
        assert added
        assert len(lb) == 1
        assert lb.worst_score() == 0.9
    
    def test_capacity_bound(self):
        lb = Leaderboard(capacity=3)
        for i in range(5):
            lb.add(self._make_meta(i * 0.1, suffix=str(i)))
        
        assert len(lb) == 3
        # Should have top 3 scores: 0.4, 0.3, 0.2
        scores = [m.score for m in lb.top_k()]
        assert max(scores) == 0.4
    
    def test_eviction(self):
        lb = Leaderboard(capacity=2)
        lb.add(self._make_meta(0.5, suffix="a"))
        lb.add(self._make_meta(0.6, suffix="b"))
        
        # Adding better one should evict 0.5
        lb.add(self._make_meta(0.7, suffix="c"))
        
        assert len(lb) == 2
        scores = sorted([m.score for m in lb.top_k()], reverse=True)
        assert scores == [0.7, 0.6]
    
    def test_reject_worse(self):
        lb = Leaderboard(capacity=2)
        lb.add(self._make_meta(0.5, suffix="a"))
        lb.add(self._make_meta(0.6, suffix="b"))
        
        # Adding worse one should be rejected
        added = lb.add(self._make_meta(0.4, suffix="c"))
        assert not added
        assert len(lb) == 2
    
    def test_tie_breaking_by_timestamp(self):
        lb = Leaderboard(capacity=1)
        
        # First entry
        lb.add(self._make_meta(0.5, timestamp=2000, suffix="later"))
        
        # Same score but earlier timestamp should win
        added = lb.add(self._make_meta(0.5, timestamp=1000, suffix="earlier"))
        assert added
        
        entries = lb.top_k()
        assert len(entries) == 1
        assert entries[0].timestamp_ms == 1000
    
    def test_get_active_policy(self):
        lb = Leaderboard(capacity=10)
        lb.add(self._make_meta(0.3, suffix="a"))
        lb.add(self._make_meta(0.9, suffix="b"))
        lb.add(self._make_meta(0.5, suffix="c"))
        
        active = lb.get_active_policy()
        assert active is not None
        assert active.score == 0.9
    
    def test_serialization_roundtrip(self):
        lb = Leaderboard(capacity=5)
        for i in range(3):
            lb.add(self._make_meta(i * 0.3, suffix=str(i)))
        
        d = lb.to_dict()
        lb2 = Leaderboard.from_dict(d)
        
        assert len(lb) == len(lb2)
        assert lb.best_score() == lb2.best_score()


class TestParetoFrontier:
    """Tests for Pareto frontier."""
    
    def _make_meta(
        self, reward: float, risk: float, complexity: float, suffix: str = ""
    ) -> ContributionMeta:
        h = hashlib.sha256(f"{reward}_{risk}_{complexity}_{suffix}".encode()).digest()
        return ContributionMeta(
            pack_hash=h,
            metrics=Metrics(reward=reward, risk=risk, complexity=complexity),
            score=reward - 0.5 * risk,
            contributor_id="test",
            timestamp_ms=int(time.time() * 1000),
        )
    
    def test_empty_frontier(self):
        pf = ParetoFrontier()
        assert len(pf) == 0
        assert pf.frontier() == []
    
    def test_single_entry(self):
        pf = ParetoFrontier()
        meta = self._make_meta(0.9, 0.1, 0.2)
        added = pf.add(meta)
        assert added
        assert len(pf) == 1
    
    def test_dominated_rejected(self):
        pf = ParetoFrontier()
        # Add a strong candidate
        pf.add(self._make_meta(0.9, 0.1, 0.1, suffix="strong"))
        
        # Add a dominated candidate (worse in all dimensions)
        added = pf.add(self._make_meta(0.8, 0.2, 0.2, suffix="weak"))
        assert not added
        assert len(pf) == 1
    
    def test_non_dominated_added(self):
        pf = ParetoFrontier()
        # Add candidate with high reward, high risk
        pf.add(self._make_meta(0.9, 0.5, 0.1, suffix="risky"))
        
        # Add candidate with lower reward, lower risk (not dominated)
        added = pf.add(self._make_meta(0.7, 0.1, 0.1, suffix="safe"))
        assert added
        assert len(pf) == 2
    
    def test_dominating_entry_removes_dominated(self):
        pf = ParetoFrontier()
        # Add a weak candidate
        pf.add(self._make_meta(0.5, 0.5, 0.5, suffix="weak"))
        assert len(pf) == 1
        
        # Add a dominating candidate
        pf.add(self._make_meta(0.9, 0.1, 0.1, suffix="strong"))
        assert len(pf) == 1
        
        # Only the strong one should remain
        frontier = pf.frontier()
        assert frontier[0].metrics.reward == 0.9


# =============================================================================
# Dedup Tests
# =============================================================================

class TestBloomFilter:
    """Tests for Bloom filter."""
    
    def test_empty_bloom(self):
        bf = BloomFilter(expected_items=100)
        assert bf.count == 0
        assert not bf.maybe_contains(b"test")
    
    def test_add_and_check(self):
        bf = BloomFilter(expected_items=100)
        bf.add(b"hello")
        assert bf.maybe_contains(b"hello")
    
    def test_false_negative_impossible(self):
        """Bloom filter must never have false negatives."""
        bf = BloomFilter(expected_items=1000)
        items = [f"item_{i}".encode() for i in range(100)]
        
        for item in items:
            bf.add(item)
        
        for item in items:
            assert bf.maybe_contains(item), f"False negative for {item}"
    
    def test_reasonable_false_positive_rate(self):
        """False positive rate should be close to target."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        
        # Add expected items
        for i in range(10000):
            bf.add(f"item_{i}".encode())
        
        # Check items that were NOT added
        false_positives = 0
        test_count = 10000
        for i in range(test_count):
            if bf.maybe_contains(f"not_added_{i}".encode()):
                false_positives += 1
        
        actual_fpr = false_positives / test_count
        # Should be roughly around 1% (allow some variance)
        assert actual_fpr < 0.05, f"FPR too high: {actual_fpr}"
    
    def test_serialization_roundtrip(self):
        bf = BloomFilter(expected_items=100)
        for i in range(10):
            bf.add(f"item_{i}".encode())
        
        d = bf.to_dict()
        bf2 = BloomFilter.from_dict(d)
        
        for i in range(10):
            assert bf2.maybe_contains(f"item_{i}".encode())


class TestDedupIndex:
    """Tests for DedupIndex."""
    
    def test_empty_index(self):
        idx = DedupIndex()
        assert len(idx) == 0
        assert not idx.contains(b"\x00" * 32)
    
    def test_add_and_contains(self):
        idx = DedupIndex()
        h = b"\xab" * 32
        idx.add(h, 42)
        assert idx.contains(h)
        assert idx.get_log_index(h) == 42
    
    def test_invalid_hash_length(self):
        idx = DedupIndex()
        with pytest.raises(ValueError, match="must be 32 bytes"):
            idx.add(b"short", 0)


class TestDedupService:
    """Tests for two-tier DedupService."""
    
    def test_empty_service(self):
        ds = DedupService()
        assert ds.count == 0
        assert not ds.is_duplicate(b"\x00" * 32)
    
    def test_add_and_check(self):
        ds = DedupService()
        h = b"\xab" * 32
        ds.add(h, 0)
        assert ds.is_duplicate(h)
        assert ds.get_log_index(h) == 0
    
    def test_fast_path_for_new_items(self):
        """Items not in Bloom should skip index lookup."""
        ds = DedupService(expected_contributions=1000)
        
        # Add some items
        for i in range(10):
            ds.add(hashlib.sha256(f"item_{i}".encode()).digest(), i)
        
        # Check something definitely not added
        # (Bloom filter should return False immediately)
        new_hash = hashlib.sha256(b"definitely_new").digest()
        assert not ds.is_duplicate(new_hash)
    
    def test_serialization_roundtrip(self):
        ds = DedupService()
        for i in range(5):
            h = hashlib.sha256(f"item_{i}".encode()).digest()
            ds.add(h, i)
        
        d = ds.to_dict()
        ds2 = DedupService.from_dict(d)
        
        assert ds.count == ds2.count
        for i in range(5):
            h = hashlib.sha256(f"item_{i}".encode()).digest()
            assert ds2.is_duplicate(h)


# =============================================================================
# Integration Test
# =============================================================================

class TestPhase1Integration:
    """Integration test combining all Phase 1 components."""
    
    def test_full_flow(self):
        """Test creating contributions and adding to data structures."""
        # Create goal
        goal_id = GoalID("TEST_GOAL")
        
        # Create MMR for log
        mmr = MerkleMountainRange()
        
        # Create leaderboard
        lb = Leaderboard(capacity=5)
        
        # Create dedup service
        dedup = DedupService()
        
        # Simulate adding contributions
        for i in range(10):
            # Create agent pack
            pack = AgentPack(
                version="1.0",
                parameters=f"params_{i}".encode(),
                metadata={"iteration": i},
            )
            
            # Check dedup
            if dedup.is_duplicate(pack.pack_hash):
                continue
            
            # Create metrics
            metrics = Metrics(
                reward=0.5 + i * 0.05,
                risk=0.1 + i * 0.02,
                complexity=0.1,
            )
            
            # Add to log
            log_index = mmr.append(pack.pack_hash)
            
            # Add to dedup
            dedup.add(pack.pack_hash, log_index)
            
            # Create meta and add to leaderboard
            meta = ContributionMeta(
                pack_hash=pack.pack_hash,
                metrics=metrics,
                score=metrics.reward - 0.5 * metrics.risk,
                contributor_id="test_user",
                timestamp_ms=int(time.time() * 1000) + i,
                log_index=log_index,
            )
            lb.add(meta)
        
        # Verify state
        assert mmr.size == 10
        assert len(lb) == 5  # Capacity
        assert dedup.count == 10
        
        # Verify leaderboard has best candidates
        top = lb.top_k()
        assert len(top) == 5
        
        # Get root for anchoring
        log_root = mmr.get_root()
        lb_root = lb.get_root()
        
        assert len(log_root) == 32
        assert len(lb_root) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
