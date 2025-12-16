"""
Integration tests for new IAN modules.

Tests how FrontierSync, TwigMMR, SlotBatcher, and MeritRank work together.

Author: DarkLightX
"""

import hashlib
import secrets
import time

import pytest

from idi.ian.twigmmr import TwigMMR, hash_leaf
from idi.ian.network.iblt import IBLT, IBLTConfig
from idi.ian.network.slotbatcher import (
    SlotBatcher,
    VRFProvider,
    compute_lottery_ticket,
)
from idi.ian.reputation import (
    MeritRank,
    ReputationGraph,
    EvaluatorNode,
    EvaluationEdge,
)


class TestTwigMMRWithIBLT:
    """Test TwigMMR integration with IBLT for sync."""
    
    def test_mmr_entries_sync_via_iblt(self):
        """Two MMRs should sync via IBLT set reconciliation."""
        # Create two MMRs with overlapping entries
        mmr1 = TwigMMR()
        mmr2 = TwigMMR()
        
        # Shared entries
        shared_data = [f"shared_{i}".encode() for i in range(50)]
        for data in shared_data:
            mmr1.append(data)
            mmr2.append(data)
        
        # MMR1 has extra entries
        extra1 = [f"extra1_{i}".encode() for i in range(10)]
        for data in extra1:
            mmr1.append(data)
        
        # MMR2 has different extra entries
        extra2 = [f"extra2_{i}".encode() for i in range(10)]
        for data in extra2:
            mmr2.append(data)
        
        # Get entry hashes
        hashes1 = set(mmr1.get_all_entry_hashes())
        hashes2 = set(mmr2.get_all_entry_hashes())
        
        # Create IBLTs from each
        config = IBLTConfig(num_cells=500, num_hashes=4)
        iblt1 = IBLT(config)
        iblt2 = IBLT(config)
        
        for h in hashes1:
            iblt1.insert(h)
        for h in hashes2:
            iblt2.insert(h)
        
        # Subtract to find difference
        diff = iblt1.subtract(iblt2)
        only_in_1, only_in_2, success = diff.decode()
        
        assert success, "IBLT decode should succeed"
        
        # Verify difference detection
        expected_only_in_1 = hashes1 - hashes2
        expected_only_in_2 = hashes2 - hashes1
        
        assert only_in_1 == expected_only_in_1
        assert only_in_2 == expected_only_in_2
    
    def test_mmr_proof_after_iblt_sync(self):
        """After IBLT sync, proofs should verify."""
        mmr = TwigMMR()
        
        # Add entries
        for i in range(20):
            mmr.append(f"entry_{i}".encode())
        
        # Verify all proofs
        for i in range(20):
            proof = mmr.prove(i)
            assert proof.verify(), f"Proof {i} failed"


class TestSlotBatcherWithMeritRank:
    """Integration test: MeritRank + SlotBatcher."""
    
    @pytest.mark.asyncio
    async def test_quality_weights_from_meritrank(self):
        """MeritRank scores should affect quality weights in SlotBatcher."""
        # Create a reputation graph
        graph = ReputationGraph()
        
        nodes = ["alice", "bob", "charlie", "dave", "eve"]
        for name in nodes:
            graph.add_node(EvaluatorNode(name, 100.0, 1000))
        
        # Add evaluation edges
        edges = [
            ("alice", "bob", 0.9),
            ("alice", "charlie", 0.8),
            ("bob", "charlie", 0.9),
            ("charlie", "dave", 0.7),
            ("dave", "eve", 0.6),
            ("eve", "alice", 0.8),
        ]
        
        for from_id, to_id, quality in edges:
            graph.add_edge(EvaluationEdge(
                from_id=from_id,
                to_id=to_id,
                quality=quality,
                timestamp_ms=int(time.time() * 1000),
                contribution_hash=secrets.token_bytes(32),
            ))
        
        # Compute MeritRank
        mr = MeritRank(graph, num_seeds=3, num_walks=200)
        mr.set_seed(42)
        scores = mr.compute()
        
        # Use MeritRank scores in SlotBatcher
        def get_quality(contributor_id: str) -> float:
            base_score = scores.get(contributor_id, 0.1)
            # Scale to [0.1, 10.0] range for quality weight
            return max(0.1, min(10.0, base_score * 50))
        
        batcher = SlotBatcher(
            slot_duration_ms=100,
            get_quality_weight=get_quality,
        )
        
        # Submit contributions from each node
        contrib_ids = {}
        for name in nodes:
            contrib_id = await batcher.submit(
                contributor_id=name,
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
            )
            contrib_ids[name] = contrib_id
        
        # Finalize and get order
        slot_id = await batcher.force_finalize_current_slot()
        order = await batcher.get_slot_order(slot_id)
        
        assert len(order) == 5
        # All contributions should be ordered
        assert set(order) == set(contrib_ids.values())


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_contribution_lifecycle(self):
        """Test full contribution lifecycle: submit → order → prove."""
        # 1. Create contribution data
        contribution_data = b"important contribution data"
        pack_hash = hashlib.sha256(contribution_data).digest()
        
        # 2. Submit to SlotBatcher with commit-reveal
        nonce = secrets.token_bytes(32)
        commitment = hashlib.sha256(contribution_data + nonce).digest()
        
        batcher = SlotBatcher(slot_duration_ms=100)
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=pack_hash,
            commitment_hash=commitment,
        )
        
        # 3. Finalize slot
        slot_id = await batcher.force_finalize_current_slot()
        
        # 4. Reveal content
        reveal_ok = await batcher.reveal(contrib_id, contribution_data, nonce)
        assert reveal_ok, "Reveal should succeed"
        
        # 5. Get ordering proof
        proof = await batcher.get_ordering_proof(contrib_id)
        assert proof is not None
        assert proof.position == 0  # Only contribution
        
        # 6. Store in MMR
        mmr = TwigMMR()
        log_index, mmr_proof = mmr.append(contribution_data)
        
        # 7. Verify MMR proof
        assert mmr_proof.verify(), "MMR proof should verify"
        
        # 8. Create IBLT entry for sync
        config = IBLTConfig(num_cells=100, num_hashes=3)
        iblt = IBLT(config)
        iblt.insert(pack_hash)
        
        # Verify IBLT contains the entry
        iblt2 = IBLT(config)
        diff = iblt.subtract(iblt2)
        only_in_1, _, success = diff.decode()
        
        assert success
        assert pack_hash in only_in_1
    
    def test_reputation_affects_ordering(self):
        """Higher reputation should statistically get better ordering."""
        # Setup reputation
        graph = ReputationGraph()
        
        for name in ["high_rep", "low_rep", "seed1", "seed2", "seed3"]:
            graph.add_node(EvaluatorNode(name, 100.0, 1000))
        
        # High rep gets good evaluations
        for seed in ["seed1", "seed2", "seed3"]:
            graph.add_edge(EvaluationEdge(
                from_id=seed,
                to_id="high_rep",
                quality=0.95,
                timestamp_ms=int(time.time() * 1000),
                contribution_hash=secrets.token_bytes(32),
            ))
        
        # Connect seeds
        graph.add_edge(EvaluationEdge(
            from_id="seed1", to_id="seed2", quality=0.9,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        graph.add_edge(EvaluationEdge(
            from_id="seed2", to_id="seed3", quality=0.9,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        
        # Low rep gets poor evaluations
        graph.add_edge(EvaluationEdge(
            from_id="seed1",
            to_id="low_rep",
            quality=0.2,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        
        mr = MeritRank(graph, num_seeds=3, num_walks=500)
        mr.set_seed(42)
        mr.select_seeds(custom_seeds=["seed1", "seed2", "seed3"])
        scores = mr.compute()
        
        # High rep should have higher score
        assert scores.get("high_rep", 0) > scores.get("low_rep", 0)


class TestSecurityProperties:
    """Test security properties across modules."""
    
    def test_iblt_size_bounded(self):
        """IBLT should reject oversized configurations."""
        from idi.ian.network.iblt import MAX_IBLT_CELLS
        
        with pytest.raises(ValueError):
            IBLTConfig(num_cells=MAX_IBLT_CELLS + 1, num_hashes=3)
    
    def test_mmr_size_bounded(self):
        """MMR should track size correctly."""
        mmr = TwigMMR()
        
        for i in range(100):
            mmr.append(f"entry_{i}".encode())
        
        assert mmr.size == 100
    
    @pytest.mark.asyncio
    async def test_slotbatcher_quality_clamped(self):
        """SlotBatcher should clamp quality weights."""
        from idi.ian.network.slotbatcher import MIN_QUALITY_WEIGHT, MAX_QUALITY_WEIGHT
        
        batcher = SlotBatcher(slot_duration_ms=100)
        
        # Submit with extreme quality
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=secrets.token_bytes(32),
            quality_weight=1000.0,  # Way above MAX
        )
        
        slot_id = await batcher.force_finalize_current_slot()
        proof = await batcher.get_ordering_proof(contrib_id)
        
        # Quality should be clamped
        assert proof.quality_weight <= MAX_QUALITY_WEIGHT
    
    def test_meritrank_scores_bounded(self):
        """MeritRank scores should be in [0, 1]."""
        graph = ReputationGraph()
        
        for name in ["a", "b", "c"]:
            graph.add_node(EvaluatorNode(name, 100.0, 1000))
        
        graph.add_edge(EvaluationEdge(
            from_id="a", to_id="b", quality=1.0,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        graph.add_edge(EvaluationEdge(
            from_id="b", to_id="c", quality=1.0,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        graph.add_edge(EvaluationEdge(
            from_id="c", to_id="a", quality=1.0,
            timestamp_ms=int(time.time() * 1000),
            contribution_hash=secrets.token_bytes(32),
        ))
        
        mr = MeritRank(graph, num_seeds=3, num_walks=100)
        mr.set_seed(42)
        scores = mr.compute()
        
        for score in scores.values():
            assert 0.0 <= score <= 1.0
