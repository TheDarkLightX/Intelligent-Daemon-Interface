import asyncio
import logging
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any, Dict

# Fix path
current_dir = os.getcwd()
sys.path.append(current_dir)

from idi.ian.network.skiplist_mempool import IndexedSkipListMempool as SkipListMempool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockContribution:
    pack_hash: bytes
    contributor_id: str
    goal_id: str
    seed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_hash": self.pack_hash.hex(),
            "contributor_id": self.contributor_id,
            "goal_id": self.goal_id,
            "seed": self.seed
        }

class TestMempoolLifecycle(unittest.TestCase):
    def setUp(self):
        # Use small max_size for easy testing of FULL state
        self.mempool = SkipListMempool(max_size=3)

    def tearDown(self):
        pass

    def test_lifecycle_states(self):
        """Test transitioning between ACTIVE, FULL, PAUSED, DRAINING."""
        asyncio.run(self._test_lifecycle())

    async def _test_lifecycle(self):
        mp = self.mempool
        
        # 1. Initial State
        self.assertEqual(mp.status, 'ACTIVE')
        self.assertTrue(mp.is_accepting)
        
        # 2. Add until FULL
        c1 = MockContribution(b'1'*32, 'alice', 'goal1')
        c2 = MockContribution(b'2'*32, 'bob', 'goal1')
        c3 = MockContribution(b'3'*32, 'charlie', 'goal1')
        
        ok, msg = await mp.add(c1)
        self.assertTrue(ok)
        self.assertEqual(mp.status, 'ACTIVE')
        
        ok, msg = await mp.add(c2)
        self.assertTrue(ok)
        self.assertEqual(mp.status, 'ACTIVE')
        
        ok, msg = await mp.add(c3)
        self.assertTrue(ok)
        # Should be FULL now (count=3, max=3)
        self.assertEqual(mp.status, 'FULL')
        self.assertFalse(mp.is_accepting)
        
        # 3. Add to FULL (with eviction)
        # SkipListMempool AUTO EVICTS.
        # So it should: Evict (status -> ACTIVE) -> Add (status -> FULL).
        # Externally status seems FULL before and after.
        c4 = MockContribution(b'4'*32, 'dave', 'goal1')
        ok, msg = await mp.add(c4)
        self.assertTrue(ok)
        self.assertEqual(mp.status, 'FULL')
        
        # Verify c1 (oldest) was evicted?
        # c1 timestamp was earliest (wall clock).
        # We need to ensure c1 is gone.
        # But wait, ordering key uses wall clock time. 
        # With asyncio.run, time might be same.
        # But eviction removes min by received_at (wall clock).
        # c1 added first, likely earliest.
        self.assertFalse(await mp.contains(b'1'*32)) # Should be gone strictly?
        # Actually it depends on sort.
        # But invariant: size <= 3.
        self.assertEqual(mp.size, 3)

        # 4. Pause
        res = mp.pause()
        self.assertTrue(res)
        self.assertEqual(mp.status, 'PAUSED')
        self.assertFalse(mp.is_accepting)
        
        # Try add (should fail)
        c5 = MockContribution(b'5'*32, 'eve', 'goal1')
        ok, msg = await mp.add(c5)
        self.assertFalse(ok) # REJECTED
        self.assertEqual(mp.status, 'PAUSED')

        # 5. Resume
        res = mp.resume()
        self.assertTrue(res)
        self.assertEqual(mp.status, 'FULL') # Resume to FULL since count=3
        
        # 6. Pop
        entry = await mp.pop_next()
        self.assertIsNotNone(entry)
        # Count 2. Max 3. Status -> ACTIVE.
        self.assertEqual(mp.status, 'ACTIVE')
        self.assertTrue(mp.is_accepting)
        self.assertEqual(mp.size, 2)
        
        # 7. Drain
        res = mp.drain()
        self.assertTrue(res)
        self.assertEqual(mp.status, 'DRAINING')
        
        # Add should fail
        c6 = MockContribution(b'6'*32, 'frank', 'goal1')
        ok, msg = await mp.add(c6)
        self.assertFalse(ok)
        
        # Pop should work
        entry = await mp.pop_next()
        self.assertIsNotNone(entry)
        self.assertEqual(mp.status, 'DRAINING')
        self.assertEqual(mp.size, 1)
        
        entry = await mp.pop_next()
        self.assertIsNotNone(entry)
        self.assertEqual(mp.status, 'DRAINING')
        self.assertEqual(mp.size, 0)
        
        logger.info("Mempool Lifecycle Verification PASSED")

if __name__ == '__main__':
    unittest.main()
