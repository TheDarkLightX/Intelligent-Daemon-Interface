from idi.training.python.idi_iann.episodic_control import EpisodicQMemory
from idi.training.python.idi_iann.domain import Action


def test_episodic_memory_query_write():
    mem = EpisodicQMemory(capacity=4, k=2, decay=1.0)
    mem.write("s0", Action.BUY, 1.0)
    mem.write("s0", Action.BUY, 3.0)
    est = mem.query("s0", Action.BUY)
    assert est is not None
    assert 1.0 <= est <= 3.0

    # Ensure capacity eviction doesn't crash
    mem.write("s1", Action.SELL, 2.0)
    mem.write("s2", Action.SELL, 2.5)
    mem.write("s3", Action.SELL, 2.6)
