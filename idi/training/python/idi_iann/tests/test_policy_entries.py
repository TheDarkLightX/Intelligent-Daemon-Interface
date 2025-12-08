from idi.training.python.idi_iann.policy import LookupPolicy
from idi.training.python.idi_iann.domain import Action


def test_lookup_policy_to_entries_roundtrip():
    policy = LookupPolicy()
    policy.update("s0", Action.BUY, 1.5)
    policy.update("s0", Action.HOLD, -0.25)
    entries = policy.to_entries()

    assert "s0" in entries
    assert entries["s0"]["buy"] > entries["s0"]["hold"]
