from __future__ import annotations

from idi.ian.network.economics import ChallengeBond


def test_challenge_bond_persists_min_challenge_bond_roundtrip() -> None:
    bond = ChallengeBond(
        challenger_id="challenger_1",
        goal_id="GOAL_1",
        challenged_commit_hash=b"\xAB" * 32,
        amount=1234,
        min_challenge_bond=1234,
    )

    data = bond.to_dict()
    assert data["min_challenge_bond"] == 1234

    loaded = ChallengeBond.from_dict(data)
    assert loaded.min_challenge_bond == 1234

