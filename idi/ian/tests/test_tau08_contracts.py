"""Native execution against independent integer/boundary oracles."""

from itertools import product
from pathlib import Path
import random
import pytest
from idi.ian.decision_core import half_slash, deposit, U64_MAX

tau = pytest.importorskip("tau")
ROOT = Path(__file__).parents[1] / "tau_rules"


def execute(name, rows):
    tau.reset()
    tau.set_severity("error")
    tau.set_charvar(False)
    result = tau.get_interpreter((ROOT / (name + ".tau")).read_text())
    assert result, result.report.errors
    for row in rows:
        inputs = tau.get_inputs_for_step(result.value).value
        output = tau.step(result.value, {n: str(row.get(n.name, 0)) for n in inputs})
        assert output, output.report.errors
        yield {n.name: int(v) for n, v in output.value.items()}


def test_registration_all_boolean_conditions():
    rows = [
        dict(i20=2 if op else 1, i21=exists, i22=hash_, i23=length, i24=gov, i25=sig)
        for op, exists, hash_, length, sig in product(range(2), repeat=5)
        for gov in range(4)
    ]
    for r, o in zip(rows, execute("ian_goal_register", rows)):
        expected = (
            r["i20"] == 2
            and not r["i21"]
            and bool(r["i22"])
            and bool(r["i23"])
            and r["i24"] <= 2
            and bool(r["i25"])
        )
        assert o["o20"] == expected and o["o21"] == int(not expected)


def test_log_commit_required_evidence_and_monotonic_size():
    base = dict(
        i30=3,
        i31=1,
        i32=11,
        i33=100,
        i34=12,
        i35=20,
        i36=123,
        i37=1000,
        i38=99,
        i39=123,
        i3A=1,
        i3B=1,
        i3C=1000,
        i3D=124,
    )
    rows = [base] + [
        dict(base, **{k: v})
        for k, v in [
            ("i30", 2),
            ("i31", 0),
            ("i32", 0),
            ("i33", 98),
            ("i34", 0),
            ("i35", 101),
            ("i36", 0),
            ("i37", 999),
            ("i3A", 0),
            ("i3B", 0),
            ("i3D", 0),
        ]
    ]
    for j, (r, o) in enumerate(zip(rows, execute("ian_log_commit", rows))):
        assert o["o30"] == int(j == 0)
        assert o["o31"] == (r["i3D"] if j == 0 else r["i39"])


def test_upgrade_clock_governance_and_counter_boundaries():
    base = dict(
        i40=4,
        i41=1,
        i42=123,
        i43=12,
        i44=0,
        i45=55,
        i46=11,
        i47=100,
        i48=110,
        i49=1,
        i4A=1,
        i4B=1,
        i4C=55,
        i4D=1,
        i4E=10,
        i4F=1,
        i4G=1,
        i4H=7,
    )
    rows = [base] + [
        dict(base, **{k: v})
        for k, v in [
            ("i40", 3),
            ("i41", 0),
            ("i42", 0),
            ("i43", 11),
            ("i44", 1),
            ("i45", 56),
            ("i48", 99),
            ("i48", 109),
            ("i4A", 0),
            ("i4F", 0),
            ("i4G", 0),
            ("i4H", U64_MAX),
        ]
    ]
    for j, (r, o) in enumerate(zip(rows, execute("ian_upgrade", rows))):
        assert o["o40"] == int(j == 0) and o["o41"] == (8 if j == 0 else r["i4H"])


def test_challenge_window_and_exclusive_slash():
    rows = []
    for now, exists, resolved, proof, unequal, bond in product(
        [99, 100, 109, 110, 111], [0, 1], [0, 1], [0, 1], [0, 1], [4, 5]
    ):
        rows.append(
            dict(
                i50=5,
                i51=exists,
                i52=resolved,
                i54=100,
                i55=now,
                i56=bond,
                i57=7,
                i58=8 if unequal else 7,
                i59=proof,
                i5A=10,
                i5B=5,
            )
        )
    for r, o in zip(rows, execute("ian_challenge", rows)):
        admitted = bool(
            r["i51"] and not r["i52"] and 100 <= r["i55"] < 110 and r["i56"] >= 5
        )
        fraud = admitted and bool(r["i59"]) and r["i57"] != r["i58"]
        assert o == dict(
            o50=int(admitted), o51=int(fraud), o52=int(admitted and not fraud)
        )


def test_economics_matches_integer_oracle_and_conserves_all_units():
    rng = random.Random(20261104)
    bonds = (
        list(range(32))
        + [U64_MAX - 1, U64_MAX]
        + [rng.randrange(1 << 64) for _ in range(100)]
    )
    rows = [dict(i60=6, i61=1, i63=b, i64=0, i65=U64_MAX) for b in bonds]
    rows += [
        dict(i60=7, i61=1, i63=b, i64=a, i65=m)
        for b, a, m in [
            (8, 2, 10),
            (8, 3, 10),
            (U64_MAX, 1, U64_MAX),
            (U64_MAX - 1, 1, U64_MAX),
            (5, 0, 9),
        ]
    ]
    for r, o in zip(rows, execute("ian_economics", rows)):
        if r["i60"] == 6:
            remaining, reward, burn, treasury = half_slash(r["i63"])
            assert (o["o65"], o["o62"], o["o63"], o["o64"]) == (
                remaining,
                reward,
                burn,
                treasury,
            )
            assert o["o65"] + o["o62"] + o["o63"] + o["o64"] == r["i63"]
        else:
            assert o["o65"] == deposit(r["i63"], r["i64"], r["i65"]) and o["o61"] == 0


def test_one_goal_state_rejects_conflicting_updates():
    rows = [
        {},
        dict(i0=1),
        dict(i1=1, i3=123),
        dict(i2=1, i4=456),
        dict(i1=1, i2=1, i3=9, i4=8),
        {},
    ]
    output = list(execute("ian_state", rows))
    assert output[0] == dict(o0=0, o1=0, o2=0, o3=0)
    assert output[2] == dict(o0=1, o1=123, o2=0, o3=0)
    assert output[3] == dict(o0=1, o1=123, o2=456, o3=1)
    assert output[4] == output[3] == output[5]
