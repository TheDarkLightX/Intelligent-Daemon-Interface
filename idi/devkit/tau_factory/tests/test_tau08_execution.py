"""Native 0.8 behavioral oracles; skipped when the separately licensed API is absent."""

import itertools, json, re, random
from pathlib import Path
import pytest
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

tau = pytest.importorskip("tau")
ROOT = Path(__file__).parent / "tau08_cases"


def execute(pattern, rows, text=None):
    tau.reset()
    tau.set_severity("error")
    tau.set_charvar(False)
    text = re.sub(
        r'(in|out) file\("[^"\n]+"\)',
        r"\1 console",
        text if text is not None else (ROOT / (pattern + ".tau")).read_text(),
    )
    r = tau.get_interpreter(text)
    assert r, r.report.errors
    output = []
    for row in rows:
        names = tau.get_inputs_for_step(r.value).value
        z = tau.step(r.value, {n: str(row[int(n.name[1:], 16)]) for n in names})
        assert z, z.report.errors
        output.append({n.name: int(v) for n, v in z.value.items()})
    return output


def test_counter_wraps_and_holds():
    bits = [0] + [1] * 18 + [0, 1, 0, 1]
    actual = execute("multi_bit_counter", [[b] for b in bits])
    state = 0
    for t, row in enumerate(actual):
        if t:
            state = (state + bits[t]) % 8
        assert row["o0"] == state


def test_weighted_sum_exhaustive():
    rows = list(itertools.product(range(2), repeat=2))
    for inputs, output in zip(rows, execute("weighted_vote", rows)):
        assert output["o0"] == sum(a * b for a, b in zip(inputs, [3, 2]))


def test_time_lock_boundary_and_modular_end():
    rows = [[100, 20, t] for t in [0, 99, 100, 119, 120, 121, 65535]] + [
        [65535, 2, t] for t in [0, 1, 2, 65535]
    ]
    for (start, duration, now), output in zip(rows, execute("time_lock", rows)):
        assert output["o0"] == int(now < ((start + duration) % 65536))


def test_independent_single_input_regions_are_combinational():
    rows = list(itertools.product(range(2), repeat=2)) * 3
    for (a, b), output in zip(rows, execute("orthogonal_regions", rows)):
        assert (output["o0"], output["o1"]) == (a, b)


def test_entry_exit_and_hold():
    rows = [
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [0, 0],
    ]
    expected = [0, 1, 1, 2, 2, 0, 0, 1, 2, 0]
    assert [x["o0"] for x in execute("entry_exit_fsm", rows)] == expected


def test_history_restores_previous_and_tracks_source():
    rows = [[0, 0], [3, 0], [1, 1], [2, 0], [0, 1], [0, 0], [1, 1]]
    assert [x["o0"] for x in execute("history_state", rows)] == [0, 3, 3, 2, 2, 0, 0]


def test_proposal_cancel_has_priority():
    rows = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert [x["o0"] for x in execute("proposal_fsm", rows)] == [0, 1, 2, 3, 3, 4]


def test_tcp_reset_and_active_close():
    rows = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
    ]
    assert [x["o0"] for x in execute("tcp_connection_fsm", rows)] == [
        0,
        2,
        4,
        5,
        6,
        9,
        0,
    ]


def test_mode_transitions_hold_and_simultaneous_signals():
    rows = [[0, 0], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]]
    assert [o["o0"] for o in execute("mode_switch", rows)] == [0, 1, 1, 0, 1, 0]


def test_risk_normal_priority_and_guarded_escalation():
    rows = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]]
    assert [o["o0"] for o in execute("risk_fsm", rows)] == [0, 1, 2, 0, 0, 1, 2]


def test_streak_reset_priority_and_wrap():
    rows = [[0, 0]] + [[1, 0]] * 17 + [[1, 1], [0, 0], [1, 0]]
    state = 0
    for t, o in enumerate(execute("streak_counter", rows)):
        if t:
            state = 0 if rows[t][1] else (state + rows[t][0]) % 16
        assert o["o0"] == state


def test_boolean_weighted_threshold_all_inputs():
    schema = AgentSchema(
        name="threshold",
        strategy="custom",
        streams=tuple(StreamConfig(n, "sbf") for n in ("a", "b", "c", "decision")),
        logic_blocks=(
            LogicBlock(
                "weighted_vote",
                ("a", "b", "c"),
                "decision",
                {"weights": [3, 2, 1], "threshold": 4},
            ),
        ),
    )
    from dataclasses import replace

    schema = replace(
        schema,
        streams=schema.streams[:-1]
        + (StreamConfig("decision", "sbf", is_input=False),),
    )
    rows = list(itertools.product(range(2), repeat=3))
    text = generate_tau_spec(schema)
    for r, o in zip(rows, execute("threshold", rows, text)):
        assert o["o0"] == int(sum(a * b for a, b in zip(r, [3, 2, 1])) >= 4)


def test_numeric_weighted_vote_keeps_modular_arithmetic():
    schema = AgentSchema(
        name="numeric",
        strategy="custom",
        streams=(
            StreamConfig("a", "bv", 8),
            StreamConfig("b", "bv", 8),
            StreamConfig("result", "bv", 8, is_input=False),
        ),
        logic_blocks=(
            LogicBlock("weighted_vote", ("a", "b"), "result", {"weights": [3, 2]}),
        ),
    )
    rows = list(itertools.product([0, 1, 127, 255], repeat=2))
    for r, o in zip(rows, execute("numeric", rows, generate_tau_spec(schema))):
        assert o["o0"] == (3 * r[0] + 2 * r[1]) % 256


def test_remaining_time_is_explicitly_modular():
    schema = AgentSchema(
        name="remaining",
        strategy="custom",
        streams=tuple(StreamConfig(n, "bv", 16) for n in ("start", "duration", "now"))
        + (StreamConfig("remaining", "bv", 16, is_input=False),),
        logic_blocks=(
            LogicBlock("time_lock", ("start", "duration", "now"), "remaining"),
        ),
    )
    rows = [[100, 20, n] for n in [119, 120, 121]] + [[65535, 2, n] for n in [0, 1, 2]]
    for r, o in zip(rows, execute("remaining", rows, generate_tau_spec(schema))):
        assert o["o0"] == (r[0] + r[1] - r[2]) % 65536


def test_enabled_mirrors_are_equal_to_inputs():
    s = AgentSchema(
        name="mirror",
        strategy="custom",
        streams=(
            StreamConfig("a", "sbf"),
            StreamConfig("result", "sbf", is_input=False),
        ),
        logic_blocks=(LogicBlock("passthrough", ("a",), "result"),),
        include_mirrors=True,
    )
    rows = [[0], [1], [1], [0]]
    for r, o in zip(rows, execute("mirror", rows, generate_tau_spec(s))):
        assert o["mi0"] == o["o0"] == r[0]


def test_lfsr_cycle_and_native_trace():
    # Independent integer enumeration establishes the internal nonzero cycle.
    def advance(state):
        return (state >> 1) ^ (0xB400 if state & 1 else 0)

    seen = set()
    state = 1
    while state not in seen:
        seen.add(state)
        state = advance(state)
    assert state == 1 and len(seen) == 65535 and 0 not in seen
    path = Path(__file__).parents[4] / "specification/prng_module.tau"
    for seed in (0, 1, 65535):
        tau.reset()
        tau.set_severity("error")
        tau.set_charvar(False)
        interpreter = tau.get_interpreter(path.read_text())
        assert interpreter, interpreter.report.errors
        state = 0
        for step in range(85):
            if step:
                state = (seed or 1) if state == 0 else advance(state)
            names = tau.get_inputs_for_step(interpreter.value).value
            result = tau.step(interpreter.value, {name: str(seed) for name in names})
            assert result, result.report.errors
            values = {name.name: int(value) for name, value in result.value.items()}
            assert values == {"o_lfsr": state, "o_rand": state ^ seed if step else 0}


@pytest.mark.parametrize("forecast", [False, True])
def test_trading_transition_equations_on_native_traces(forecast):
    name = "agent4_testnet_v39a.tau" if forecast else "agent4_testnet_v40_core.tau"
    path = Path(__file__).parents[4] / "specification" / name
    text = re.sub(r'(in|out) file\("[^"\n]+"\)', r"\1 console", path.read_text())
    tau.reset()
    tau.set_charvar(False)
    tau.set_severity("error")
    interpreter = tau.get_interpreter(text)
    assert interpreter, interpreter.report.errors
    rng = random.Random(1071)
    previous = None
    for _ in range(128):
        inputs = [rng.randrange(2) for _ in range(7)]
        names = tau.get_inputs_for_step(interpreter.value).value
        result = tau.step(
            interpreter.value, {n: str(inputs[int(n.name[1:])]) for n in names}
        )
        assert result, result.report.errors
        actual = {n.name: int(v) for n, v in result.value.items()}
        if previous is not None:
            p = previous
            timeout = p["o7"] and p["o6"]
            entry = (
                not p["o0"]
                and not inputs[0]
                and inputs[1]
                and inputs[2]
                and not p["o1"]
                and not timeout
                and not p["o9"]
                and not inputs[4]
            )
            if forecast:
                entry = entry and inputs[5] and not inputs[6]
            state = entry or (
                p["o0"]
                and not (inputs[0] and p["o1"])
                and not timeout
                and inputs[1]
                and not inputs[4]
            )
            buy = state and not p["o0"] and not p["o1"]
            sell = p["o0"] and not state and p["o1"]
            profit = sell and inputs[0] and not p["o10"] and inputs[3]
            expected = {
                "o0": state,
                "o1": buy or (not sell and p["o1"]),
                "o2": buy,
                "o3": sell,
                "o6": state and not p["o6"],
                "o7": state and (p["o7"] != p["o6"]),
                "o9": buy or (p["o0"] and not sell and p["o9"]),
                "o10": (buy and inputs[0])
                or (p["o0"] and not buy and not sell and p["o10"]),
                "o11": profit,
                "o13": p["o13"] or profit,
            }
            assert actual == {key: int(value) for key, value in expected.items()}
        # The historical examples do not specify initial state. Check transitions
        # from the actual first state without claiming initialization or returns.
        previous = actual
