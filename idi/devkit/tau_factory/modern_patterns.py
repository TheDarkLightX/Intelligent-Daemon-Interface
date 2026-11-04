"""Tau 0.8 guarded equations for previously malformed conditional patterns.

State transitions use ordered, disjoint guards and an explicit hold fallback.
Bitvector arithmetic retains modular semantics; no sentinel is truncated into
an actual state. This module emits constraints and performs no effects.
"""

from dataclasses import replace
import re

SPECIAL = {
    "weighted_vote",
    "time_lock",
    "multi_bit_counter",
    "streak_counter",
    "mode_switch",
    "proposal_fsm",
    "risk_fsm",
    "entry_exit_fsm",
    "orthogonal_regions",
    "tcp_connection_fsm",
    "history_state",
}


def formula(schema, block):
    streams = {s.name: s for s in schema.streams}
    names = {}
    for inp in (True, False):
        for i, s in enumerate(s for s in schema.streams if s.is_input == inp):
            names[s.name] = (
                s.name.replace("-", "_")
                if schema.descriptive_names
                else ("i" if inp else "o") + format(i, "X")
            )

    def ref(name, t="t"):
        if name not in names:
            raise ValueError(f"Undeclared stream: {name}")
        return f"{names[name]}[{t}]"

    def bit(name):
        return f"{ref(name)} != 0" if name else "F"

    def const(n, name=block.output):
        s = streams[name]
        if s.stream_type == "sbf":
            if n not in (0, 1):
                raise ValueError("Boolean output cannot encode this state")
            return str(n)
        if not 0 <= n < (1 << s.width):
            raise ValueError(f"State {n} is outside bv[{s.width}]")
        return f"{{{n}}}:bv[{s.width}]"

    def state(n, name=block.output):
        return f'{ref(name,"t-1")} = {const(n,name)}'

    def choose(out, cases, default, initial=None):
        clauses = []
        earlier = []
        for condition, value in cases:
            guard = " && ".join([f"!({c})" for c in earlier] + [f"({condition})"])
            clauses.append(f"(({guard}) -> {ref(out)} = {value})")
            earlier.append(condition)
        guard = " && ".join(f"!({c})" for c in earlier) or "T"
        clauses.append(f"(({guard}) -> {ref(out)} = {default})")
        if initial is not None:
            clauses.append(f'({ref(out,"0")} = {const(initial,out)})')
        return " && ".join(clauses)

    p = block.params
    ins = block.inputs
    out = block.output
    previous = ref(out, "t-1")

    def inp(key, i):
        return p.get(key, ins[i] if i < len(ins) else None)

    if block.pattern == "weighted_vote":
        weights = p.get("weights", [1] * len(ins))
        if len(weights) != len(ins) or len(ins) > 12:
            raise ValueError(
                "Weighted vote needs one weight per input and at most 12 inputs"
            )
        s = streams[out]
        if any(type(w) != int or w < 0 for w in weights):
            raise ValueError("Weights must be nonnegative integers")
        threshold = p.get("threshold", sum(weights) // 2 + 1)
        if type(threshold) != int or threshold < 0:
            raise ValueError("Threshold must be a nonnegative integer")
        if any(streams[n].stream_type == "bv" for n in ins):
            widths = {streams[n].width for n in ins if streams[n].stream_type == "bv"}
            if len(widths) != 1 or any(streams[n].stream_type != "bv" for n in ins):
                raise ValueError("Numeric votes require equal bitvector input widths")
            width = next(iter(widths))
            if s.stream_type == "bv" and s.width != width:
                raise ValueError("Weighted sum output width must match the inputs")
            if any(w >= (1 << width) for w in weights) or threshold >= (1 << width):
                raise ValueError("Weights and threshold must fit the declared width")
            total = " + ".join(
                f"({ref(n)} * {{{w}}}:bv[{width}])" for n, w in zip(ins, weights)
            )
            return (
                f"(({ref(out)} != 0) <-> (({total}) >= {{{threshold}}}:bv[{width}]))"
                if s.stream_type == "sbf"
                else f"{ref(out)} = ({total})"
            )
        # Enumerated guards make all combinations explicit and mutually exclusive.
        rows = []
        for mask in range(1 << len(ins)):
            cond = " && ".join(
                f'({ref(n)} {"!=" if mask>>i&1 else "="} 0)' for i, n in enumerate(ins)
            )
            total = sum(w for i, w in enumerate(weights) if mask >> i & 1)
            value = (
                int(total >= threshold)
                if s.stream_type == "sbf"
                else total % (1 << s.width)
            )
            rows.append(f"(({cond}) -> {ref(out)} = {const(value)})")
        return " && ".join(rows)
    if block.pattern == "time_lock":
        a, b, c = inp("lock_start", 0), inp("lock_duration", 1), inp("current_time", 2)
        if (
            not all(n in streams and streams[n].stream_type == "bv" for n in (a, b, c))
            or len({streams[n].width for n in (a, b, c)}) != 1
        ):
            raise ValueError("Time lock requires equal-width bitvector inputs")
        if streams[out].stream_type == "bv":
            if streams[out].width != streams[a].width:
                raise ValueError("Time lock output width must match the inputs")
            return f"{ref(out)} = (({ref(a)} + {ref(b)}) - {ref(c)})"
        # End-time arithmetic is modular in the declared width, as before.
        return f"(({ref(out)} != 0) <-> ({ref(c)} < ({ref(a)} + {ref(b)})))"
    if block.pattern in ("multi_bit_counter", "streak_counter"):
        increment = ins[0]
        reset = p.get("reset_input", ins[1] if len(ins) > 1 else None)
        cases = []
        if block.pattern == "streak_counter":
            reset = p.get("opposite_event", reset)
        if reset:
            cases.append((bit(reset), const(0)))
        cases.append((bit(increment), f"({previous} + {const(1)})"))
        return choose(out, cases, previous, p.get("initial_value", 0))
    if block.pattern == "mode_switch":
        modes = p.get("modes", ["DEFENSIVE", "AGGRESSIVE"])
        transitions = p.get("transitions", {})
        cases = []
        if transitions:
            for mode, transition in transitions.items():
                if mode not in modes or transition["to"] not in modes:
                    raise ValueError("Unknown mode in transition")
                cases.append(
                    (
                        f'({state(modes.index(mode))}) && ({bit(transition["on"])})',
                        const(modes.index(transition["to"])),
                    )
                )
        else:
            cases = [(bit(ins[0]), const(1))]
        return choose(out, cases, previous, p.get("initial_mode", 0))
    if block.pattern == "proposal_fsm":
        create, vote, execute, cancel = [
            inp(k, i)
            for i, k in enumerate(
                ("create_input", "vote_input", "execute_input", "cancel_input")
            )
        ]
        cases = []
        if cancel:
            cases.append((bit(cancel), const(4)))
        cases.append((f"({state(0)}) && ({bit(create)})", const(1)))
        if vote:
            condition = f"({state(1)}) && ({bit(vote)})"
            if p.get("quorum_met"):
                condition += f' && ({bit(p["quorum_met"])})'
            cases.append((condition, const(2)))
        if execute:
            cases.append((f"({state(2)}) && ({bit(execute)})", const(3)))
        return choose(out, cases, previous, 0)
    if block.pattern == "risk_fsm":
        warning, critical, normal = [
            inp(k, i)
            for i, k in enumerate(
                ("warning_signal", "critical_signal", "normal_signal")
            )
        ]
        cases = []
        if normal:
            cases.append((bit(normal), const(0)))
        if critical:
            cases.append((f"({state(1)}) && ({bit(critical)})", const(2)))
        cases.append((f"({state(0)}) && ({bit(warning)})", const(1)))
        return choose(out, cases, previous, 0)
    if block.pattern == "entry_exit_fsm":
        out = p.get("phase_output", out)
        previous = ref(out, "t-1")
        entry = inp("entry_signal", 0)
        exits = [
            inp(k, i)
            for i, k in enumerate(
                ("entry_signal", "exit_signal", "stop_loss", "take_profit")
            )
        ][1:]
        exit_cond = " || ".join("(" + bit(x) + ")" for x in exits if x) or "F"
        c = lambda n: const(n, out)
        st = lambda n: state(n, out)
        result = choose(
            out,
            [
                (f"({st(0)}) && ({bit(entry)})", c(1)),
                (f"({st(1)}) && ({exit_cond})", c(2)),
                (f"({st(2)}) && !({bit(entry)})", c(0)),
            ],
            previous,
            0,
        )
        if p.get("position_output"):
            result += (
                f' && (({ref(p["position_output"])} != 0) <-> ({ref(out)} = {c(1)}))'
            )
        return result
    if block.pattern == "orthogonal_regions":
        result = []
        for region, name in zip(p["regions"], p["region_outputs"]):
            rin = region["inputs"]
            if len(rin) == 1:
                result.append(
                    choose(name, [(bit(rin[0]), const(1, name))], const(0, name))
                )
                continue
            cases = [(bit(rin[0]), const(1, name))]
            if len(rin) > 1:
                cases.append((bit(rin[1]), const(0, name)))
            result.append(
                choose(name, cases, ref(name, "t-1"), region.get("initial_state", 0))
            )
        return " && ".join(result)
    if block.pattern == "history_state":
        source = inp("substate_input", 0)
        entry = inp("superstate_entry", 1)
        return choose(
            out,
            [
                (f"({bit(entry)}) && ({previous} != {const(0)})", previous),
                (bit(entry), const(p.get("initial_substate", 0))),
            ],
            ref(source),
            p.get("initial_substate", 0),
        )
    if block.pattern == "tcp_connection_fsm":
        syn, ack, fin, rst = [
            inp(k, i)
            for i, k in enumerate(("syn_flag", "ack_flag", "fin_flag", "rst_flag"))
        ]
        cases = []
        if rst:
            cases.append((bit(rst), const(0)))
        # Minimal finite handshake model. FIN from ESTABLISHED follows the
        # active-close branch; passive close requires a distinct input contract.
        for old, new, signals in [
            (0, 2, [syn]),
            (1, 3, [syn]),
            (2, 4, [syn, ack]),
            (3, 4, [ack]),
            (4, 5, [fin]),
            (5, 6, [ack]),
            (5, 8, [fin]),
            (6, 9, [fin]),
            (7, 10, [fin]),
            (8, 9, [ack]),
            (9, 0, [p.get("timeout_signal")]),
            (10, 0, [ack]),
        ]:
            if all(signals):
                guards = ["(" + state(old) + ")"] + [
                    "(" + bit(n) + ")" for n in signals
                ]
                if rst:
                    guards.append("!(" + bit(rst) + ")")
                if old == 5 and new == 8 and ack:
                    guards.append("!(" + bit(ack) + ")")
                cases.append((" && ".join(guards), const(new)))
        # State guards are disjoint; only ACK/FIN in FIN_WAIT_1 needed priority.
        # Avoid a quadratic expansion of redundant preceding-state negations.
        clauses = [f"(({guard}) -> {ref(out)} = {value})" for guard, value in cases]
        clauses.append(
            f'((!({" || ".join("("+c+")" for c,_ in cases)})) -> {ref(out)} = {previous})'
        )
        clauses.append(f'{ref(out,"0")} = {const(p.get("initial_state",0))}')
        return " && ".join(clauses)
    raise ValueError("Unknown modern pattern")


def modern_formula(schema, legacy_recurrence):
    clauses = []
    for block in schema.logic_blocks:
        if block.pattern in SPECIAL:
            clauses.append(formula(schema, block))
        else:
            lines = legacy_recurrence(
                replace(schema, logic_blocks=(block,)), schema.descriptive_names
            )
            body = next(x[2:] for x in lines if x.startswith("r "))
            # A combinational output's initial value must not constrain an
            # environment input. Preserve explicit state whenever history reads it.
            for s in (s for s in schema.streams if not s.is_input):
                outputs = [v for v in schema.streams if not v.is_input]
                name = (
                    s.name.replace("-", "_")
                    if schema.descriptive_names
                    else "o" + format(outputs.index(s), "X")
                )
                if not re.search(r"\b" + re.escape(name) + r"\[t-\d+\]", body):
                    body = re.sub(
                        r"\s*&&\s*\(" + re.escape(name) + r"\[0\]\s*=\s*[^)]+\)",
                        "",
                        body,
                    )
            clauses.append(body)
    if schema.include_mirrors:
        for i, s in enumerate(s for s in schema.streams if s.is_input):
            name = (
                s.name.replace("-", "_")
                if schema.descriptive_names
                else "i" + format(i, "X")
            )
            mirror = (
                (s.name + "_mirror").replace("-", "_")
                if schema.descriptive_names
                else "mi" + format(i, "X")
            )
            clauses.append(f"{mirror}[t] = {name}[t]")
    return " && ".join("(" + c + ")" for c in clauses)
