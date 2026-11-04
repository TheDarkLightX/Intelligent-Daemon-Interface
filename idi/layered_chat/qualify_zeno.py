"""Native ZenoFCIS qualification of folding primitives, with fresh mutants."""

import argparse, hashlib, itertools, json, subprocess
from pathlib import Path
from .graph import fold


def lower(g):
    ns = []
    cache = {}

    def add(n):
        key = tuple(n)
        if key not in cache:
            cache[key] = len(ns)
            ns.append(n)
        return cache[key]

    values = [add(["Input", i]) for i in range(g["inputs"])]
    constant = [add(["Bool", 0]), add(["Bool", 1])]
    for a, b, f in g["nodes"]:
        lo = add(["Select", values[b], constant[(f >> 2) & 1], constant[(f >> 3) & 1]])
        hi = add(["Select", values[b], constant[f & 1], constant[(f >> 1) & 1]])
        values.append(add(["Select", values[a], hi, lo]))
    roots = []
    for group, bias in zip(g["votes"], g["bias"]):
        assert len(group) + bias <= 1
        if not group:
            roots.append(constant[bias])
        else:
            v, p = group[0]
            roots.append(add(["Not", values[v]]) if p else values[v])
    return dict(
        inputs=[{"kind": "Bool"}] * g["inputs"],
        outputs=[{"kind": "Bool"}] * len(roots),
        nodes=ns,
        roots=roots,
    )


def run(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []

    def call(args):
        p = subprocess.run(
            [str(binary), "--no-kb", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            r = json.loads(p.stdout)
        except ValueError:
            r = {"stdout": p.stdout, "stderr": p.stderr}
        return p.returncode, r

    def canonical(program, name):
        path = out / (name + ".json")
        path.write_text(json.dumps(program, indent=2) + "\n")
        dest = out / (name + ".zcve")
        code, r = call(["loop", "encode", "--program", path, "--out", dest])
        assert code == 0, (code, r)
        return dest

    adapters = [None, 12, None, 12, 0, 15]
    for ai, bi in itertools.product(range(6), repeat=2):
        nodes = []

        def literal(i):
            if i == 0:
                return 0
            if i == 2:
                return 1
            nodes.append([0 if i < 2 or i >= 4 else 1, 0, adapters[i]])
            return 2 + len(nodes) - 1

        a, b = literal(ai), literal(bi)
        votes = []
        for f in range(16):
            nodes.append([a, b, f])
            votes.append([[2 + len(nodes) - 1, 0]])
        original = dict(inputs=2, classes=16, nodes=nodes, votes=votes, bias=[0] * 16)
        candidate = fold(original)
        name = f"fold-{ai}-{bi}"
        op = canonical(lower(original), name + "-original")
        cp = canonical(lower(candidate), name + "-candidate")
        receipt = out / (name + "-receipt.json")
        code, result = call(
            [
                "transform",
                "check",
                "--original",
                op,
                "--candidate",
                cp,
                "--receipt",
                receipt,
            ]
        )
        assert code == 0 and result["status"] == "equivalent", result
        code, replay = call(
            [
                "transform",
                "replay",
                "--original",
                op,
                "--candidate",
                cp,
                "--receipt",
                receipt,
            ]
        )
        assert code == 0 and replay["status"] == "replayed", replay
        mutant = lower(candidate)
        mutant["nodes"].append(["Not", mutant["roots"][0]])
        mutant["roots"][0] = len(mutant["nodes"]) - 1
        mp = canonical(mutant, name + "-mutant")
        code, mutation = call(
            ["transform", "check", "--original", op, "--candidate", mp]
        )
        assert code != 0 and mutation.get("status") == "counterexample", mutation
        rows.append(
            dict(
                case=name,
                gates=16,
                tuples=4,
                check=result,
                replay=replay,
                mutation=mutation,
            )
        )
        print(name, flush=True)
    report = dict(
        schema="idi/zeno-fold-qualification/1",
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        version=subprocess.check_output([str(binary), "--version"], text=True).strip(),
        cases=len(rows),
        gate_contexts=len(rows) * 16,
        input_output_observations=len(rows) * 16 * 4,
        mutants_detected=len(rows),
        rows=rows,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.binary.resolve(), a.out.resolve())
