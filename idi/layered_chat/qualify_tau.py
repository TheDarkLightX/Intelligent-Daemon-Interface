"""Native Tau 0.8 gate-table execution and fold-law qualification.

Requires the user's independently obtained tau-nanobind package. The package
and its framework are never copied into an experiment or the public page.
"""

import argparse, hashlib, itertools, json, time
from pathlib import Path
from .graph import fold


def term(f, a, b):
    cells = [
        f"(({a})" + ("'" if not x else "") + f" & ({b})" + ("'" if not y else "") + ")"
        for x in (0, 1)
        for y in (0, 1)
        if (f >> (3 - 2 * x - y)) & 1
    ]
    return "0" if not cells else "1" if len(cells) == 4 else " | ".join(cells)


def expression(g, c):
    values = ["x:sbf", "y:sbf"]
    for a, b, f in g["nodes"]:
        values.append(term(f, values[a], values[b]))
    if not g["votes"][c]:
        return str(g["bias"][c])
    v, p = g["votes"][c][0]
    return "(" + values[v] + ")" + ("'" if p else "")


def run(out):
    import tau

    tau.set_severity("error")
    tau.set_charvar(False)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    bank = (
        "always ("
        + " && ".join(
            f'o{f}[t]:sbf = ({term(f,"i0[t]:sbf","i1[t]:sbf")})' for f in range(16)
        )
        + ")."
    )
    (out / "gate-bank.tau").write_text(bank + "\n")
    r = tau.get_interpreter(bank)
    assert r, r.report.errors
    table = [[None] * 4 for _ in range(16)]
    steps = []
    for a, b in itertools.product(range(2), repeat=2):
        names = tau.get_inputs_for_step(r.value).value
        inputs = {n: str(a if n.name == "i0" else b) for n in names}
        t = time.monotonic()
        result = tau.step(r.value, inputs)
        assert result, result.report.errors
        values = {n.name: v for n, v in result.value.items()}
        for f in range(16):
            table[f][2 * a + b] = int(values[f"o{f}"])
            assert table[f][2 * a + b] == (f >> (3 - 2 * a - b)) & 1
        steps.append(dict(a=a, b=b, outputs=values, seconds=time.monotonic() - t))
    laws = []
    adapters = [None, 12, None, 12, 0, 15]
    for ai, bi in itertools.product(range(6), repeat=2):
        nodes = []

        def literal(i):
            if i == 0:
                return 0
            if i == 2:
                return 1
            nodes.append([0 if i < 2 or i >= 4 else 1, 0, adapters[i]])
            return len(nodes) + 1

        a, b = literal(ai), literal(bi)
        votes = []
        for f in range(16):
            nodes.append([a, b, f])
            votes.append([[len(nodes) + 1, 0]])
        original = dict(inputs=2, classes=16, nodes=nodes, votes=votes, bias=[0] * 16)
        candidate = fold(original)
        for f in range(16):
            question = f"({expression(original,f)}) = ({expression(candidate,f)})."
            verdict = tau.valid(question)
            assert verdict and verdict.value is True, (question, verdict.report.errors)
            laws.append(dict(left=ai, right=bi, gate=f, formula=question, valid=True))
    mutants = []
    for f in range(16):
        question = f'({term(f,"x:sbf","y:sbf")}) = ({term(f^1,"x:sbf","y:sbf")}).'
        v = tau.valid(question)
        assert v and v.value is False
        mutants.append(dict(gate=f, mutant=f ^ 1, formula=question, valid=False))
    report = dict(
        schema="idi/tau-gate-qualification/1",
        tau_source_commit="95a7d261ca3db5ff6f7c198a0b4d81c880816058",
        official_ci_artifact=13293842679,
        artifact_zip_sha256="c58361a499c7d7f0de73d87cfd3bb6e0fce925f0286ff439794a0bc9f6fb5582",
        seconds=round(time.monotonic() - start, 3),
        gate_applications=64,
        fold_laws=len(laws),
        fresh_semantic_mutants=len(mutants),
        steps=steps,
        table=table,
        laws=laws,
        mutants=mutants,
        scope="Tau decides primitive Boolean identities and executes the 16-function gate bank. Graph scheduling and integer vote addition remain independent host operations. No claim about conversational accuracy or native Mealy compilation.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["steps", "table", "laws", "mutants"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.out)
