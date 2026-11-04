"""SMT equivalence of original/folded votes for every Boolean input vector.

Each class is a separate QF_LIA query over shared Boolean inputs. Solver answers
and query digests are evidence; these are not independently replayed proof terms.
"""

import argparse, gzip, hashlib, json, time
from pathlib import Path
import z3
from .graph import from_layers, fold


def symbolic(graph, inputs):
    bits = list(inputs)
    for a, b, f in graph["nodes"]:
        values = [z3.BoolVal(bool(f >> (3 - i) & 1)) for i in range(4)]
        bits.append(
            z3.simplify(
                z3.If(
                    bits[a],
                    z3.If(bits[b], values[3], values[2]),
                    z3.If(bits[b], values[1], values[0]),
                )
            )
        )
    return [
        z3.simplify(
            base
            + z3.Sum([z3.If(z3.Not(bits[v]) if p else bits[v], 1, 0) for v, p in group])
        )
        for base, group in zip(graph["bias"], graph["votes"])
    ]


def run(checkpoint, out):
    out.mkdir(parents=True, exist_ok=False)
    model = json.loads(checkpoint.read_text())
    g = from_layers(model)
    folded = fold(g)
    start = time.monotonic()
    inputs = [z3.Bool(f"x{i}") for i in range(g["inputs"])]
    a = symbolic(g, inputs)
    b = symbolic(folded, inputs)
    rows = []
    for i, (left, right) in enumerate(zip(a, b)):
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.add(left != right)
        text = solver.to_smt2()
        path = out / f"class-{i:02d}.smt2.gz"
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
        t = time.monotonic()
        verdict = str(solver.check())
        row = dict(
            class_index=i,
            id=model["ids"][i],
            result=verdict,
            seconds=round(time.monotonic() - t, 6),
            query_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        rows.append(row)
        print(row, flush=True)
        if verdict == "sat":
            row["counterexample"] = str(solver.model())
    report = dict(
        schema="idi/full-domain-fold-equivalence/1",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        z3=z3.get_version_string(),
        inputs=g["inputs"],
        original_tables=len(g["nodes"]),
        folded_tables=len(folded["nodes"]),
        all_unsat=all(r["result"] == "unsat" for r in rows),
        results=rows,
        seconds=round(time.monotonic() - start, 3),
        scope="Solver-attested equality of every integer class vote for all Boolean input vectors, for this frozen checkpoint. Does not establish feature encoding correctness, training quality or solver soundness.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.checkpoint, a.out)
