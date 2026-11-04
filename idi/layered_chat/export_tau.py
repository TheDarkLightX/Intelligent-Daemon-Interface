"""Export the exact Boolean DAG as original Tau 0.8 fragments, without Tau code.

Each fragment takes external signal bits and returns its internal node bits.
The manifest defines graph wiring and integer class-vote addition in the host.
"""

import argparse, gzip, hashlib, json, random, time
from pathlib import Path
from .qualify_tau import term


def fragments(graph, size=32):
    n = graph["inputs"]
    result = []
    for start in range(0, len(graph["nodes"]), size):
        end = min(len(graph["nodes"]), start + size)
        base = n + start
        inputs = sorted(
            {v for node in graph["nodes"][start:end] for v in node[:2] if v < base}
        )
        names = {v: f"i{i}" for i, v in enumerate(inputs)}
        names.update({n + j: f"o{j-start}" for j in range(start, end)})
        ref = lambda v: "(" + names[v] + "[t]:sbf)"
        equations = [
            f"o{k}[t]:sbf = ({term(f,ref(a),ref(b))})"
            for k, (a, b, f) in enumerate(graph["nodes"][start:end])
        ]
        result.append(
            dict(
                start=start,
                inputs=inputs,
                count=end - start,
                spec="always (\n" + " &&\n".join(equations) + "\n).",
            )
        )
    return result


def run(bundle_path, out, qualify=False):
    out.mkdir(parents=True, exist_ok=False)
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    g = bundle["topic"]
    chunks = fragments(g)
    manifest = dict(
        schema="idi/tau-layered-graph/1",
        bundle_sha256=hashlib.sha256(raw).hexdigest(),
        table_order="msb-00-01-10-11",
        input_bits=g["inputs"],
        classes=g["classes"],
        votes=g["votes"],
        bias=g["bias"],
        fragments=chunks,
        scope="Tau fragments compute Boolean node values. The host supplies feature bits, schedules fragments and sums integer votes. These specifications are independent work; no Tau implementation is included.",
    )
    (out / "fragments.json.gz").write_bytes(
        gzip.compress(json.dumps(manifest, separators=(",", ":")).encode(), mtime=0)
    )
    if qualify:
        import tau

        start = time.monotonic()
        rng = random.Random(20261104)
        indices = sorted(
            {0, len(chunks) - 1, *rng.sample(range(len(chunks)), min(30, len(chunks)))}
        )
        cases = []
        for k in indices:
            chunk = chunks[k]
            tau.reset()
            tau.set_charvar(False)
            tau.set_severity("error")
            r = tau.get_interpreter(chunk["spec"])
            assert r, r.report.errors
            for trial in range(2):
                signals = [rng.randrange(2) for _ in range(g["inputs"])]
                for a, b, f in g["nodes"]:
                    signals.append((f >> (3 - 2 * signals[a] - signals[b])) & 1)
                names = tau.get_inputs_for_step(r.value).value
                z = tau.step(
                    r.value,
                    {n: str(signals[chunk["inputs"][int(n.name[1:])]]) for n in names},
                )
                assert z, z.report.errors
                outputs = {n.name: int(v) for n, v in z.value.items()}
                for j in range(chunk["count"]):
                    assert outputs[f"o{j}"] == signals[g["inputs"] + chunk["start"] + j]
                cases.append(dict(fragment=k, trial=trial, node_values=chunk["count"]))
        report = dict(
            tau_commit="95a7d261ca3db5ff6f7c198a0b4d81c880816058",
            bundle_sha256=manifest["bundle_sha256"],
            fragments=len(chunks),
            fragments_executed=len(indices),
            node_values_checked=sum(c["node_values"] for c in cases),
            seconds=round(time.monotonic() - start, 3),
            cases=cases,
            scope="Native Tau execution of sampled literal graph fragments, two deterministic random input vectors each. Primitive gate and folding certificates are separate evidence; this sample is not exhaustive.",
        )
        (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    run(a.bundle, a.out, a.qualify)
