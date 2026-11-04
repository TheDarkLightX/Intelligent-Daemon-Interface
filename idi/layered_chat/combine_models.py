"""Compose independent layered networks by exact, equal-weight class votes."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from .train import scores, encode
from .corpus import dataset


def combine(models):
    first = models[0]
    classes = first["classes"]
    widths = [len(m["layers"][0]["f"]) // classes for m in models]
    depth = len(first["layers"])
    total = sum(widths)
    for m, v in zip(models, widths):
        if (
            m["features"] != first["features"]
            or m["ids"] != first["ids"]
            or len(m["layers"]) != depth
            or any(len(l["f"]) != v * classes for l in m["layers"])
        ):
            raise ValueError(
                "Compatible features, classes, depth and class-group widths are required"
            )
    result = {
        k: first[k]
        for k in ("schema", "table_order", "inputs", "features", "classes", "ids")
    }
    result["layers"] = []
    for li in range(depth):
        layer = dict(a=[], b=[], f=[])
        for c in range(classes):
            offset = 0
            for model, votes in zip(models, widths):
                old = model["layers"][li]
                for j in range(c * votes, (c + 1) * votes):
                    for k in ("a", "b"):
                        parent = old[k][j]
                        layer[k].append(
                            parent
                            if li == 0
                            else (parent // votes) * total + offset + parent % votes
                        )
                    layer["f"].append(old["f"][j])
                offset += votes
        result["layers"].append(layer)
    return result


def run(paths, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    raw = [p.read_bytes() for p in paths]
    models = [json.loads(r) for r in raw]
    model = combine(models)
    _, ds = dataset()
    x, y = encode(ds["validation"], model["features"])
    actual = scores(model, x)
    expected = sum(scores(m, x) for m in models)
    assert np.array_equal(actual, expected)
    random = np.random.default_rng(20261104).integers(
        0, 2, (256, model["inputs"]), dtype=np.uint8
    )
    assert np.array_equal(scores(model, random), sum(scores(m, random) for m in models))
    accuracy = float((actual.argmax(1) == y).mean())
    members = [
        dict(
            run=p.parent.name,
            sha256=hashlib.sha256(r).hexdigest(),
            selected_epoch=m["training"]["selected_epoch"],
        )
        for p, r, m in zip(paths, raw, models)
    ]
    model["training"] = dict(
        selected_epoch=None,
        architecture="equal-vote ensemble",
        members=members,
        validation_accuracy=accuracy,
    )
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")) + "\n")
    report = dict(
        architecture="seven independent 15,872-table networks",
        tables=sum(len(l["f"]) for l in model["layers"]),
        best_validation_accuracy=accuracy,
        members=members,
        composition_parity_vectors=len(x) + len(random),
        seconds=round(time.monotonic() - start, 3),
        scope="Validation-only architecture selection. Exact class-score sums checked; no extra fitting after composition.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.models, a.out)
