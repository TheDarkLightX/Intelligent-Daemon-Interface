"""Freeze a selected checkpoint and export compact, inspectable browser assets."""

import argparse, gzip, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
from .corpus import dataset
from .train import encode, scores
from .graph import from_layers, fold, evaluate


def write_knowledge(collection, out):
    """Keep each raw fallback asset small; preserve the canonical gzip bytes."""
    raw = (json.dumps(collection, separators=(",", ":")) + "\n").encode()
    (out / "knowledge.json.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    names = []
    for start in range(0, len(collection["records"]), 3000):
        name = f"knowledge-{len(names):02d}.json"
        (out / name).write_text(json.dumps(collection["records"][start:start+3000], separators=(",", ":")) + "\n")
        names.append(name)
    manifest = {k:v for k,v in collection.items() if k != "records"}
    manifest["shards"] = names
    (out / "knowledge-manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return names + ["knowledge-manifest.json", "knowledge.json.gz"], dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(checkpoint, relevance, out, final=False, data=None):
    root = Path(__file__).parent
    model = json.loads(checkpoint.read_text())
    rel = json.loads(relevance.read_text())
    cs, ds = dataset()
    start = time.monotonic()
    data = data or root / "data/combined"
    g = fold(from_layers(model))
    rg = fold(from_layers(rel))
    rng = np.random.default_rng(20261104)
    checks = 0
    metrics = {}
    for split in ["validation"] + (["test", "challenge"] if final else []):
        x, y = encode(ds[split], model["features"])
        a = scores(model, x)
        b = evaluate(g, x)
        assert np.array_equal(a, b), split
        checks += len(x)
        metrics[split] = dict(
            rows=len(y),
            correct=int((a.argmax(1) == y).sum()),
            accuracy=float((a.argmax(1) == y).mean()),
        )
    random = rng.integers(0, 2, (256, model["inputs"]), dtype=np.uint8)
    assert np.array_equal(scores(model, random), evaluate(g, random))
    checks += 256
    rx = rng.integers(0, 2, (256, rel["inputs"]), dtype=np.uint8)
    assert np.array_equal(scores(rel, rx), evaluate(rg, rx))
    checks += 256
    # Confidence cutoff is fixed from validation, selecting maximal coverage at
    # >=95% precision among accepted, non-unknown topic predictions.
    vx, vy = encode(ds["validation"], model["features"])
    vs = scores(model, vx)
    order = np.sort(vs, axis=1)
    margins = order[:, -1] - order[:, -2]
    pred = vs.argmax(1)
    threshold = int(margins.max()) + 1
    for cutoff in range(1, int(margins.max()) + 1):
        accept = (margins >= cutoff) & (pred != model["ids"].index("unknown"))
        if accept.any() and (pred[accept] == vy[accept]).mean() >= 0.95:
            threshold = cutoff
            break
    accept = (margins >= threshold) & (pred != model["ids"].index("unknown"))
    metrics["validation_selective"] = dict(
        margin=threshold,
        accepted=int(accept.sum()),
        coverage=float(accept.mean()),
        accuracy=float((pred[accept] == vy[accept]).mean()) if accept.any() else None,
    )
    if final:
        for split in ("test", "challenge"):
            sx, sy = encode(ds[split], model["features"])
            ss = scores(model, sx)
            ordered = np.sort(ss, axis=1)
            sp = ss.argmax(1)
            selected = (ordered[:, -1] - ordered[:, -2] >= threshold) & (
                sp != model["ids"].index("unknown")
            )
            metrics[split + "_selective"] = dict(
                margin=threshold,
                accepted=int(selected.sum()),
                coverage=float(selected.mean()),
                accuracy=(
                    float((sp[selected] == sy[selected]).mean())
                    if selected.any()
                    else None
                ),
            )
    meta = dict(
        trainedTopicTables=sum(len(l["f"]) for l in model["layers"]),
        topicLayers=len(model["layers"]),
        integerReadoutOffsets=len(model.get("readout_bias", [])),
        executedTopicTables=len(g["nodes"]),
        trainedRelevanceTables=sum(len(l["f"]) for l in rel["layers"]),
        executedRelevanceTables=len(rg["nodes"]),
        inputBits=model["inputs"],
        classes=model["classes"],
        checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        selectedEpoch=model["training"]["selected_epoch"],
        parityInputVectors=checks,
        provisional=not final,
    )
    bundle = dict(
        schema="idi/browser-layered-agent/1",
        features=model["features"],
        ids=model["ids"],
        topic=g,
        relevance=rg,
        thresholds={"topicMargin": threshold},
        metadata=meta,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(bundle, separators=(",", ":")) + "\n")
    (out / "model.json.gz").write_bytes(
        gzip.compress((out / "model.json").read_bytes(), compresslevel=9, mtime=0)
    )
    shutil.copyfile(root / "data/cards.json", out / "cards.json")
    collection = json.loads((data / "knowledge.json").read_text())
    markers = ("/home/", "/Users/", "C:\\Users", "/workspace/", "/root/")
    omitted = [r["id"] for r in collection["records"] if any(
        marker in "\n".join(str(r.get(k, "")) for k in ("q", "a", "context"))
        for marker in markers
    )]
    collection["records"] = [r for r in collection["records"] if r["id"] not in omitted]
    collection["export_filter"] = dict(
        omitted_ids=omitted,
        reason="Public-page export omits answers containing user-specific filesystem path examples. Training inputs remain unchanged."
    )
    knowledge_assets, canonical_knowledge = write_knowledge(collection, out)
    shutil.copyfile(root / "data/openassistant/LICENSE", out / "OPENASSISTANT-LICENSE")
    fixtures = []
    for row in ds["validation"][::13]:
        x, _ = encode([row], model["features"])
        fixtures.append(
            dict(
                text=row["text"],
                bits=x[0].astype(int).tolist(),
                scores=scores(model, x)[0].tolist(),
            )
        )
    from .openassistant import relation

    knowledge = collection
    pairs = [
        dict(query=r["q"], record=r, bits=relation(r["q"], r))
        for r in knowledge["records"][:20]
    ]
    (root / "tests/parity-fixtures.json").write_text(
        json.dumps(dict(topic=fixtures, relevance=pairs), separators=(",", ":")) + "\n"
    )
    report = dict(
        metadata=meta,
        metrics=metrics,
        retrieval_records=len(collection["records"]),
        retrieval_export_filter=collection["export_filter"],
        scope="Topic wrapper tests share topic aliases with training. Challenge descriptions do not. Parity checks every class vote for the listed input vectors, not the entire input domain.",
        canonicalKnowledge=canonical_knowledge,
        assets={
            n: dict(
                bytes=(out / n).stat().st_size,
                gzip_bytes=len(gzip.compress((out / n).read_bytes(), mtime=0)),
                sha256=hashlib.sha256((out / n).read_bytes()).hexdigest(),
            )
            for n in ["model.json", "model.json.gz", "cards.json", "evaluator.wasm"] + knowledge_assets
        },
        seconds=round(time.monotonic() - start, 3),
    )
    if final:
        from .train import scores as score

        ar = np.load(data / "test-pairs.npz")
        p = score(rel, ar["x"]).argmax(1)
        y = ar["y"]
        report["relevance_pair_test"] = dict(
            rows=len(y),
            balanced_accuracy=float(
                ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()) / 2
            ),
            positive_recall=float((p[y == 1] == 1).mean()),
            negative_recall=float((p[y == 0] == 0).mean()),
            scope="Held-out conversation trees / Dolly rows; weak pair labels and lexical perturbations. The search index contains the evaluation documents. Not a factuality or open-chat accuracy score.",
        )
        fixed_rules = {
            "query_coverage_at_least_half": ar["x"][:, 3] == 1,
            "both_question_coverages_at_least_half": (ar["x"][:, 3] == 1) & (ar["x"][:, 11] == 1),
            "identical_normalized_question_tokens": ar["x"][:, 63] == 1,
        }
        report["relevance_fixed_rule_baselines"] = {
            name: dict(balanced_accuracy=float(((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2))
            for name, pred in fixed_rules.items()
        }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--relevance", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    run(a.checkpoint, a.relevance, a.out, a.final, a.data)
