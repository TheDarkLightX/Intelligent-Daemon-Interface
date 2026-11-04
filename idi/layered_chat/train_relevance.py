"""Fit a small Boolean-table relevance stage on grouped OpenAssistant pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .network import Network
from .train import frozen, scores


def run(data, out):
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
        for n in ("train_relevance.py", "openassistant.py", "network.py", "train.py")
    }
    arrays = {
        s: np.load(data / (s + "-pairs.npz")) for s in ("train", "validation", "test")
    }
    x, y = arrays["train"]["x"], arrays["train"]["y"]
    vx, vy = arrays["validation"]["x"], arrays["validation"]["y"]
    net = Network(64, [1024] * 3, "choice", 2, 20261104)
    rng = np.random.default_rng(20261104)
    history = []
    best = -1
    start = time.monotonic()
    exposures = 0
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    for epoch in range(60):
        order = np.r_[positive, rng.choice(negative, len(positive), replace=False)]
        rng.shuffle(order)
        for s in range(0, len(order), 256):
            ix = order[s : s + 256]
            _, grads = net.loss_and_grads(x[ix], y[ix], 2, 30)
            for layer, grad in zip(net.layers, grads):
                layer.adam(grad, 0.03)
            exposures += len(ix)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model = frozen(
                net,
                [f"overlap-{i}" for i in range(64)],
                [{"id": "other"}, {"id": "paired"}],
            )
            z = scores(model, vx)
            p = z.argmax(1)
            balanced = float(((p[vy == 1] == 1).mean() + (p[vy == 0] == 0).mean()) / 2)
            row = dict(
                epoch=epoch + 1,
                validation_balanced_accuracy=balanced,
                seconds=round(time.monotonic() - start, 2),
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            if balanced > best:
                best = balanced
                model["training"] = dict(
                    epoch=epoch + 1, seed=20261104, validation_balanced_accuracy=best
                )
                (out / "model.json").write_text(
                    json.dumps(model, separators=(",", ":")) + "\n"
                )
    report = dict(
        tables=3072,
        examples_seen=exposures,
        train_pairs=len(y),
        validation_pairs=len(vy),
        test_pairs=len(arrays["test"]["y"]),
        seconds=round(time.monotonic() - start, 3),
        best_validation_balanced_accuracy=best,
        history=history,
        source_sha256=source_hashes,
        pair_sha256={
            split: hashlib.sha256((data / (split + "-pairs.npz")).read_bytes()).hexdigest()
            for split in ("train", "validation", "test")
        },
        numpy_version=np.__version__,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.out)
