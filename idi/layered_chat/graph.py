"""Exact Boolean DAG folding, with explicit output polarity and constant bias."""

from __future__ import annotations
import numpy as np


def apply(f, a, b):
    return (f >> (3 - 2 * a - b)) & 1


def from_layers(model):
    previous = list(range(model["inputs"]))
    nodes = []
    for layer in model["layers"]:
        current = []
        for a, b, f in zip(layer["a"], layer["b"], layer["f"]):
            current.append(model["inputs"] + len(nodes))
            nodes.append([previous[a], previous[b], f])
        previous = current
    group = len(previous) // model["classes"]
    return dict(
        inputs=model["inputs"],
        classes=model["classes"],
        nodes=nodes,
        votes=[
            [[v, 0] for v in previous[c * group : (c + 1) * group]]
            for c in range(model["classes"])
        ],
        bias=list(model.get("readout_bias", [0] * model["classes"])),
    )


def fold(graph, certificate=None):
    """Substitute finite functions. A literal is (signal, negation), or (None, bit)."""
    n = graph["inputs"]
    resolved = [(i, 0) for i in range(n)]
    nodes = []
    memo = {}
    for ia, ib, f in graph["nodes"]:
        a, b = resolved[ia], resolved[ib]
        if a[0] is None or b[0] is None or a[0] == b[0]:
            variable = a[0] if a[0] is not None else b[0]
            bits = [
                apply(
                    f,
                    a[1] if a[0] is None else x ^ a[1],
                    b[1] if b[0] is None else x ^ b[1],
                )
                for x in (0, 1)
            ]
            resolved.append(
                (None, bits[0]) if bits[0] == bits[1] else (variable, bits[0])
            )
            continue
        values = [apply(f, x ^ a[1], y ^ b[1]) for x in (0, 1) for y in (0, 1)]
        g = sum(v << (3 - i) for i, v in enumerate(values))
        if g in (0, 15):
            resolved.append((None, int(g == 15)))
            continue
        if g in (3, 12):
            resolved.append((a[0], int(g == 12)))
            continue
        if g in (5, 10):
            resolved.append((b[0], int(g == 10)))
            continue
        aa, bb = a[0], b[0]
        if aa > bb:
            aa, bb = bb, aa
            g = (g & 9) | ((g & 4) >> 1) | ((g & 2) << 1)
        key = (aa, bb, g)
        if key not in memo:
            memo[key] = n + len(nodes)
            nodes.append(list(key))
        resolved.append((memo[key], 0))
    votes = []
    bias = []
    for group, base in zip(graph["votes"], graph["bias"]):
        out = []
        constant = base
        for signal, neg in group:
            var, pol = resolved[signal]
            if var is None:
                constant += pol ^ neg
            else:
                out.append([var, pol ^ neg])
        votes.append(out)
        bias.append(constant)
    live = set()
    stack = [v for group in votes for v, _ in group if v >= n]
    while stack:
        v = stack.pop()
        if v in live:
            continue
        live.add(v)
        a, b, _ = nodes[v - n]
        stack.extend(x for x in (a, b) if x >= n)
    remap = {i: i for i in range(n)}
    kept = []
    for i, (a, b, f) in enumerate(nodes, n):
        if i in live:
            remap[i] = n + len(kept)
            kept.append([remap[a], remap[b], f])
    result = dict(
        inputs=n,
        classes=graph["classes"],
        nodes=kept,
        votes=[[[remap[v], p] for v, p in group] for group in votes],
        bias=bias,
    )
    if certificate is not None:
        certificate.update(
            schema="idi/boolean-fold-certificate/1",
            mapping=resolved,
            nodes=nodes,
            votes=votes,
            bias=bias,
        )
    return result


def evaluate(graph, inputs):
    # A separate topological evaluator; no training code is called.
    x = np.asarray(inputs, dtype=np.uint8)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] != graph["inputs"] or np.any(x > 1):
        raise ValueError("Boolean input width mismatch")
    bits = np.empty((x.shape[0], graph["inputs"] + len(graph["nodes"])), np.uint8)
    bits[:, : graph["inputs"]] = x
    for j, (a, b, f) in enumerate(graph["nodes"], graph["inputs"]):
        bits[:, j] = (f >> (3 - 2 * bits[:, a] - bits[:, b])) & 1
    scores = np.zeros((len(x), graph["classes"]), np.int32)
    for c, group in enumerate(graph["votes"]):
        scores[:, c] = graph["bias"][c]
        for v, neg in group:
            scores[:, c] += bits[:, v] ^ neg
    return scores
