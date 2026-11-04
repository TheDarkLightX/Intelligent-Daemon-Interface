"""Replay Boolean folding without the folder, trainer, NumPy or an SMT solver.

Induction on original node order proves every recorded substitution. Each local
step enumerates at most four independent signal bits, and output vote multisets
must agree exactly. Dead-node removal is checked as a separate graph projection.
"""

from collections import Counter
from itertools import product
import argparse, gzip, hashlib, json, time
from pathlib import Path


class Refused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Refused(message)


def gate(f, a, b):
    return tuple(int(v) for v in format(f, "04b"))[2 * a + b]


def replay(original, folded, certificate):
    n = original["inputs"]
    mapping = certificate["mapping"]
    nodes = certificate["nodes"]
    require(len(mapping) == n + len(original["nodes"]), "mapping length")
    require(
        mapping[:n] == [[i, 0] for i in range(n)]
        or mapping[:n] == [(i, 0) for i in range(n)],
        "input identity",
    )
    for j, node in enumerate(nodes):
        require(
            len(node) == 3 and all(type(x) is int for x in node), "node representation"
        )
        a, b, f = node
        require(0 <= a < n + j and 0 <= b < n + j and 0 <= f < 16, "candidate topology")
    for v, p in mapping:
        require(
            p in (0, 1) and (v is None or type(v) is int and 0 <= v < n + len(nodes)),
            "literal representation",
        )
    checked = 0
    for i, (a, b, f) in enumerate(original["nodes"], n):
        require(0 <= a < i and 0 <= b < i and 0 <= f < 16, "original topology")
        left, right = mapping[a], mapping[b]
        candidate = mapping[i]
        atoms = {v for v, _ in (left, right) if v is not None}
        leaves = set(atoms)
        v, pol = candidate
        if v is not None:
            if v < n or v in atoms:
                leaves.add(v)
            else:
                leaves.update(nodes[v - n][:2])
        require(len(leaves) <= 4, "local support exceeds four bits")
        leaves = sorted(leaves)
        for values in product(range(2), repeat=len(leaves)):
            env = dict(zip(leaves, values))
            read = lambda lit: lit[1] if lit[0] is None else env[lit[0]] ^ lit[1]
            expected = gate(f, read(left), read(right))
            if v is None:
                actual = pol
            elif v < n or v in atoms:
                actual = env[v] ^ pol
            else:
                aa, bb, ff = nodes[v - n]
                actual = gate(ff, env[aa], env[bb]) ^ pol
            require(actual == expected, f"local substitution at original node {i}")
            checked += 1
    cv = certificate["votes"]
    cb = certificate["bias"]
    require(len(cv) == len(cb) == original["classes"], "class count")
    for c, group in enumerate(original["votes"]):
        counts = Counter()
        base = original["bias"][c]
        for source, negation in group:
            v, p = mapping[source]
            if v is None:
                base += p ^ negation
            else:
                counts[(v, p ^ negation)] += 1
        require(
            base == cb[c] and counts == Counter(tuple(v) for v in cv[c]),
            f"vote multiset {c}",
        )
    live = set()
    todo = [v for group in cv for v, _ in group if v >= n]
    while todo:
        v = todo.pop()
        if v in live:
            continue
        live.add(v)
        todo.extend(w for w in nodes[v - n][:2] if w >= n)
    remap = {i: i for i in range(n)}
    projected = []
    for i, node in enumerate(nodes, n):
        if i in live:
            a, b, f = node
            remap[i] = n + len(projected)
            projected.append([remap[a], remap[b], f])
    expected = dict(
        inputs=n,
        classes=original["classes"],
        nodes=projected,
        votes=[[[remap[v], p] for v, p in group] for group in cv],
        bias=cb,
    )
    require(expected == folded, "dead-node projection")
    return dict(
        original_nodes=len(original["nodes"]),
        folded_nodes=len(projected),
        local_boolean_valuations=checked,
        classes=original["classes"],
        scope="All Boolean input vectors, by local substitution induction and exact vote multisets. Trusts this Python checker and its execution; no proof-assistant kernel claim.",
    )


def run(original_path, folded_path, certificate_path):
    read = lambda p: json.loads(
        gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    )
    print(
        json.dumps(
            replay(read(original_path), read(folded_path), read(certificate_path)),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("original", type=Path)
    p.add_argument("folded", type=Path)
    p.add_argument("certificate", type=Path)
    a = p.parse_args()
    run(a.original, a.folded, a.certificate)
