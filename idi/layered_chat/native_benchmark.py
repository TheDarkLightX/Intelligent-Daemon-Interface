"""Build checked ZAL gate bytes into a 64-lane native DAG evaluator and measure it."""

import argparse, ctypes, hashlib, json, platform, statistics, subprocess, time
from pathlib import Path
import numpy as np
from .graph import evaluate
from .corpus import dataset
from .train import encode


def run(out, cc="cc"):
    root = Path(__file__).parent
    out.mkdir(parents=True, exist_ok=False)
    assembly = []
    identities = []
    for f in range(16):
        artifact = root / f"evidence/zal-v2/gate-{f:02}-artifact/machine"
        blob = (artifact / "kernel.bin").read_bytes()
        identities.append(
            dict(gate=f, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        )
        assembly.append(
            (artifact / "kernel.S").read_text().replace("zeno_kernel", f"gate_{f}")
        )
    (out / "gates.S").write_text("\n".join(assembly))
    library = out / "liblayered.so"
    cmd = [
        cc,
        "-O3",
        "-fPIC",
        "-shared",
        str(root / "native/evaluator.c"),
        str(out / "gates.S"),
        "-o",
        str(library),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    lib = ctypes.CDLL(str(library.resolve()))
    fn = lib.evaluate_batch
    pointer = ctypes.c_void_p
    u32 = ctypes.c_uint32
    fn.argtypes = [
        pointer,
        u32,
        pointer,
        u32,
        pointer,
        u32,
        pointer,
        pointer,
        u32,
        u32,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    g = json.loads((root / "web/model.json").read_text())["topic"]
    nodes = np.array(g["nodes"], np.uint32)
    votes = np.array(
        [[v, p, c] for c, group in enumerate(g["votes"]) for v, p in group], np.uint32
    )
    bias = np.array(g["bias"], np.int32)

    def call(x, zal):
        bits = np.zeros(g["inputs"] + len(g["nodes"]), np.uint64)
        bits[: g["inputs"]] = np.bitwise_or.reduce(
            x.astype(np.uint64)
            * (np.uint64(1) << np.arange(len(x), dtype=np.uint64))[:, None],
            axis=0,
        )
        scores = np.zeros((len(x), g["classes"]), np.int32)
        code = fn(
            bits.ctypes.data,
            g["inputs"],
            nodes.ctypes.data,
            len(nodes),
            votes.ctypes.data,
            len(votes),
            bias.ctypes.data,
            scores.ctypes.data,
            g["classes"],
            len(x),
            zal,
        )
        assert code == 0
        return scores

    bundle = json.loads((root / "web/model.json").read_text())
    _, ds = dataset()
    x, _ = encode(ds["validation"], bundle["features"])
    rng = np.random.default_rng(20261104)
    x = np.r_[x, rng.integers(0, 2, (256, g["inputs"]))].astype(np.uint8)
    expected = evaluate(g, x)
    for zal in (0, 1):
        result = np.concatenate(
            [call(x[i : i + 64], zal) for i in range(0, len(x), 64)]
        )
        assert np.array_equal(result, expected)
    measurements = {}
    for zal in (0, 1):
        call(x[:64], zal)
        runs = []
        for _ in range(9):
            start = time.perf_counter_ns()
            for _ in range(40):
                call(x[:64], zal)
            runs.append((time.perf_counter_ns() - start) / 40 / 1e6)
        measurements["zal" if zal else "portable_c"] = dict(
            batch=64, median_ms=statistics.median(runs), samples_ms=runs
        )
    report = dict(
        schema="idi/zal-native-benchmark/1",
        model_sha256=hashlib.sha256((root / "web/model.json").read_bytes()).hexdigest(),
        compiler=subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
        platform=platform.platform(),
        vectors_checked=len(x),
        class_votes_checked=len(x) * g["classes"] * 2,
        kernels=identities,
        measurements=measurements,
        includes="Python/FFI input packing, native DAG evaluation and vote accumulation",
        excludes="Text encoding, retrieval, corpus loading; no browser speed comparison",
        scope="Measured without process isolation; other experiment jobs may run concurrently. ZAL proof covers gate words; whole-host correspondence is differential testing.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cc", default="cc")
    a = p.parse_args()
    run(a.out, a.cc)
