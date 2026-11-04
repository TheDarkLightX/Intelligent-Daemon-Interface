"""Test whether seven smaller trained networks use a 111,104-table budget better."""

import concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent


def train(seed):
    name = f"ensemble-v64-d4-s{seed}"
    out = ROOT / "runs" / name
    start = time.monotonic()
    args = [
        sys.executable,
        "-m",
        "idi.layered_chat.train",
        "--out",
        str(out),
        "--votes",
        "64",
        "--depth",
        "4",
        "--epochs",
        "30",
        "--seed",
        str(seed),
        "--encoder",
        "words-first",
        "--features",
        "3072",
        "--anneal",
        "--rate",
        ".025",
    ]
    with (ROOT / "runs" / (name + ".log")).open("w") as log:
        p = subprocess.run(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        )
    row = dict(
        seed=seed, returncode=p.returncode, seconds=round(time.monotonic() - start, 3)
    )
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    plan = dict(
        seeds=[0, 2, 3, 4, 5, 6],
        existing_seed1="control-15872-annealed-seed1",
        members=7,
        tables=111104,
        rule="Equal integer votes; fixed seven seeds. Validation-only architecture selection; test remains sealed.",
    )
    (ROOT / "runs/ensemble-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(train, plan["seeds"]))
    (ROOT / "runs/ensemble-campaign.json").write_text(json.dumps(rows, indent=2) + "\n")
