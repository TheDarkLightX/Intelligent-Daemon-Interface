"""Migrate file envelopes and record native Tau 0.8 qualification per source.

Historical sketches remain distinguishable from executable contracts. A parse
or synthesis refusal is recorded, never promoted to a successful migration.
"""

import argparse, concurrent.futures, hashlib, json, re, subprocess, sys, time
from pathlib import Path
from .tau08 import migrate

CHILD = r"""
import json,re,sys,tau
text=open(sys.argv[1]).read()
text=re.sub(r'(in|out)\s+file\("[^"\n]*"\)',lambda m:m[1]+' console',text)
tau.set_severity('error');tau.set_charvar(False)
r=tau.get_interpreter(text)
print('IDI_RESULT '+json.dumps(dict(accepted=bool(r),errors=[str(e) for e in r.report.errors])))
"""


def qualify(path, timeout=8):
    started = time.monotonic()
    try:
        p = subprocess.run(
            [sys.executable, "-c", CHILD, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = [
            l[len("IDI_RESULT ") :]
            for l in p.stdout.splitlines()
            if l.startswith("IDI_RESULT ")
        ]
        result = (
            json.loads(lines[-1])
            if lines
            else dict(
                accepted=False,
                errors=[(p.stderr or p.stdout)[-1500:]],
                returncode=p.returncode,
            )
        )
        result["status"] = (
            "native-interpreter-accepted"
            if result.get("accepted")
            else "native-refused"
        )
    except subprocess.TimeoutExpired:
        result = dict(
            accepted=False,
            status="budget-inconclusive",
            errors=["native admission exceeded the wall-clock budget"],
        )
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def run(repo, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    jobs = []
    sources = [
        p
        for p in sorted(repo.rglob("*.tau"))
        if not any(x in p.parts for x in ["archive", ".git", "layered_chat", "tau08"])
    ]
    for p in sources:
        rel = p.relative_to(repo)
        text = p.read_text()
        m = migrate(text)
        target = out / "migrated" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(m.text)
        row = dict(
            source=str(rel),
            source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            target=str(target.relative_to(repo)),
            target_sha256=hashlib.sha256(m.text.encode()).hexdigest(),
            migration_issues=list(m.issues),
        )
        rows.append(row)
        jobs.append(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row, result in zip(rows, pool.map(qualify, jobs)):
            row["native"] = result
            print(row["source"], result["status"], flush=True)
    counts = {
        k: sum(r["native"]["status"] == k for r in rows)
        for k in {r["native"]["status"] for r in rows}
    }
    report = dict(
        tau_source_commit="95a7d261ca3db5ff6f7c198a0b4d81c880816058",
        files=len(rows),
        counts=counts,
        scope="Stream/file-envelope migration followed by native interpreter admission with file streams remapped to console. Admission is not execution, invariants or preservation of intended behavior. Historical files remain in place.",
        files_detail=rows,
    )
    (out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.repo, a.out)
