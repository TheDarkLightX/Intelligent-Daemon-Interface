"""Source-bound ZAL synthesis of 64-lane Boolean gate kernels.

Native CPU kernels are separate from the portable browser evaluator.
The claim is returned u64-word equivalence under ZAL's ISA model.
"""

import argparse, json, subprocess
from pathlib import Path


def expression(f):
    cells = []
    for a in (0, 1):
        for b in (0, 1):
            if (f >> (3 - 2 * a - b)) & 1:
                cells.append(
                    f'(and {"x" if a else "(not x)"} {"y" if b else "(not y)"})'
                )
    if not cells:
        return "0"
    while len(cells) > 1:
        cells = [f"(or {cells[0]} {cells[1]})"] + cells[2:]
    return cells[0]


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    cli = source / "integrations/zal/run.py"

    def call(*args):
        p = subprocess.run(
            ["python", str(cli), "word-kernel", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            r = json.loads(p.stdout)
        except ValueError:
            r = {"stdout": p.stdout, "stderr": p.stderr}
        return p.returncode, r

    palette = [
        ["mov", "rax", "rdi"],
        ["mov", "rax", "rsi"],
        ["xor", "rax", "rax"],
        ["imm", "rdx", "18446744073709551615"],
        ["xor", "rax", "rdx"],
        ["and", "rax", "rsi"],
        ["or", "rax", "rsi"],
        ["or", "rax", "rdi"],
        ["xor", "rax", "rsi"],
        ["mov", "rcx", "rsi"],
        ["xor", "rcx", "rdx"],
        ["and", "rax", "rcx"],
        ["or", "rax", "rcx"],
    ]
    for f in range(16):
        spec = out / f"gate-{f:02}.zal"
        spec.write_text(
            f'profile zal/word-kernel/1;\nkernel gate{f};\ntarget x86_64-sysv-register/1;\ninputs x:u64,y:u64;\nresult {expression(f)} modulo 2^64;\npalette {json.dumps(palette,separators=(",",":"))};\nmax-body 4;\nobjective bytes;\nlimits candidates=20000 checks=32 deadline-ms=3000 solver-ms=500;\n'
        )
        code, read = call("read", "--source", spec)
        assert code == 0, read
        revision = (
            read.get("revision")
            or read.get("binding", {}).get("revision")
            or read["binding"]["source_sha256"]
        )
        session = out / f"gate-{f:02}-session"
        code, opened = call(
            "open", "--source", spec, "--revision", revision, "--session", session
        )
        assert code == 0, opened
        code, searched = call("search", "--session", session, "--revision", revision)
        artifact = out / f"gate-{f:02}-artifact"
        ec, emitted = call(
            "emit", "--session", session, "--revision", revision, "--out", artifact
        )
        replay = None
        if ec == 0:
            rc, replay = call("replay", "--artifact", artifact, "--revision", revision)
            assert rc == 0, replay
        rows.append(
            dict(gate=f, read=read, search=searched, emit=emitted, replay=replay)
        )
        print(
            json.dumps(
                dict(gate=f, search=searched.get("status"), emit=emitted.get("status"))
            ),
            flush=True,
        )
    report = dict(
        schema="idi/zal-gate-kernels/1",
        source_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="Returned u64-word equivalence under the registered ZAL ISA model; no browser, complete ABI or speed claim.",
        rows=rows,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.source.resolve(), a.out.resolve())
