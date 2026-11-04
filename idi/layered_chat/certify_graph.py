"""Emit and independently replay a certificate for a frozen model."""

import argparse, gzip, hashlib, json, time, copy
from pathlib import Path
from .graph import from_layers, fold
from .check_certificate import replay, Refused


def run(checkpoint, out):
    out.mkdir(parents=True, exist_ok=False)
    raw = checkpoint.read_bytes()
    original = from_layers(json.loads(raw))
    certificate = {}
    start = time.monotonic()
    folded = fold(original, certificate)
    result = replay(original, folded, certificate)
    for name, data in [
        ("original", original),
        ("folded", folded),
        ("certificate", certificate),
    ]:
        (out / (name + ".json.gz")).write_bytes(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
        )
    mutants = []
    for name in ["gate", "output-bias", "drop-vote", "projection", "input-map"]:
        c = copy.deepcopy(certificate)
        f = copy.deepcopy(folded)
        if name == "gate":
            c["nodes"][0][2] ^= 1
        elif name == "output-bias":
            c["bias"][0] += 1
        elif name == "drop-vote":
            c["votes"][0].pop()
        elif name == "projection":
            f["nodes"][0][2] ^= 1
        else:
            c["mapping"][0] = (0, 1)
        try:
            replay(original, f, c)
        except (Refused, KeyError) as e:
            mutants.append(dict(mutation=name, rejected=True, reason=str(e)))
        else:
            raise AssertionError("Mutant survived: " + name)
    result.update(
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        seconds=round(time.monotonic() - start, 3),
        mutants=mutants,
        checker_sha256=hashlib.sha256(
            (Path(__file__).parent / "check_certificate.py").read_bytes()
        ).hexdigest(),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.checkpoint, a.out)
