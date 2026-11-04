"""Check deployed identities and replay the final folding certificate, offline."""

import gzip, hashlib, json
from pathlib import Path
from .check_certificate import replay


def run():
    root = Path(__file__).parent
    web = root / "web"
    report = json.loads((web / "report.json").read_text())
    bundle = json.loads((web / "model.json").read_text())
    assert report["metadata"]["provisional"] is False, "Release is still provisional"
    assert report["metadata"] == bundle["metadata"]
    assert bundle["metadata"]["trainedTopicTables"] >= 100000
    for name, meta in report["assets"].items():
        raw = (web / name).read_bytes()
        assert (
            len(raw) == meta["bytes"]
            and hashlib.sha256(raw).hexdigest() == meta["sha256"]
        ), name
    raw_knowledge = gzip.decompress((web / "knowledge.json.gz").read_bytes())
    assert len(raw_knowledge) == report["canonicalKnowledge"]["bytes"]
    assert hashlib.sha256(raw_knowledge).hexdigest() == report["canonicalKnowledge"]["sha256"]
    manifest = json.loads((web / "knowledge-manifest.json").read_text())
    records = [r for name in manifest.pop("shards") for r in json.loads((web / name).read_text())]
    assert dict(manifest, records=records) == json.loads(raw_knowledge)
    evidence = root / "evidence/final-fold"
    assert gzip.decompress((web / "model.json.gz").read_bytes()) == (web / "model.json").read_bytes()
    read = lambda name: json.loads(
        gzip.decompress((evidence / (name + ".json.gz")).read_bytes())
    )
    original, folded, certificate = (
        read(n) for n in ("original", "folded", "certificate")
    )
    assert folded == bundle["topic"]
    verdict = replay(original, folded, certificate)
    proof = json.loads((evidence / "report.json").read_text())
    assert proof["checkpoint_sha256"] == bundle["metadata"]["checkpointSha256"]
    assert len(original["nodes"]) == bundle["metadata"]["trainedTopicTables"]
    assert len(folded["nodes"]) == bundle["metadata"]["executedTopicTables"]
    assert (
        proof["checker_sha256"]
        == hashlib.sha256((root / "check_certificate.py").read_bytes()).hexdigest()
    )
    assert all(m["rejected"] for m in proof["mutants"])
    knowledge = json.loads(raw_knowledge)
    assert len({r["id"] for r in knowledge["records"]}) == len(knowledge["records"])
    assert all(
        r["license"] in ("Apache-2.0", "CC-BY-SA-3.0")
        and r["source"].startswith("https://huggingface.co/datasets/")
        for r in knowledge["records"]
    )
    print(
        json.dumps(
            dict(
                release="verified", answers=len(knowledge["records"]), folding=verdict
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
