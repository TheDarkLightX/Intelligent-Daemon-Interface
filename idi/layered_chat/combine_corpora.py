"""Combine separately attributed OpenAssistant and Dolly educational answers.

This exported collection and its relevance model use CC BY-SA 3.0. Apache-2.0
notices for OpenAssistant are retained. No textbook passages are included.
"""

import argparse, hashlib, json
from pathlib import Path
from .openassistant import EXCLUDE, terms, pairs

DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_SOURCE = f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/{DOLLY_REVISION}"


def run(oa, dolly, out):
    data = json.loads(oa.read_text())
    records = data["records"]
    seen = {" ".join(terms(r["q"])) for r in records}
    for r in records:
        r.update(source=data["source"], license="Apache-2.0", dataset="OpenAssistant")
    count = 0
    for i, line in enumerate(dolly.read_text().splitlines()):
        row = json.loads(line)
        q = row["instruction"].strip()
        a = row["response"].strip()
        context = row.get("context", "").strip()
        key = " ".join(terms(q))
        if (
            not 15 <= len(q) <= 400
            or not 50 <= len(a) <= 2600
            or len(context) > 2200
            or len(terms(q)) < 2
            or key in seen
            or EXCLUDE.search(q + " " + a)
        ):
            continue
        ident = "dolly-" + str(i)
        b = int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16) % 10
        records.append(
            dict(
                id=ident,
                answer_id=ident,
                tree=ident,
                q=q,
                a=a,
                context=context,
                quality=None,
                quality_basis="Dolly human-written instruction dataset; not independently fact-checked",
                split="test" if b == 0 else "validation" if b == 1 else "train",
                source=DOLLY_SOURCE,
                license="CC-BY-SA-3.0",
                dataset="Dolly 15k",
                category=row["category"],
            )
        )
        seen.add(key)
        count += 1
    result = dict(
        schema="idi/attributed-answer-collection/2",
        source=data["source"],
        license="CC-BY-SA-3.0",
        attribution="OpenAssistant / LAION contributors and Databricks Dolly contributors",
        changes="Selected by language annotations, scope, size and duplicate filters; original answers retained. Per-record source and license added. Relevance split by conversation tree or Dolly row.",
        sources=[
            dict(
                name="OpenAssistant",
                url=data["source"],
                license="Apache-2.0",
                sha256=data["source_sha256"],
            ),
            dict(
                name="Databricks Dolly 15k",
                url=DOLLY_SOURCE,
                license="CC-BY-SA-3.0",
                sha256=hashlib.sha256(dolly.read_bytes()).hexdigest(),
            ),
        ],
        records=records,
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "knowledge.json").write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    report = dict(
        records=len(records),
        openassistant_records=len(records) - count,
        dolly_records=count,
        bytes=(out / "knowledge.json").stat().st_size,
        sources=result["sources"],
        license=result["license"],
    )
    (out / "knowledge-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    pairs(result, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--oa", type=Path, required=True)
    p.add_argument("--dolly", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.oa, a.dolly, a.out)
