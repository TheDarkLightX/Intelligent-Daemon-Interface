"""Regenerate the recorded reference schemas and qualify with a user-installed Tau API."""

import concurrent.futures, hashlib, json
from pathlib import Path
from .schema import AgentSchema, StreamConfig, LogicBlock
from .generator import generate_tau_spec
from .audit_tau08 import qualify


def regenerate(directory):
    records = json.loads((directory / "cases.json").read_text())
    for row in records:
        s = dict(row["schema"])
        s["streams"] = tuple(StreamConfig(**v) for v in s["streams"])
        s["logic_blocks"] = tuple(
            LogicBlock(**dict(v, inputs=tuple(v["inputs"]))) for v in s["logic_blocks"]
        )
        text = generate_tau_spec(AgentSchema(**s))
        (directory / (row["pattern"] + ".tau")).write_text(text)
    return records


def run():
    directory = Path(__file__).parent / "tests/tau08_cases"
    rows = regenerate(directory)
    paths = [directory / (r["pattern"] + ".tau") for r in rows]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda p: qualify(p, 30), paths))
    report = dict(
        tau_commit="95a7d261ca3db5ff6f7c198a0b4d81c880816058",
        scope="One recorded reference schema per factory pattern. Native interpreter admission, not all parameter combinations or execution traces.",
        results=[
            dict(
                pattern=r["pattern"],
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                **v,
            )
            for r, p, v in zip(rows, paths, results)
        ],
    )
    (directory / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            status: sum(r["status"] == status for r in report["results"])
            for status in {r["status"] for r in report["results"]}
        }
    )


if __name__ == "__main__":
    run()
