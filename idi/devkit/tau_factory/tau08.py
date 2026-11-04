"""Tau 0.8 file specifications and conservative migration of 0.7 REPL scripts.

This translator changes the file envelope and stream bindings. It does not
invent missing types, hash verification, persistence, or an initial state.
Native execution is a separate qualification step.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

BINDING = re.compile(
    r'^(\s*)([A-Za-z_][A-Za-z_0-9]*\s*:\s*(?:sbf|bv\[\d+\]))\s*(?::=|=)\s*((?:in|out)\s+(?:file\("[^"\n]*"\)|console))\s*\.?\s*(#.*)?$'
)
OLD_FILE = re.compile(
    r'^(\s*)(sbf|bv\[\d+\])\s+([A-Za-z_]\w*)\s*=\s*([io])file\(("[^"\n]*")\)\s*\.?\s*(#.*)?$'
)
OLD_CONSOLE = re.compile(
    r"^(\s*)(sbf|bv\[\d+\])\s+([io]\d+)\s*=\s*console\s*\.?\s*(#.*)?$"
)


@dataclass(frozen=True)
class Migration:
    text: str
    changed: bool
    issues: tuple[str, ...]


def without_comment(line: str) -> str:
    depth = 0
    quoted = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quoted:
            escaped = True
            continue
        if ch == '"':
            quoted = not quoted
        if quoted:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "#" and depth == 0:
            return line[:i]
    return line


def migrate(text: str) -> Migration:
    lines = text.splitlines()
    out = []
    issues = []
    running = False
    body = []
    had_run = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            (body if running else out).append(line)
            continue
        old = OLD_FILE.match(line)
        if old:
            line = (
                f'{old[1]}{old[3]}:{old[2]} := {"in" if old[4]=="i" else "out"} file({old[5]}). '
                + (old[6] or "")
            )
        old = OLD_CONSOLE.match(line)
        if old:
            line = (
                f'{old[1]}{old[3]}:{old[2]} := {"in" if old[3][0]=="i" else "out"} console. '
                + (old[4] or "")
            )
        m = BINDING.match(line)
        if m:
            if running:
                issues.append("stream binding after the run formula")
            out.append(f"{m[1]}{m[2]} := {m[3]}. " + (m[4] or ""))
            continue
        if re.match(r"^(?:set\s+)?charvar\s+(?:off|on)\.?$", stripped):
            continue
        if stripped in ("q", "q.", "quit", "quit.", "n", "n."):
            continue
        if running and re.fullmatch(r"[01]+", stripped):
            issues.append(
                "removed interactive Boolean input transcript after run formula"
            )
            continue
        if re.match(r"^r(?:\s|\(|$)", stripped):
            if had_run:
                issues.append("multiple run commands")
            had_run = True
            running = True
            body.append(re.sub(r"^r\s*", "", stripped, count=1))
            continue
        if stripped == "defs" or stripped.startswith("defs("):
            issues.append("legacy defs block needs a semantic rewrite")
        (body if running else out).append(line)
    if had_run:
        # Remove only the formula's trailing period; preserve comments and content.
        formula = "\n".join(body).strip()
        code_lines = [
            i
            for i, l in enumerate(formula.splitlines())
            if l.strip() and not l.lstrip().startswith("#")
        ]
        fs = formula.splitlines()
        if code_lines:
            j = code_lines[-1]
            before = without_comment(fs[j])
            comment = fs[j][len(before) :]
            fs[j] = before.rstrip().removesuffix(".") + (
                " " + comment if comment else ""
            )
        formula = "\n".join(fs)
        # A historical single ampersand joined a formula and an initial-state
        # equality. Tau 0.8 distinguishes formula conjunction from BF meet.
        formula = re.sub(r"\)\s*&\s*(?=\(\w+\[0\]\s*=)", ") && ", formula)
        out.append("always (\n" + formula + "\n).")
    result = "\n".join(out).rstrip() + "\n"
    if not re.search(r"\b(?:always|sometimes)\b", result) and had_run:
        issues.append("no temporal formula")
    return Migration(result, result != text, tuple(dict.fromkeys(issues)))


def validate_file_spec(text: str) -> tuple[bool, list[str]]:
    """Envelope lint only; never reported as a Tau parser or solver verdict."""
    errors = []
    code = "\n".join(without_comment(l) for l in text.splitlines())
    if re.search(r"^\s*(?:r|q|n)\s*$", code, re.M):
        errors.append("REPL command in a 0.8 file specification")
    if re.search(r":\s*(?:sbf|bv\[\d+\])\s*=\s*(?:in|out)\b", code):
        errors.append("stream binding requires :=")
    if "always" not in code and "sometimes" not in code:
        errors.append("missing temporal requirement")
    if code.count("(") != code.count(")"):
        errors.append("unbalanced parentheses")
    if not code.rstrip().endswith("."):
        errors.append("missing final period")
    return not errors, errors
