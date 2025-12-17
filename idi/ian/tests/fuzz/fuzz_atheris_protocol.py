from __future__ import annotations

import sys

from idi.ian.network.protocol import Message


def _bounded(data: bytes, *, max_len: int = 1_000_000) -> bool:
    return len(data) <= max_len


def main() -> int:
    try:
        import atheris  # type: ignore
    except Exception:
        print("atheris not installed; skipping fuzz harness")
        return 0

    def TestOneInput(data: bytes) -> None:
        if not _bounded(data):
            return
        try:
            Message.from_wire(data)
        except Exception:
            return

    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
