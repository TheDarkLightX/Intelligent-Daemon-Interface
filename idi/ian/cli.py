from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .simulations.trading_agent_demo import DemoConfig, run_demo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IAN CLI - Trading Agent Competition Demo (Simulated Evaluation)",
    )
    parser.add_argument(
        "--contributors",
        type=int,
        default=5,
        help="Number of contributors",
    )
    parser.add_argument(
        "--contributions",
        type=int,
        default=3,
        help="Contributions per contributor",
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help="Disable security hardening",
    )
    parser.add_argument(
        "--no-tau",
        action="store_true",
        help="Disable Tau integration",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = DemoConfig(
        num_contributors=args.contributors,
        contributions_per_contributor=args.contributions,
        enable_security=not args.no_security,
        enable_tau=not args.no_tau,
        verbose=not args.quiet,
    )

    run_demo(config)


if __name__ == "__main__":
    main()

