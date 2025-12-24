"""Atheris fuzzing harness for InputValidator.

Run with:
    python -m idi.ian.tests.fuzz.fuzz_atheris_input_validator

Requires: pip install atheris

This harness tests InputValidator's robustness against malformed/adversarial input.
The validator handles untrusted data from network peers, so it must not crash.
"""
from __future__ import annotations

import sys


def _bounded(data: bytes, *, max_len: int = 100_000) -> bool:
    """Reject excessively large inputs to avoid OOM."""
    return len(data) <= max_len


def main() -> int:
    try:
        import atheris  # type: ignore
    except ImportError:
        print("atheris not installed; skipping fuzz harness")
        print("Install with: pip install atheris")
        return 0

    from idi.ian.models import AgentPack, Contribution, GoalID
    from idi.ian.security import InputValidator, SecurityLimits

    # Use strict limits to catch boundary issues
    limits = SecurityLimits(
        MAX_PACK_VERSION_LEN=32,
        MAX_PACK_PARAMETERS_SIZE=1024,
        MAX_PACK_METADATA_KEYS=10,
        MAX_PACK_METADATA_KEY_LEN=32,
        MAX_PACK_METADATA_VALUE_SIZE=256,
        MAX_PACK_METADATA_SIZE=4096,
        MAX_CONTRIBUTOR_ID_LEN=64,
        MAX_GOAL_ID_LEN=64,
        MAX_PROOFS_COUNT=10,
        MAX_PROOF_SIZE=1024,
        MAX_TOTAL_PROOFS_SIZE=8192,
    )
    validator = InputValidator(limits)

    @atheris.instrument_func
    def TestAgentPack(data: bytes) -> None:
        """Fuzz AgentPack validation - must not crash."""
        if not _bounded(data):
            return

        fdp = atheris.FuzzedDataProvider(data)

        # Generate fuzzed AgentPack fields
        version = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
        params_len = fdp.ConsumeIntInRange(0, min(2048, fdp.remaining_bytes()))
        parameters = fdp.ConsumeBytes(params_len)

        # Generate fuzzed metadata
        num_keys = fdp.ConsumeIntInRange(0, 20)
        metadata: dict[str, str] = {}
        for _ in range(num_keys):
            key = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
            value = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 300))
            if key:  # Skip empty keys
                metadata[key] = value

        try:
            pack = AgentPack(version=version, parameters=parameters, metadata=metadata)
            # Validator must handle any input without crashing
            result = validator.validate_agent_pack(pack)
            # Result must be well-formed
            assert hasattr(result, "valid")
            assert hasattr(result, "field")
            assert hasattr(result, "error")
        except (ValueError, TypeError):
            # These are acceptable rejections from AgentPack constructor
            pass

    @atheris.instrument_func
    def TestContribution(data: bytes) -> None:
        """Fuzz Contribution validation - must not crash."""
        if not _bounded(data):
            return

        fdp = atheris.FuzzedDataProvider(data)

        # Generate fuzzed fields
        goal_id_str = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
        contributor_id = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
        version = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
        params_len = fdp.ConsumeIntInRange(0, min(1024, fdp.remaining_bytes()))
        parameters = fdp.ConsumeBytes(params_len)

        # Generate fuzzed proofs
        num_proofs = fdp.ConsumeIntInRange(0, 15)
        proofs: dict[str, bytes] = {}
        for i in range(num_proofs):
            proof_name = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(1, 20))
            proof_len = fdp.ConsumeIntInRange(0, min(2048, fdp.remaining_bytes()))
            proof_data = fdp.ConsumeBytes(proof_len)
            if proof_name:
                proofs[proof_name or f"p{i}"] = proof_data

        try:
            goal_id = GoalID(goal_id_str) if goal_id_str else GoalID("DEFAULT")
            pack = AgentPack(version=version or "1.0", parameters=parameters, metadata={})
            contrib = Contribution(
                goal_id=goal_id,
                agent_pack=pack,
                contributor_id=contributor_id or "anon",
                proofs=proofs,
                seed=fdp.ConsumeIntInRange(0, 2**32 - 1),
            )
            # Validator must handle any input without crashing
            result = validator.validate_contribution(contrib)
            assert hasattr(result, "valid")
        except (ValueError, TypeError):
            # Acceptable rejections from constructors
            pass

    def TestOneInput(data: bytes) -> None:
        """Main fuzz target - randomly choose what to fuzz."""
        if not data:
            return
        if data[0] % 2 == 0:
            TestAgentPack(data[1:])
        else:
            TestContribution(data[1:])

    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
