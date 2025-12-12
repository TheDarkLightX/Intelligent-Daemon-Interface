"""Tests for proof manager timeout handling.

These tests verify that proof generation and verification:
1. Have timeout bounds on external subprocess calls.
2. Handle subprocess.TimeoutExpired gracefully.
3. Expose timeout configuration for customization.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from idi.zk.proof_manager import (
    generate_proof,
    PROVER_TIMEOUT_SECONDS,
)


class TestProverTimeoutConstant:
    """Tests for timeout constant definition."""

    def test_timeout_constant_exists(self) -> None:
        """PROVER_TIMEOUT_SECONDS should be defined."""
        assert PROVER_TIMEOUT_SECONDS is not None

    def test_timeout_is_reasonable(self) -> None:
        """Timeout should be reasonable (not too short or too long)."""
        assert PROVER_TIMEOUT_SECONDS >= 30  # At least 30 seconds
        assert PROVER_TIMEOUT_SECONDS <= 600  # At most 10 minutes


class TestGenerateProofTimeout:
    """Tests for timeout in generate_proof function."""

    def test_timeout_passed_to_subprocess(self, tmp_path: Path) -> None:
        """generate_proof should pass timeout to subprocess.run."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test": true}')
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_text("0\n1\n")
        out_dir = tmp_path / "out"

        # Mock subprocess.run to capture the call
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)

            # Use a custom prover command
            generate_proof(
                manifest_path=manifest,
                stream_dir=streams,
                out_dir=out_dir,
                prover_command="echo {manifest} {streams} {proof} {receipt}",
                auto_detect_risc0=False,
            )

            # Verify timeout was passed
            assert mock_run.called
            call_kwargs = mock_run.call_args.kwargs
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] == PROVER_TIMEOUT_SECONDS

    def test_timeout_expired_raises_appropriate_error(self, tmp_path: Path) -> None:
        """generate_proof should propagate TimeoutExpired as appropriate error."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test": true}')
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_text("0\n1\n")
        out_dir = tmp_path / "out"

        # Mock subprocess.run to raise TimeoutExpired
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="slow_prover",
                timeout=PROVER_TIMEOUT_SECONDS,
            )

            with pytest.raises(subprocess.TimeoutExpired):
                generate_proof(
                    manifest_path=manifest,
                    stream_dir=streams,
                    out_dir=out_dir,
                    prover_command="slow_prover {manifest} {streams} {proof} {receipt}",
                    auto_detect_risc0=False,
                )

    def test_stub_proof_no_timeout_needed(self, tmp_path: Path) -> None:
        """Stub proofs (no external command) should complete without subprocess."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test": true}')
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_text("0\n1\n")
        out_dir = tmp_path / "out"

        # Mock subprocess.run - should NOT be called for stub proofs
        with mock.patch("subprocess.run") as mock_run:
            bundle = generate_proof(
                manifest_path=manifest,
                stream_dir=streams,
                out_dir=out_dir,
                prover_command=None,
                auto_detect_risc0=False,
            )

            # Stub proof should not call subprocess
            mock_run.assert_not_called()

            # But should still create proof files
            assert bundle.proof_path.exists()
            assert bundle.receipt_path.exists()


class TestProverSubprocessSafety:
    """Tests for subprocess safety measures."""

    def test_no_shell_true_in_subprocess(self, tmp_path: Path) -> None:
        """generate_proof should never use shell=True."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test": true}')
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_text("0\n1\n")
        out_dir = tmp_path / "out"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)

            generate_proof(
                manifest_path=manifest,
                stream_dir=streams,
                out_dir=out_dir,
                prover_command="prover {manifest} {streams} {proof} {receipt}",
                auto_detect_risc0=False,
            )

            # Verify shell=True is NOT used
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("shell", False) is False

    def test_command_uses_shlex_split(self, tmp_path: Path) -> None:
        """generate_proof should use shlex.split for safe command parsing."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test": true}')
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_text("0\n1\n")
        out_dir = tmp_path / "out"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)

            generate_proof(
                manifest_path=manifest,
                stream_dir=streams,
                out_dir=out_dir,
                prover_command="prover --arg value {manifest} {streams} {proof} {receipt}",
                auto_detect_risc0=False,
            )

            # First positional arg should be a list (shlex.split result)
            call_args = mock_run.call_args.args[0]
            assert isinstance(call_args, list)
            assert call_args[0] == "prover"
            assert "--arg" in call_args
            assert "value" in call_args
