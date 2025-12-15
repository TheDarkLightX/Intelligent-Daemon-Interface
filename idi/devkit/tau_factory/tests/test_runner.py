"""Tests for TauRunner process spawning and output capture (TDD)."""

import os
import time
import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from idi.devkit.tau_factory.runner import TauResult, run_tau_spec, TauConfig, TauMode
except ImportError:
    pytest.skip("Runner module not yet implemented", allow_module_level=True)


class TestTauRunner:
    """Test TauRunner functionality."""

    def test_tau_result_creation(self):
        """TauResult should store execution results."""
        result = TauResult(
            success=True,
            outputs={"position.out": ["0", "1", "0"]},
            errors=[],
            duration_ms=123.45,
        )
        assert result.success is True
        assert "position.out" in result.outputs
        assert len(result.outputs["position.out"]) == 3

    def test_run_tau_spec_creates_temp_spec(self, tmp_path):
        """run_tau_spec should create temporary spec file."""
        spec_content = "i0:sbf = in file(\"inputs/test.in\").\no0:sbf = out file(\"outputs/test.out\").\ndefs\nr (o0[t] = i0[t])\nn\nq"
        spec_path = tmp_path / "test.tau"
        spec_path.write_text(spec_content)
        
        # Create inputs directory
        inputs_dir = tmp_path / "inputs"
        inputs_dir.mkdir()
        (inputs_dir / "test.in").write_text("1\n0\n1\n")
        
        # Create outputs directory
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        # Mock tau binary (just echo for testing)
        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\ncat > /dev/null\n")
        mock_tau.chmod(0o755)

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[str(mock_tau)],
                returncode=0,
                stdout=b"",
                stderr=b"",
            )

            result = run_tau_spec(spec_path, mock_tau)
            assert result.duration_ms > 0

    def test_file_mode_places_spec_before_flags(self, tmp_path):
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        def _assert_cmd(cmd, *, cwd, timeout, stdin):
            assert cmd[0] == str(mock_tau)
            assert cmd[1] == str(spec_path)
            assert "-x" in cmd
            assert cmd.index("-x") > 1
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

        with patch("idi.devkit.tau_factory.runner._run_subprocess", side_effect=_assert_cmd):
            result = run_tau_spec(
                spec_path,
                mock_tau,
                config=TauConfig(mode=TauMode.FILE, experimental=True, timeout=1.0),
            )
            assert result.success is True

    def test_file_mode_detects_repl_fallback(self, tmp_path):
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        repl_stdout = b"tau> Welcome\n"

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[str(mock_tau), str(spec_path), "-x"],
                returncode=0,
                stdout=repl_stdout,
                stderr=b"",
            )
            result = run_tau_spec(
                spec_path,
                mock_tau,
                config=TauConfig(mode=TauMode.FILE, experimental=True, timeout=1.0),
            )
            assert result.success is False
            assert any("entered repl" in err.lower() for err in result.errors)

    def test_run_tau_spec_handles_timeout(self, tmp_path):
        """run_tau_spec should handle timeout gracefully."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")
        
        # Mock a binary that hangs
        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nsleep 60\n")
        mock_tau.chmod(0o755)

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="tau", timeout=1)
            result = run_tau_spec(spec_path, mock_tau, config=TauConfig(timeout=1.0))
            assert result.success is False
            assert "timed out" in result.errors[0].lower() or result.duration_ms > 0

    def test_run_tau_spec_timeout_terminates_child_processes(self, tmp_path):
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        child_pid_path = tmp_path / "child.pid"
        mock_tau = tmp_path / "tau"
        mock_tau.write_text(
            "#!/bin/bash\n"
            "sleep 60 &\n"
            f"echo $! > {child_pid_path}\n"
            "sleep 60\n"
        )
        mock_tau.chmod(0o755)

        result = run_tau_spec(spec_path, mock_tau, config=TauConfig(timeout=1.0))
        assert result.success is False
        assert any("timed out" in err.lower() for err in result.errors)

        # Best-effort: ensure the child process was killed.
        assert child_pid_path.exists(), "Child PID file was not created; timeout test is inconclusive"
        child_pid = int(child_pid_path.read_text().strip())

        # Allow a short grace period for SIGKILL to take effect.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)

        pytest.fail("Child process still alive after timeout; process group termination likely failed")

    def test_run_tau_spec_captures_outputs(self, tmp_path):
        """run_tau_spec should capture output files."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
            def _side_effect(*args, **kwargs):
                (outputs_dir / "position.out").write_text("0\n1\n0\n")
                (outputs_dir / "buy_signal.out").write_text("1\n0\n0\n")
                return subprocess.CompletedProcess(
                    args=[str(mock_tau)],
                    returncode=0,
                    stdout=b"",
                    stderr=b"",
                )

            mock_run.return_value = subprocess.CompletedProcess(
                args=[str(mock_tau)],
                returncode=0,
                stdout=b"",
                stderr=b"",
            )

            mock_run.side_effect = _side_effect

            result = run_tau_spec(spec_path, mock_tau)
            assert "position.out" in result.outputs

    def test_run_tau_spec_handles_missing_tau_binary(self, tmp_path):
        """run_tau_spec should handle missing tau binary."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")
        fake_tau = Path("/nonexistent/tau/binary")
        
        result = run_tau_spec(spec_path, fake_tau)
        assert result.success is False
        assert len(result.errors) > 0

    def test_run_tau_spec_parses_stderr(self, tmp_path):
        """run_tau_spec should capture stderr errors."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("invalid syntax")

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 1\n")
        mock_tau.chmod(0o755)

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[str(mock_tau)],
                returncode=1,
                stdout=b"",
                stderr=b"Syntax Error: Unexpected token",
            )

            result = run_tau_spec(spec_path, mock_tau)
            assert result.success is False
            assert "syntax" in result.errors[0].lower() or "error" in result.errors[0].lower()

    def test_run_tau_spec_measures_duration(self, tmp_path):
        """run_tau_spec should measure execution duration."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run, patch("time.perf_counter") as mock_time:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[str(mock_tau)],
                returncode=0,
                stdout=b"",
                stderr=b"",
            )

            mock_time.side_effect = [0.0, 0.123]  # Start, end

            result = run_tau_spec(spec_path, mock_tau)
            assert result.duration_ms > 0

    def test_stdin_mode_falls_back_to_file_mode_on_usage_error(self, tmp_path):
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        calls = []

        def _side_effect(cmd, *, cwd, timeout, stdin):
            calls.append((cmd, stdin))
            if len(calls) == 1:
                assert cmd[0] == str(mock_tau)
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=2,
                    stdout=b"",
                    stderr=b"Usage: tau <spec.tau>\n",
                )

            assert cmd[0] == str(mock_tau)
            assert cmd[1] == str(spec_path)
            assert stdin is subprocess.DEVNULL

            (outputs_dir / "position.out").write_text("0\n1\n")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=b"",
                stderr=b"",
            )

        with patch("idi.devkit.tau_factory.runner._run_subprocess", side_effect=_side_effect):
            result = run_tau_spec(spec_path, mock_tau, config=TauConfig(mode=TauMode.STDIN, timeout=1.0))
            assert result.success is True
            assert len(calls) == 2
            assert "position.out" in result.outputs

    def test_run_tau_spec_detects_tau_error_patterns(self, tmp_path):
        """run_tau_spec should detect various Tau error patterns."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")

        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nexit 0\n")
        mock_tau.chmod(0o755)

        error_cases = [
            (b"(Error) Type mismatch due to predefinition", "Type mismatch"),
            (b"(Error) Tau specification is unsat", "is unsat"),
            (b"(Error) Unresolved function or predicate symbol", "Unresolved function"),
            (b"(Error) Failed to find output stream for stream 'x'", "Failed to find output stream"),
        ]

        for stdout_content, expected_pattern in error_cases:
            with patch("idi.devkit.tau_factory.runner._run_subprocess") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[str(mock_tau)],
                    returncode=0,
                    stdout=stdout_content,
                    stderr=b"",
                )

                result = run_tau_spec(spec_path, mock_tau)
                assert result.success is False, f"Should fail for: {expected_pattern}"
                assert any(expected_pattern in e for e in result.errors), f"Should detect: {expected_pattern}"

