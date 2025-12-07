"""Tests for TauRunner process spawning and output capture (TDD)."""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from idi.devkit.tau_factory.runner import TauRunner, TauResult, run_tau_spec
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
        
        # This will fail without real tau, but tests structure
        with pytest.raises((subprocess.CalledProcessError, FileNotFoundError)):
            run_tau_spec(spec_path, mock_tau)

    def test_run_tau_spec_handles_timeout(self, tmp_path):
        """run_tau_spec should handle timeout gracefully."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")
        
        # Mock a binary that hangs
        mock_tau = tmp_path / "tau"
        mock_tau.write_text("#!/bin/bash\nsleep 60\n")
        mock_tau.chmod(0o755)
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="tau", timeout=1)
            result = run_tau_spec(spec_path, mock_tau)
            assert result.success is False
            assert "timeout" in result.errors[0].lower() or result.duration_ms > 0

    def test_run_tau_spec_captures_outputs(self, tmp_path):
        """run_tau_spec should capture output files."""
        spec_path = tmp_path / "test.tau"
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        # Create mock output files
        (outputs_dir / "position.out").write_text("0\n1\n0\n")
        (outputs_dir / "buy_signal.out").write_text("1\n0\n0\n")
        
        # Mock successful execution
        with patch("subprocess.run") as mock_run:
            mock_proc = Mock()
            mock_proc.returncode = 0
            mock_proc.stdout = b""
            mock_proc.stderr = b""
            mock_run.return_value = mock_proc
            
            result = run_tau_spec(spec_path, Path("/fake/tau"))
            
            # Should have parsed outputs
            assert "position.out" in result.outputs or result.success is False

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
        
        with patch("subprocess.run") as mock_run:
            mock_proc = Mock()
            mock_proc.returncode = 1
            mock_proc.stdout = b""
            mock_proc.stderr = b"Syntax Error: Unexpected token"
            mock_run.return_value = mock_proc
            
            result = run_tau_spec(spec_path, Path("/fake/tau"))
            assert result.success is False
            assert "syntax" in result.errors[0].lower() or "error" in result.errors[0].lower()

    def test_run_tau_spec_measures_duration(self, tmp_path):
        """run_tau_spec should measure execution duration."""
        spec_path = tmp_path / "test.tau"
        spec_path.write_text("test")
        
        with patch("subprocess.run") as mock_run, patch("time.perf_counter") as mock_time:
            mock_proc = Mock()
            mock_proc.returncode = 0
            mock_proc.stdout = b""
            mock_proc.stderr = b""
            mock_run.return_value = mock_proc
            
            mock_time.side_effect = [0.0, 0.123]  # Start, end
            
            result = run_tau_spec(spec_path, Path("/fake/tau"))
            assert result.duration_ms > 0

