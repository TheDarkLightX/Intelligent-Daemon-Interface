from pathlib import Path
import subprocess
import sys


def test_run_idi_trainer_help() -> None:
    script = Path(__file__).resolve().parents[1] / "run_idi_trainer.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "drift-bull" in result.stdout

