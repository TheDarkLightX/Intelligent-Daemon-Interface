import pytest

from idi import cli


def test_cli_build_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["build", "--help"])
    assert exc.value.code == 0
