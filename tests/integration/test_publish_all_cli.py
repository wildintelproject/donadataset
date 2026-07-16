"""Integration tests for 'publish all'.

"""
from typer.testing import CliRunner

from donadataset.commands import publish_all as publish_all_cmd
from donadataset.main import app

runner = CliRunner()

















    assert result.exit_code == 0, result.output




    assert result.exit_code == 0, result.output




    assert result.exit_code == 0, result.output


def test_publish_all_stops_on_first_failing_step(monkeypatch):

    result = runner.invoke(app, ["publish", "all"])

    assert result.exit_code == 1
