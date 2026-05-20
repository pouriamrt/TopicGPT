"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from topicgpt.cli import app


def test_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("fit", "info", "eval", "viz"):
        assert sub in result.stdout


def test_fit_help_describes_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.stdout
    assert "--out" in result.stdout


def test_viz_help_describes_kinds() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["viz", "--help"])
    assert result.exit_code == 0
    assert "kind" in result.stdout.lower()
