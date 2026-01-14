"""Tests for CLI interface."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from apex_harness.cli import create_parser, main, validate_environment
from apex_harness.models import ModelConfig


def test_create_parser():
    """Test argument parser creation."""
    parser = create_parser()
    assert parser is not None

    # Test valid arguments
    args = parser.parse_args(["--model", "claude"])
    assert args.model == "claude"


def test_parser_requires_model():
    """Test that model argument is required."""
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_validates_model_choices():
    """Test that invalid model is rejected."""
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "invalid-model"])


def test_parser_tasks_argument():
    """Test tasks argument parsing."""
    parser = create_parser()
    args = parser.parse_args(["--model", "claude", "--tasks", "task1", "task2"])

    assert args.tasks == ["task1", "task2"]


def test_parser_parallel_flag():
    """Test parallel execution flag."""
    parser = create_parser()
    args = parser.parse_args(["--model", "claude", "--parallel"])

    assert args.parallel is True


def test_parser_dry_run_flag():
    """Test dry run flag."""
    parser = create_parser()
    args = parser.parse_args(["--model", "claude", "--dry-run"])

    assert args.dry_run is True


def test_validate_environment_no_requirement():
    """Test environment validation with no requirements."""
    config = ModelConfig(
        name="Test",
        identifier="test",
        report_prefix="test",
        status_csv_prefix="test",
    )

    # Should not raise
    validate_environment(config)


def test_validate_environment_with_requirement(monkeypatch):
    """Test environment validation with required variable."""
    config = ModelConfig(
        name="Test",
        identifier="test",
        report_prefix="test",
        status_csv_prefix="test",
        requires_env_var="TEST_API_KEY",
    )

    # Test without variable set
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    validate_environment(config)  # Should print warning but not fail

    # Test with variable set
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    validate_environment(config)  # Should not print warning


@patch("apex_harness.cli.get_all_tasks")
@patch("apex_harness.cli.run_sequential")
def test_main_basic_execution(mock_run, mock_get_tasks, tmp_path, monkeypatch):
    """Test basic CLI execution flow."""
    # Setup
    mock_get_tasks.return_value = ["task1", "task2"]
    mock_run.return_value = [("task1", 0, "", ""), ("task2", 0, "", "")]

    # Create tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["apex-runner", "--model", "claude", "--dry-run"]
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
