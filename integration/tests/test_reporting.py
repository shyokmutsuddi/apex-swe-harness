"""Tests for result reporting."""

from apex_harness.reporting import get_exit_code, print_summary


def test_print_summary_all_success(capsys):
    """Test summary printing with all successful tasks."""
    results = [
        ("task1", 0, "output1", ""),
        ("task2", 0, "output2", ""),
    ]

    print_summary(results)
    captured = capsys.readouterr()

    assert "Total tasks: 2" in captured.out
    assert "Successful: 2" in captured.out
    assert "Failed: 0" in captured.out
    assert "task1" in captured.out
    assert "task2" in captured.out


def test_print_summary_with_failures(capsys):
    """Test summary printing with failed tasks."""
    results = [
        ("task1", 0, "output1", ""),
        ("task2", 1, "", "error message"),
        ("task3", 0, "output3", ""),
    ]

    print_summary(results)
    captured = capsys.readouterr()

    assert "Total tasks: 3" in captured.out
    assert "Successful: 2" in captured.out
    assert "Failed: 1" in captured.out
    assert "task2" in captured.out
    assert "error message" in captured.out


def test_get_exit_code_all_success():
    """Test exit code determination with all successful tasks."""
    results = [
        ("task1", 0, "output1", ""),
        ("task2", 0, "output2", ""),
    ]

    assert get_exit_code(results) == 0


def test_get_exit_code_with_failures():
    """Test exit code determination with failed tasks."""
    results = [
        ("task1", 0, "output1", ""),
        ("task2", 1, "", "error"),
    ]

    assert get_exit_code(results) == 1


def test_get_exit_code_all_failures():
    """Test exit code determination with all failed tasks."""
    results = [
        ("task1", 1, "", "error1"),
        ("task2", 1, "", "error2"),
    ]

    assert get_exit_code(results) == 1


def test_print_summary_empty_results(capsys):
    """Test summary printing with no results."""
    results = []

    print_summary(results)
    captured = capsys.readouterr()

    assert "Total tasks: 0" in captured.out
    assert "Successful: 0" in captured.out
    assert "Failed: 0" in captured.out
