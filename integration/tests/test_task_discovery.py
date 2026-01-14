"""Tests for task discovery."""

from pathlib import Path

import pytest

from apex_harness.task_discovery import get_all_tasks


def test_get_all_tasks_empty_dir(tmp_path):
    """Test task discovery in empty directory."""
    tasks = get_all_tasks(tmp_path)
    assert tasks == []


def test_get_all_tasks_with_valid_tasks(tmp_path):
    """Test task discovery with valid task directories."""
    (tmp_path / "task1").mkdir()
    (tmp_path / "task2").mkdir()
    (tmp_path / "task3").mkdir()

    tasks = get_all_tasks(tmp_path)

    assert len(tasks) == 3
    assert "task1" in tasks
    assert "task2" in tasks
    assert "task3" in tasks


def test_get_all_tasks_excludes_shared(tmp_path):
    """Test that 'shared' directory is excluded."""
    (tmp_path / "task1").mkdir()
    (tmp_path / "shared").mkdir()
    (tmp_path / "task2").mkdir()

    tasks = get_all_tasks(tmp_path)

    assert len(tasks) == 2
    assert "shared" not in tasks


def test_get_all_tasks_excludes_hidden(tmp_path):
    """Test that hidden directories are excluded."""
    (tmp_path / "task1").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "task2").mkdir()

    tasks = get_all_tasks(tmp_path)

    assert len(tasks) == 2
    assert ".hidden" not in tasks


def test_get_all_tasks_excludes_files(tmp_path):
    """Test that files are excluded."""
    (tmp_path / "task1").mkdir()
    (tmp_path / "somefile.txt").touch()
    (tmp_path / "task2").mkdir()

    tasks = get_all_tasks(tmp_path)

    assert len(tasks) == 2
    assert "somefile.txt" not in tasks


def test_get_all_tasks_sorted(tmp_path):
    """Test that tasks are returned sorted."""
    (tmp_path / "task3").mkdir()
    (tmp_path / "task1").mkdir()
    (tmp_path / "task2").mkdir()

    tasks = get_all_tasks(tmp_path)

    assert tasks == ["task1", "task2", "task3"]


def test_get_all_tasks_nonexistent_dir(tmp_path):
    """Test error handling for nonexistent directory."""
    nonexistent = tmp_path / "nonexistent"

    with pytest.raises(SystemExit):
        get_all_tasks(nonexistent)
