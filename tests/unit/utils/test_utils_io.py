from pathlib import Path

import pytest
from lwrepcoupling import SecurityViolationError
from lwrepcoupling._utils import assert_path_is_safe_for_purging


def test_assert_path_is_safe_with_valid_deep_path() -> None:
    """Ensures a safe, deeply nested local project folder passes checks cleanly."""
    safe_path = Path("/home/user/workspace/project/runs/b_0")
    # Should run with no exception raised
    assert_path_is_safe_for_purging(safe_path)


def test_assert_path_rejects_system_roots() -> None:
    """Ensures critical system paths immediately trigger security violations."""
    root_path = Path("/")
    with pytest.raises(SecurityViolationError, match="directly matches a protected system"):
        assert_path_is_safe_for_purging(root_path)


def test_assert_path_rejects_shallow_directories() -> None:
    """Ensures dangerously shallow paths (depth < 3) are blocked to protect data."""
    shallow_path = Path("/home")  # Length of parts is 2
    with pytest.raises(SecurityViolationError, match="too close to the file system root"):
        assert_path_is_safe_for_purging(shallow_path)
