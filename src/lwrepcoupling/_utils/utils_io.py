import os
from pathlib import Path

from ..exceptions import SecurityViolationError


def assert_path_is_safe_for_purging(target_path: Path) -> None:
    """Validates that a directory path is safe to clear without risking system stability.

    Args:
        target_path: The absolute Path directory targeted for cleanup.

    Raises:
        SecurityViolationError: If the path matches an OS root or core user folder.
    """
    resolved_path = target_path.resolve()

    # Dynamic evaluation of critical system landmarks on the host machine
    protected_paths = {
        Path(resolved_path.anchor).resolve(),
        Path.home().resolve(),
        Path.home().resolve() / "Desktop",
        Path.home().resolve() / "Documents",
        Path.home().resolve() / "Downloads",
    }

    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            protected_paths.add(Path(program_files).resolve())
    else:
        protected_paths.update(
            {
                Path("/usr").resolve(),
                Path("/var").resolve(),
                Path("/etc").resolve(),
                Path("/opt").resolve(),
            }
        )

    # Guardrail A: Reject exact matches to any protected directory
    if resolved_path in protected_paths:
        raise SecurityViolationError(
            target_path=resolved_path,
            violation_reason="Path directly matches a protected system or user profile directory",
        )

    # Guardrail B: Reject dangerously shallow paths (e.g., Less than 3 levels deep)
    if len(resolved_path.parts) < 3:
        raise SecurityViolationError(
            target_path=resolved_path,
            violation_reason=f"Path is too close to the file system root "
            f"(directory depth: {len(resolved_path.parts)})",
        )
