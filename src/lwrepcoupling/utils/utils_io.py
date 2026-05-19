import os
from pathlib import Path


class SecurityViolationError(PermissionError):
    """Raised when a directory path breaches systemic filesystem safety protocols.

    Inherits from the standard PermissionError to remain compatible with generic
    system access catch blocks, but adds granular tracking properties for debugging.

    Attributes:
        target_path: The absolute Path that triggered the security block.
        violation_reason: A string explaining which guardrail rule was breached.
    """

    def __init__(self, target_path: Path, violation_reason: str):
        self.target_path = target_path
        self.violation_reason = violation_reason
        # Generate a crystal-clear, standard message for logs and console printouts
        self.message = (
            f"Security Violation: Access denied to path '{self.target_path}'. "
            f"Reason: {self.violation_reason}."
        )
        super().__init__(self.message)


class WorkspaceConflictError(FileExistsError):
    """Raised when a target workspace directory is occupied and cannot be modified.

    Inherits from standard FileExistsError.

    Attributes:
        target_path: The absolute Path to the blocked directory.
    """

    def __init__(self, target_path: Path, message: str):
        self.target_path = target_path
        super().__init__(message)


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
            violation_reason=f"Path is too close to the file system root (directory depth: {len(resolved_path.parts)})",
        )
