"""Centralized exception hierarchy for the long-wave radiation coupling engine."""

from pathlib import Path

class SimulationCrashError(RuntimeError):
    """Raised when the isolated long-wave radiation simulation crashes."""


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