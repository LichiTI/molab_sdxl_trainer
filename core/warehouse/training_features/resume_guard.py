"""Resume Guard — detects checkpoint state for safe training resume.

Scans checkpoint directories, identifies the latest valid checkpoint,
validates state directory completeness, and detects output artifact
collisions.  Pure-stdlib, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Checkpoint file extensions ordered by preference.
_CHECKPOINT_EXTS: tuple[str, ...] = (".safetensors", ".pt", ".pth", ".ckpt", ".bin")
_STATE_ARTIFACT_EXTS: frozenset[str] = frozenset({".safetensors", ".ckpt", ".pt", ".pth", ".bin"})

# Matches directory or file names like ``step-1000``, ``steps_5000``,
# ``checkpoint-200``, ``epoch-3``, or plain numeric ``0005``.
_STEP_RE = re.compile(
    r"(?:step[s_-]*|checkpoint[s_-]*|epoch[s_-]*|ckpt[s_-]*)(\d+)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"^(\d+)$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointEntry:
    """A single discovered checkpoint."""

    path: Path
    step: int | None
    size_bytes: int


@dataclass
class ResumeReport:
    """Outcome of a checkpoint directory scan."""

    found: bool
    latest: CheckpointEntry | None = None
    all_checkpoints: list[CheckpointEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaunchGuardReport:
    """Outcome of a pre-launch resume guard check."""

    ok: bool
    message: str = ""
    state_complete: bool | None = None
    output_collision: bool = False


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

def _extract_step(name: str) -> int | None:
    """Try to extract a numeric step/epoch from a file or directory name."""
    m = _STEP_RE.search(name)
    if m:
        return int(m.group(1))
    m = _NUM_RE.match(name)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _scan_directory(directory: Path) -> list[CheckpointEntry]:
    """Return checkpoint files directly inside *directory* (non-recursive)."""
    entries: list[CheckpointEntry] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _CHECKPOINT_EXTS:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        entries.append(CheckpointEntry(path=p, step=_extract_step(p.stem), size_bytes=size))
    return sorted(entries, key=lambda e: (e.step is not None, e.step or 0))


def _scan_nested(directory: Path) -> list[CheckpointEntry]:
    """Scan one level of numbered/named subdirectories for checkpoints."""
    entries: list[CheckpointEntry] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        if _extract_step(child.name) is None:
            continue
        child_entries = _scan_directory(child)
        for e in child_entries:
            if e.step is None:
                entries.append(CheckpointEntry(path=e.path, step=_extract_step(child.name), size_bytes=e.size_bytes))
            else:
                entries.append(e)
    return sorted(entries, key=lambda e: (e.step is not None, e.step or 0))


# ---------------------------------------------------------------------------
# State directory completeness
# ---------------------------------------------------------------------------

# Files that should always exist in a valid save_state directory.
_STATE_REQUIRED_FILES: tuple[str, ...] = ("train_state.json", "optimizer.bin")
_STATE_MODEL_CANDIDATES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin", "model.bin")


def check_state_dir_complete(state_dir: Path) -> tuple[bool, str]:
    """Verify that a save_state directory has the minimum required files.

    Returns ``(True, "")`` when complete, ``(False, reason)`` when not.
    """
    if not state_dir.is_dir():
        return False, "state path is not a directory"

    for name in _STATE_REQUIRED_FILES:
        if not (state_dir / name).is_file():
            return False, f"missing {name}"

    # At least one model state file should exist.
    has_model = any((state_dir / name).is_file() for name in _STATE_MODEL_CANDIDATES)
    if not has_model:
        # Also check glob patterns for multi-shard saves.
        has_model = bool(list(state_dir.glob("pytorch_model*.bin")))
    if not has_model:
        has_model = bool(list(state_dir.glob("model*.safetensors")))
    if not has_model:
        return False, "missing model state file"

    return True, ""


def check_state_dir_complete_with_scheduler(
    state_dir: Path,
    *,
    scheduler_optional: bool = False,
) -> tuple[bool, str]:
    """Like :func:`check_state_dir_complete` but also checks for scheduler.bin.

    When *scheduler_optional* is True (e.g. schedulefree or constant-only
    runs), the scheduler.bin check is skipped.
    """
    ok, reason = check_state_dir_complete(state_dir)
    if not ok:
        return ok, reason
    if not scheduler_optional and not (state_dir / "scheduler.bin").is_file():
        return False, "missing scheduler.bin"
    return True, ""


# ---------------------------------------------------------------------------
# Output artifact collision detection
# ---------------------------------------------------------------------------

def _name_matches_output(filename: str, output_name: str) -> bool:
    """Check if *filename* matches a training output artifact for *output_name*.

    Training outputs files like ``{name}.safetensors``, ``{name}-000004.safetensors``,
    ``{name}-e5.safetensors`` and directories like ``{name}-state``, ``{name}-000004-state``.
    """
    if not output_name:
        return True
    if not filename.startswith(output_name):
        return False
    rest = filename[len(output_name):]
    if not rest:
        return True
    return rest[0] in ("-", ".")


def scan_output_artifacts(output_dir: Path, *, output_name: str = "") -> list[Path]:
    """Return existing training artifacts (checkpoints, state dirs) in *output_dir*.

    If *output_name* is provided, only artifacts whose names match are returned.
    """
    if not output_dir.is_dir():
        return []

    artifacts: list[Path] = []
    for child in output_dir.iterdir():
        if child.is_file():
            if child.suffix.lower() not in _STATE_ARTIFACT_EXTS:
                continue
            if not _name_matches_output(child.name, output_name):
                continue
            artifacts.append(child)
        elif child.is_dir() and child.name.endswith("-state"):
            if not _name_matches_output(child.name, output_name):
                continue
            artifacts.append(child)

    return artifacts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ResumeGuard:
    """Scans a checkpoint directory and determines resume readiness.

    Stateless after construction.  Call ``scan()`` with the checkpoint
    directory path to obtain a :class:`ResumeReport`.
    """

    def __init__(
        self,
        *,
        min_step: int = 0,
        min_size_bytes: int = 1,
        recursive: bool = True,
    ) -> None:
        self._min_step = min_step
        self._min_size = min_size_bytes
        self._recursive = recursive

    def scan(self, checkpoint_dir: Path) -> ResumeReport:
        """Scan *checkpoint_dir* and return a resume report."""
        checkpoint_dir = Path(checkpoint_dir)
        report = ResumeReport(found=False)

        if not checkpoint_dir.is_dir():
            report.errors.append(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return report

        entries = _scan_directory(checkpoint_dir)
        if self._recursive and not entries:
            entries = _scan_nested(checkpoint_dir)

        # Apply filters
        filtered = [
            e for e in entries
            if e.size_bytes >= self._min_size
            and (e.step is None or e.step >= self._min_step)
        ]

        if not filtered:
            report.warnings.append("No valid checkpoints found")
            report.all_checkpoints = entries
            return report

        latest = filtered[-1]
        report.found = True
        report.latest = latest
        report.all_checkpoints = filtered
        report.metadata = {
            "total_found": len(filtered),
            "latest_step": latest.step,
            "latest_path": str(latest.path),
            "latest_size_bytes": latest.size_bytes,
        }
        return report

    def latest_checkpoint(self, checkpoint_dir: Path) -> Path | None:
        """Convenience: return the path of the latest checkpoint, or None."""
        return self.scan(checkpoint_dir).latest.path

    def validate_state_dir(
        self,
        state_dir: Path,
        *,
        scheduler_optional: bool = False,
    ) -> LaunchGuardReport:
        """Validate that a resume state directory is complete and safe to use."""
        ok, reason = check_state_dir_complete_with_scheduler(
            Path(state_dir),
            scheduler_optional=scheduler_optional,
        )
        return LaunchGuardReport(ok=ok, message=reason, state_complete=ok)

    def check_output_collision(
        self,
        output_dir: Path,
        *,
        output_name: str = "",
        resume_path: Path | None = None,
    ) -> LaunchGuardReport:
        """Check whether the output directory already contains training artifacts.

        If artifacts exist and no resume path is configured, returns a guard
        failure so the caller can warn the user.
        """
        artifacts = scan_output_artifacts(Path(output_dir), output_name=output_name)
        if not artifacts:
            return LaunchGuardReport(ok=True, output_collision=False)

        if resume_path is not None:
            return LaunchGuardReport(ok=True, output_collision=True)

        return LaunchGuardReport(
            ok=False,
            message=(
                "Output directory already contains training artifacts but no "
                "resume path is configured. Set resume to a save_state directory, "
                "or change output_name / output_dir."
            ),
            output_collision=True,
        )
