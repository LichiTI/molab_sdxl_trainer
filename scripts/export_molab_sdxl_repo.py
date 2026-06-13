#!/usr/bin/env python3
"""Export a standalone SDXL LoRA molab repository from the full Lulynx tree.

Run from the original repository root:
    python molab_sdxl_trainer/scripts/export_molab_sdxl_repo.py --out dist/lulynx-sdxl-molab
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD = ROOT / "molab_sdxl_trainer"

EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "docs",
    "examples",
    "devtools",
    "warehouse",
    "data",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def _ignore_runtime(dir_path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(dir_path) / name
        if name in EXCLUDED_DIRS:
            ignored.add(name)
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            ignored.add(name)
            continue
        if name.endswith(".log") or name.endswith(".tmp"):
            ignored.add(name)
            continue
    return ignored


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_runtime)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Runtime core. Keep the whole core package for compatibility; SDXL LoRA is deeply coupled
    # to shared config/training/dataset/optimizer modules.
    _copytree(ROOT / "backend" / "core", out / "core")

    # Molab helper files.
    _copytree(SCAFFOLD / "configs", out / "configs")
    _copytree(SCAFFOLD / "notebooks", out / "notebooks")
    _copytree(SCAFFOLD / "scripts", out / "scripts")
    for req_name in (
        "requirements-molab.txt",
        "requirements-blackwell-cu128.txt",
        "requirements-sageattention.txt",
    ):
        src_req = SCAFFOLD / req_name
        if src_req.is_file():
            _copy_file(src_req, out / req_name)

    # Replace scaffold export script copy with a note? Keep it; useful for future refresh.
    readme = (SCAFFOLD / "README.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(readme, encoding="utf-8")

    (out / "pyproject.toml").write_text(
        """[project]\nname = \"lulynx-sdxl-molab\"\nversion = \"0.1.0\"\ndescription = \"Standalone SDXL LoRA training subset exported from Lulynx Trainer for molab.\"\nrequires-python = \">=3.10\"\n\n[tool.setuptools.packages.find]\nwhere = [\".\"]\ninclude = [\"core*\"]\n""",
        encoding="utf-8",
    )
    (out / ".gitignore").write_text(
        """__pycache__/\n*.py[cod]\n.venv/\n.env\nwork/models/\nwork/datasets/\nwork/outputs/\nwork/logs/\nwork/runs/\n*.safetensors\n*.ckpt\n*.pt\n*.pth\n""",
        encoding="utf-8",
    )
    for rel in ["work/models", "work/datasets", "work/outputs", "work/logs", "work/runs"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
        (out / rel / ".gitkeep").write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export standalone molab SDXL LoRA repo")
    parser.add_argument("--out", default="dist/lulynx-sdxl-molab", help="Output directory")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    export(out)
    print(f"Exported molab SDXL LoRA repo to: {out}")
    print("Next:")
    print(f"  cd {out}")
    print("  git init && git add . && git commit -m 'Initial SDXL LoRA molab trainer'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
