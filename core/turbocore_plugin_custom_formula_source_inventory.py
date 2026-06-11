"""Source-level state inventory for selected custom-formula plugin optimizers."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPTIMIZER_SOURCE_ROOT = _REPO_ROOT / "plugin" / "pytorch_optimizer-main" / "pytorch_optimizer" / "optimizer"


def build_custom_formula_state_inventory_artifact(
    name: str,
    artifacts: Mapping[str, str],
) -> dict[str, Any] | None:
    """Build a report-only state inventory from the vendored plugin source."""

    inventory = custom_formula_source_inventory(name)
    if inventory.get("status") != "ready":
        return None
    return {
        "schema_version": 1,
        "artifact": str(artifacts["state_inventory"]),
        "status": "ready",
        "report_only": True,
        "source_review_target": f"pytorch_optimizer:{name}",
        "source_scan_status": "ast_source_inventory_ready",
        "source_file": inventory["source_file"],
        "source_class": inventory["source_class"],
        "param_state_keys": inventory["param_state_keys"],
        "optional_or_lazy_state_keys": inventory["optional_or_lazy_state_keys"],
        "param_group_state_keys": inventory["param_group_state_keys"],
        "hparam_surface_keys": inventory["hparam_surface_keys"],
        "state_dict_key_inventory": inventory["state_dict_key_inventory"],
        "closure_supported": inventory["closure_supported"],
        "sparse_gradient_policy": inventory["sparse_gradient_policy"],
        "complex_tensor_policy": inventory["complex_tensor_policy"],
        "native_kernel_ready": False,
        "formula_parity_status": "pending",
        "resume_parity_status": "pending",
    }


def custom_formula_state_inventory_ready(name: str) -> bool:
    return custom_formula_source_inventory(name).get("status") == "ready"


def custom_formula_source_inventory(name: str) -> dict[str, Any]:
    normalized = str(name).strip().lower()
    source = _optimizer_source_index().get(normalized)
    if source is None:
        return {
            "schema_version": 1,
            "status": "missing",
            "optimizer_name": normalized,
            "blocked_reasons": ["pytorch_optimizer_source_class_missing"],
        }
    cls, path, class_node, text = source
    assigned_state, accessed_state = _state_keys(class_node)
    group_keys, default_keys = _group_and_default_keys(class_node)
    param_group_state = ["group.step"] if "step" in group_keys else []
    hparams = sorted((group_keys | default_keys) - {"params", "step"})
    optional_or_lazy = sorted(accessed_state - assigned_state)
    state_dict_keys = [f"state.{key}" for key in sorted(assigned_state)] + param_group_state
    return {
        "schema_version": 1,
        "status": "ready",
        "optimizer_name": normalized,
        "source_file": _relative(path),
        "source_class": cls,
        "param_state_keys": sorted(assigned_state),
        "optional_or_lazy_state_keys": optional_or_lazy,
        "param_group_state_keys": param_group_state,
        "hparam_surface_keys": hparams,
        "state_dict_key_inventory": state_dict_keys,
        "closure_supported": _step_accepts_closure(class_node),
        "sparse_gradient_policy": "reject_sparse_grad" if "NoSparseGradientError" in text else "source_check_missing",
        "complex_tensor_policy": "view_as_real" if "view_as_real" in text else "no_explicit_complex_view",
    }


@lru_cache(maxsize=1)
def _optimizer_source_index() -> dict[str, tuple[str, Path, ast.ClassDef, str]]:
    out: dict[str, tuple[str, Path, ast.ClassDef, str]] = {}
    if not _OPTIMIZER_SOURCE_ROOT.exists():
        return out
    for path in sorted(_OPTIMIZER_SOURCE_ROOT.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out[node.name.lower()] = (node.name, path, node, text)
    return out


def _state_keys(class_node: ast.ClassDef) -> tuple[set[str], set[str]]:
    assigned: set[str] = set()
    accessed: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Subscript) or not _is_name(node.value, "state"):
            continue
        key = _literal_key(node.slice)
        if key is None:
            continue
        accessed.add(key)
        if isinstance(node.ctx, ast.Store):
            assigned.add(key)
    return assigned, accessed


def _group_and_default_keys(class_node: ast.ClassDef) -> tuple[set[str], set[str]]:
    group_keys: set[str] = set()
    default_keys: set[str] = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Subscript) and _is_name(node.value, "group"):
            key = _literal_key(node.slice)
            if key is not None:
                group_keys.add(key)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _is_name(node.func.value, "group") and node.func.attr == "get" and node.args:
                key = _literal_key(node.args[0])
                if key is not None:
                    group_keys.add(key)
        elif isinstance(node, ast.Assign) and any(_is_name(target, "defaults") for target in node.targets):
            default_keys.update(_dict_literal_keys(node.value))
    return group_keys, default_keys


def _step_accepts_closure(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "step":
            continue
        return any(arg.arg == "closure" for arg in node.args.args + node.args.kwonlyargs)
    return False


def _dict_literal_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {key for item in node.keys if (key := _literal_key(item)) is not None}


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _relative(path: Path) -> str:
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "build_custom_formula_state_inventory_artifact",
    "custom_formula_source_inventory",
    "custom_formula_state_inventory_ready",
]
