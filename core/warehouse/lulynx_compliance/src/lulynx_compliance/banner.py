"""Runtime startup banner rendering.

Produces multi-line console banners that display project identity,
license, and route contract information at training launch time.
"""

from __future__ import annotations

from typing import Callable

from lulynx_route_contract import classify_route


def render_banner(
    *,
    project_name: str,
    project_version: str,
    repo_url: str,
    license_name: str,
    copyright_notice: str,
    training_type: str | None = None,
    route_kind: str | None = None,
    route_label: str | None = None,
    git_commit: str | None = None,
    runtime_mode: str | None = None,
    entry_point: str | None = None,
    extra_lines: list[str] | None = None,
) -> list[str]:
    """Build a list of banner lines for console output.

    Each line is a self-contained string suitable for ``print()``.
    The banner includes project identity, license, compliance notice
    in both English and Chinese, and the resolved route contract.
    """
    contract = classify_route(
        training_type or "",
        kind_override=route_kind,
        label_override=route_label,
    )
    lines: list[str] = [
        f"{project_name} {project_version}",
        f"Source: {repo_url}",
        f"License: {license_name}",
        f"Copyright: {copyright_notice}",
        (
            "Compliance: modified builds and hosted services must provide "
            "corresponding source and preserve notices."
        ),
        (
            "合规提示：修改版或通过网络向他人提供服务的版本，应提供对应源码并保留来源声明。"
        ),
        f"Route: {contract.label} [{contract.kind}]",
        f"Capabilities: {', '.join(contract.capability_tags)}",
    ]
    if git_commit:
        lines.append(f"Commit: {git_commit}")
    if runtime_mode:
        lines.append(f"Runtime: {runtime_mode}")
    if entry_point:
        lines.append(f"Entry: {entry_point}")
    for extra in extra_lines or []:
        lines.append(str(extra))
    return lines


def print_banner(
    printer: Callable[[str], None] | None = None,
    **kwargs,
) -> None:
    """Render and emit the banner via *printer* (defaults to ``print``)."""
    emit = printer or print
    for line in render_banner(**kwargs):
        emit(line)
