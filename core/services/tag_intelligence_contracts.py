# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Contracts for deterministic tag intelligence services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


ANALYSIS_VERSION = "tag-tooling-v1"


@dataclass
class TagFinding:
    severity: str
    code: str
    message: str
    image_path: str = ""
    route_family: str = "generic"
    related_tags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    confidence: float = 0.5


@dataclass
class TagAnalysisReport:
    version: str
    dataset_path: str
    route_family: str
    dataset_signature: str
    created_at: str
    summary: Dict[str, Any]
    findings: List[Dict[str, Any]]
    findings_by_severity: Dict[str, int]
    review_queues: Dict[str, List[str]]
    route_audit: Dict[str, Any]
    tag_distribution: Dict[str, Any]
    token_budget: Dict[str, Any]
    items: List[Dict[str, Any]]


@dataclass
class TagSuggestion:
    code: str
    message: str
    confidence: float
    source: str
    route_family: str
    image_path: str = ""
    tags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""


@dataclass
class TagSuggestionReport:
    version: str
    dataset_path: str
    route_family: str
    dataset_signature: str
    created_at: str
    status: str
    suggestions: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class TagLlmRefinement:
    status: str
    source: str = "llm"
    add_tags: List[str] = field(default_factory=list)
    remove_tags: List[str] = field(default_factory=list)
    caption_rewrite: str = ""
    style_cleanup: List[str] = field(default_factory=list)
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class CaptionTagPipelineReport:
    version: str
    dataset_path: str
    route_family: str
    dataset_signature: str
    created_at: str
    analysis: Dict[str, Any]
    suggestions: Dict[str, Any]
    cleanup_preview: Dict[str, Any]
    backup_policy: Dict[str, Any]
    apply_status: Dict[str, Any]
