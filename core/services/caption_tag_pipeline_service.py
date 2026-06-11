"""Preview-only caption/tag pipeline report service."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from core.services.tag_analysis_service import TagAnalysisService
from core.services.tag_editor_service import TagEditorService
from core.services.tag_intelligence_contracts import ANALYSIS_VERSION, CaptionTagPipelineReport
from core.services.tag_suggestion_service import TagSuggestionService


class CaptionTagPipelineService:
    def __init__(
        self,
        analysis_service: Optional[TagAnalysisService] = None,
        suggestion_service: Optional[TagSuggestionService] = None,
        tag_editor: Optional[TagEditorService] = None,
    ):
        self.analysis_service = analysis_service or TagAnalysisService(tag_editor)
        self.suggestion_service = suggestion_service or TagSuggestionService(self.analysis_service)
        self.tag_editor = tag_editor or self.analysis_service.tag_editor

    def build_report(
        self,
        directory: str,
        *,
        route_family: str = "",
        recursive: bool = True,
        caption_extension: str = "",
        max_token_count: int = 75,
        trigger_words: Optional[Sequence[str]] = None,
        selected_image_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        analysis = self.analysis_service.analyze_dataset(
            directory,
            route_family=route_family,
            recursive=recursive,
            caption_extension=caption_extension,
            max_token_count=max_token_count,
            trigger_words=trigger_words,
        )
        suggestions = self.suggestion_service.build_suggestions(
            directory,
            route_family=route_family,
            selected_image_paths=selected_image_paths,
            analysis_report=analysis,
            recursive=recursive,
            caption_extension=caption_extension,
        )
        cleanup_preview = self.tag_editor.preview_batch_action(
            directory,
            action="dedupe_tags",
            recursive=recursive,
            caption_extension=caption_extension,
        )
        cleanup_preview["dry_run"] = True
        cleanup_preview["actions"] = ["dedupe_tags"]
        report = CaptionTagPipelineReport(
            version=ANALYSIS_VERSION,
            dataset_path=str(Path(directory).resolve()),
            route_family=str(analysis.get("route_family", "generic") or "generic"),
            dataset_signature=str(analysis.get("dataset_signature", "")),
            created_at=datetime.now().isoformat(),
            analysis={
                "status": "ready",
                "summary": dict(analysis.get("summary", {}) or {}),
                "findings_by_severity": dict(analysis.get("findings_by_severity", {}) or {}),
                "finding_count": len(analysis.get("findings", []) or []),
                "route_audit": dict(analysis.get("route_audit", {}) or {}),
                "token_budget": dict(analysis.get("token_budget", {}) or {}),
                "tag_distribution": dict(analysis.get("tag_distribution", {}) or {}),
            },
            suggestions={
                "status": suggestions.get("status", "ready"),
                "summary": dict(suggestions.get("summary", {}) or {}),
                "suggestions": list(suggestions.get("suggestions", []) or [])[:50],
            },
            cleanup_preview=cleanup_preview,
            backup_policy={
                "required_for_apply": True,
                "default_create_backup": True,
                "allow_explicit_no_backup": True,
            },
            apply_status={"status": "not_applied", "mutated": False, "requires_preview": True},
        )
        return asdict(report)
