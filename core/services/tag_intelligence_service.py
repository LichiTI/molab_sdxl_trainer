"""Compatibility exports for tag intelligence services.

Concrete owners now live in focused modules:
`tag_analysis_service`, `tag_suggestion_service`, `caption_tag_pipeline_service`,
`tag_intelligence_contracts`, and `tag_job_store`.
"""

from __future__ import annotations

from core.services.caption_tag_pipeline_service import CaptionTagPipelineService
from core.services.tag_analysis_service import TagAnalysisService
from core.services.tag_intelligence_contracts import (
    ANALYSIS_VERSION,
    CaptionTagPipelineReport,
    TagAnalysisReport,
    TagFinding,
    TagLlmRefinement,
    TagSuggestion,
    TagSuggestionReport,
)
from core.services.tag_job_store import TagJobStore
from core.services.tag_suggestion_service import TagSuggestionService


__all__ = [
    "ANALYSIS_VERSION",
    "CaptionTagPipelineReport",
    "CaptionTagPipelineService",
    "TagAnalysisReport",
    "TagAnalysisService",
    "TagFinding",
    "TagJobStore",
    "TagLlmRefinement",
    "TagSuggestion",
    "TagSuggestionReport",
    "TagSuggestionService",
]
