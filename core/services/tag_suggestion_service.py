"""Rule-based and optional LLM-assisted tag suggestion service."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.llm_client import LLMClient
from core.services.tag_analysis_service import TagAnalysisService
from core.services.tag_editor_service import join_tags
from core.services.tag_intelligence_contracts import ANALYSIS_VERSION, TagLlmRefinement, TagSuggestion, TagSuggestionReport


class TagSuggestionService:
    def __init__(self, analysis_service: Optional[TagAnalysisService] = None):
        self.analysis_service = analysis_service or TagAnalysisService()

    def build_suggestions(
        self,
        directory: str,
        *,
        route_family: str = "",
        selected_image_paths: Optional[Sequence[str]] = None,
        analysis_report: Optional[Dict[str, Any]] = None,
        recursive: bool = True,
        caption_extension: str = "",
    ) -> Dict[str, Any]:
        report = analysis_report or self.analysis_service.analyze_dataset(
            directory,
            route_family=route_family,
            recursive=recursive,
            caption_extension=caption_extension,
        )
        items = list(report.get("items", []))
        selected = {
            os.path.normcase(os.path.abspath(str(path)))
            for path in (selected_image_paths or [])
            if str(path or "").strip()
        }
        chosen_items = [
            item
            for item in items
            if not selected or os.path.normcase(os.path.abspath(str(item.get("image_path", "")))) in selected
        ]
        if not chosen_items:
            chosen_items = items[:1]
        tag_counts = Counter()
        for item in items:
            tag_counts.update(item.get("tags", []))
        suggestions: List[TagSuggestion] = []
        route_value = str(report.get("route_family", "generic") or "generic")
        findings_by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for finding in report.get("findings", []):
            image_path = str(finding.get("image_path", "") or "")
            if image_path:
                findings_by_path[image_path].append(finding)

        modal_style = str(report.get("summary", {}).get("caption_style", "mixed") or "mixed")
        average_position = self._average_tag_positions(items)
        for item in chosen_items:
            tags = list(item.get("tags", []))
            current_set = {tag.lower() for tag in tags}
            cooccurrence = Counter()
            support = Counter()
            for peer in items:
                peer_tags = [tag for tag in peer.get("tags", [])]
                peer_set = {tag.lower() for tag in peer_tags}
                overlap = current_set & peer_set
                if not overlap:
                    continue
                for peer_tag in peer_tags:
                    lowered = peer_tag.lower()
                    if lowered in current_set:
                        continue
                    cooccurrence[peer_tag] += len(overlap)
                    support[peer_tag] += 1
            for tag, score in cooccurrence.most_common(5):
                suggestions.append(
                    TagSuggestion(
                        code="missing_likely_tags",
                        message=f"'{tag}' often appears with the current selection.",
                        confidence=min(0.95, 0.45 + support[tag] * 0.08),
                        source="rules",
                        route_family=route_value,
                        image_path=str(item.get("image_path", "")),
                        tags=[tag],
                        evidence={"support": support[tag], "score": score},
                        suggested_action=f"Preview adding '{tag}' if it fits the image.",
                    )
                )
            noisy = [tag for tag in tags if tag_counts.get(tag, 0) <= 1]
            if noisy:
                suggestions.append(
                    TagSuggestion(
                        code="remove_noisy_tags",
                        message="Some tags look like noisy one-offs compared with the dataset.",
                        confidence=0.66,
                        source="rules",
                        route_family=route_value,
                        image_path=str(item.get("image_path", "")),
                        tags=noisy[:6],
                        evidence={"rare_counts": {tag: tag_counts.get(tag, 0) for tag in noisy[:6]}},
                        suggested_action="Review the rare tags before keeping them.",
                    )
                )
            trigger_finding = next(
                (
                    finding
                    for finding in findings_by_path.get(str(item.get("image_path", "")), [])
                    if finding.get("code") == "trigger_word_misuse"
                ),
                None,
            )
            if trigger_finding:
                trigger_words = list(trigger_finding.get("evidence", {}).get("trigger_words", []))
                suggestions.append(
                    TagSuggestion(
                        code="trigger_word_placement",
                        message="Move trigger words toward the front for more predictable prompting behavior.",
                        confidence=0.71,
                        source="rules",
                        route_family=route_value,
                        image_path=str(item.get("image_path", "")),
                        tags=trigger_words,
                        evidence=trigger_finding.get("evidence", {}),
                        suggested_action="Preview a reordered caption with trigger words first.",
                    )
                )
            if route_value in {"sdxl", "anima"} and tags:
                expected = sorted(tags, key=lambda tag: average_position.get(tag.lower(), 9999))
                if expected != tags:
                    suggestions.append(
                        TagSuggestion(
                            code="category_order_suggestion",
                            message="Tag order differs from the dataset's usual ordering.",
                            confidence=0.61,
                            source="rules",
                            route_family=route_value,
                            image_path=str(item.get("image_path", "")),
                            tags=expected[:10],
                            evidence={"preview_order": join_tags(expected[:10])},
                            suggested_action="Preview the suggested order before applying.",
                        )
                    )
            if route_value == "newbie" and modal_style == "natural" and str(item.get("style", "")) == "tag":
                suggestions.append(
                    TagSuggestion(
                        code="route_specific_caption_shape",
                        message="Dataset mostly uses natural captions; this item looks tag-shaped.",
                        confidence=0.78,
                        source="rules",
                        route_family=route_value,
                        image_path=str(item.get("image_path", "")),
                        evidence={"dataset_style": modal_style, "item_style": item.get("style", "")},
                        suggested_action="Rewrite into a natural-sentence caption.",
                    )
                )
            review_findings = [
                finding
                for finding in findings_by_path.get(str(item.get("image_path", "")), [])
                if finding.get("severity") in {"warning", "error"}
            ]
            if review_findings:
                suggestions.append(
                    TagSuggestion(
                        code="needs_review",
                        message="This image already has high-priority findings and should be reviewed first.",
                        confidence=0.92,
                        source="rules",
                        route_family=route_value,
                        image_path=str(item.get("image_path", "")),
                        evidence={"finding_codes": [finding.get("code") for finding in review_findings[:6]]},
                        suggested_action="Open the findings panel and fix the highest-severity issues first.",
                    )
                )

        suggestion_report = TagSuggestionReport(
            version=ANALYSIS_VERSION,
            dataset_path=str(Path(directory).resolve()),
            route_family=route_value,
            dataset_signature=str(report.get("dataset_signature", "")),
            created_at=datetime.now().isoformat(),
            status="ready",
            suggestions=[asdict(suggestion) for suggestion in suggestions],
            summary={
                "selected_count": len(chosen_items),
                "suggestion_count": len(suggestions),
                "analysis_created_at": report.get("created_at"),
            },
        )
        return asdict(suggestion_report)

    async def llm_refine(
        self,
        *,
        directory: str,
        route_family: str = "",
        selected_image_paths: Optional[Sequence[str]] = None,
        analysis_report: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        prompt: str = "",
    ) -> Dict[str, Any]:
        deterministic = self.build_suggestions(
            directory,
            route_family=route_family,
            selected_image_paths=selected_image_paths,
            analysis_report=analysis_report,
        )
        if not str(api_key or "").strip():
            return {
                "status": "unavailable",
                "message": "No LLM API key configured.",
                "source": "llm",
                "deterministic": deterministic,
                "refinement": asdict(TagLlmRefinement(status="unavailable", warnings=["No API key configured."])),
                "suggestions": [],
            }
        chosen = deterministic.get("suggestions", [])[:6]
        summary = deterministic.get("summary", {})
        messages = [
            {"role": "system", "content": "You refine tag suggestions for image-caption datasets. Never replace the deterministic base. Return strict JSON with keys: add_tags, remove_tags, caption_rewrite, style_cleanup, explanation, warnings."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "dataset_path": str(Path(directory).resolve()),
                        "route_family": route_family or "generic",
                        "summary": summary,
                        "selected_image_paths": list(selected_image_paths or []),
                        "deterministic_suggestions": chosen,
                        "prompt": prompt or "Suggest a concise preview-only caption rewrite or cleanup note.",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        client = LLMClient(provider=provider or "openai", api_key=api_key, base_url=base_url, model=model)
        response_text = await client.chat(messages)
        refinement = self._parse_llm_refinement(response_text)
        return {
            "status": "ready",
            "message": "LLM refinement generated.",
            "source": "llm",
            "deterministic": deterministic,
            "refinement": asdict(refinement),
            "suggestions": [
                {
                    "code": "llm_refinement_structured",
                    "message": refinement.explanation or str(response_text or "").strip(),
                    "confidence": 0.55,
                    "source": "llm",
                    "route_family": route_family or "generic",
                    "image_path": list(selected_image_paths or [""])[0] if selected_image_paths else "",
                    "tags": list(refinement.add_tags),
                    "evidence": {
                        "remove_tags": refinement.remove_tags,
                        "style_cleanup": refinement.style_cleanup,
                        "warnings": refinement.warnings,
                        "caption_rewrite": refinement.caption_rewrite,
                    },
                    "suggested_action": "Preview the structured refinement before applying any edit.",
                }
            ],
        }

    def _parse_llm_refinement(self, response_text: Any) -> TagLlmRefinement:
        raw = str(response_text or "").strip()
        if not raw:
            return TagLlmRefinement(status="empty", warnings=["LLM returned an empty response."])
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            payload = json.loads(raw[start : end + 1] if start != -1 and end != -1 else raw)
            if not isinstance(payload, dict):
                raise ValueError("LLM response is not a JSON object")
            return TagLlmRefinement(
                status="ready",
                add_tags=[str(tag).strip() for tag in payload.get("add_tags", []) if str(tag).strip()],
                remove_tags=[str(tag).strip() for tag in payload.get("remove_tags", []) if str(tag).strip()],
                caption_rewrite=str(payload.get("caption_rewrite", "") or "").strip(),
                style_cleanup=[str(item).strip() for item in payload.get("style_cleanup", []) if str(item).strip()],
                explanation=str(payload.get("explanation", "") or "").strip(),
                warnings=[str(item).strip() for item in payload.get("warnings", []) if str(item).strip()],
                raw_text=raw,
            )
        except Exception:
            return TagLlmRefinement(status="fallback", explanation=raw, raw_text=raw, warnings=["LLM response was not valid JSON."])

    def _average_tag_positions(self, items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for item in items:
            for idx, tag in enumerate(item.get("tags", [])):
                lowered = str(tag).lower()
                totals[lowered] += idx
                counts[lowered] += 1
        return {tag: totals[tag] / counts[tag] for tag in totals if counts[tag]}
