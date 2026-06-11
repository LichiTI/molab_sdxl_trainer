"""Deterministic dataset analysis service for tag tooling."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.services.tag_editor_service import DatasetCaptionItem, TagEditorService, split_tags
from core.services.tag_intelligence_contracts import ANALYSIS_VERSION, TagAnalysisReport, TagFinding


_SENTENCE_PUNCTUATION = {".", "!", "?", "。", "！", "？"}
_CONTRADICTION_PAIRS = [
    ("day", "night"),
    ("indoors", "outdoors"),
    ("solo", "group"),
    ("smile", "frown"),
    ("happy", "sad"),
    ("1girl", "1boy"),
]


class TagAnalysisService:
    def __init__(self, tag_editor: Optional[TagEditorService] = None):
        self.tag_editor = tag_editor or TagEditorService()

    def compute_dataset_signature(self, directory: str, *, recursive: bool = True, caption_extension: str = "") -> str:
        dataset_dir = Path(directory)
        items = self.tag_editor._scan_dataset(
            dataset_dir,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=False,
        )
        parts = [
            f"{item.relative_path}|{int(item.mtime)}|{int(item.caption_path.stat().st_mtime) if item.caption_path.exists() else 0}|{len(item.caption_text)}"
            for item in items
        ]
        return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()

    # --- Dual-Mode Lint Engine ---

    def lint_dataset_lightweight(
        self,
        directory: str,
        *,
        route_family: str = "",
        recursive: bool = True,
        caption_extension: str = "",
        load_caption_from_filename: bool = False,
        filename_regex: str = "",
        filename_joiner: str = ", ",
        max_token_count: int = 75,
        trigger_words: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Lightweight lint: runs synchronously and returns the full report for small datasets."""
        return self.analyze_dataset(
            directory,
            route_family=route_family,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=load_caption_from_filename,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
            max_token_count=max_token_count,
            trigger_words=trigger_words,
        )

    def lint_dataset_heavyweight(
        self,
        directory: str,
        *,
        job_id: str,
        route_family: str = "",
        recursive: bool = True,
        caption_extension: str = "",
        load_caption_from_filename: bool = False,
        filename_regex: str = "",
        filename_joiner: str = ", ",
        max_token_count: int = 75,
        trigger_words: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Heavyweight lint: runs and persists the result to the TagJobStore."""
        report = self.analyze_dataset(
            directory,
            route_family=route_family,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=load_caption_from_filename,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
            max_token_count=max_token_count,
            trigger_words=trigger_words,
        )
        from core.services.tag_job_store import TagJobStore
        store = TagJobStore()
        store.save_result(
            kind="tag_lint",
            job_id=job_id,
            dataset_path=directory,
            route_family=report.get("route_family", "generic"),
            submitted_config={
                "route_family": route_family,
                "recursive": recursive,
                "max_token_count": max_token_count,
            },
            payload=report,
        )
        return report

    # --- /Dual-Mode Lint Engine ---

    def analyze_dataset(
        serialized_items: List[Dict[str, Any]] = []
        findings: List[TagFinding] = []
        tag_counts: Counter[str] = Counter()
        style_counts: Counter[str] = Counter()
        delimiter_counts: Counter[int] = Counter()
        token_counts: List[int] = []
        tags_per_item: List[int] = []
        trigger_set = {str(tag).strip().lower() for tag in (trigger_words or []) if str(tag).strip()}

        for item in items:
            token_info = self.tag_editor.analyze_caption_tokens(item.caption_text, max_token_count=max_token_count)
            style = self._caption_style(item.caption_text)
            delimiter_count = item.caption_text.count("|")
            style_counts.update([style])
            delimiter_counts.update([delimiter_count])
            token_counts.append(int(token_info["token_count"]))
            tags_per_item.append(len(item.tags))
            tag_counts.update(item.tags)
            serialized_items.append(
                {
                    "image_path": str(item.image_path),
                    "relative_path": item.relative_path,
                    "caption": item.caption_text,
                    "caption_source": item.caption_source,
                    "caption_exists": item.caption_exists,
                    "tags": list(item.tags),
                    "token_count": int(token_info["token_count"]),
                    "style": style,
                    "delimiter_count": delimiter_count,
                }
            )

        modal_style = style_counts.most_common(1)[0][0] if style_counts else "empty"
        modal_delimiters = delimiter_counts.most_common(1)[0][0] if delimiter_counts else 0
        avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
        avg_tag_count = (sum(tags_per_item) / len(tags_per_item)) if tags_per_item else 0.0
        coverage = (sum(1 for item in items if item.caption_text) / len(items)) if items else 0.0
        token_density = (avg_tokens / avg_tag_count) if avg_tag_count > 0 else 0.0
        long_threshold = max(25, int(avg_tokens * 1.75)) if token_counts else 25
        short_threshold = max(2, int(avg_tokens * 0.3)) if token_counts else 2

        typo_pairs = self._find_typo_pairs(tag_counts)
        for item in items:
            token_info = self.tag_editor.analyze_caption_tokens(item.caption_text, max_token_count=max_token_count)
            tags = item.tags
            lower_tags = [tag.lower() for tag in tags]
            duplicates = [tag for tag, count in Counter(lower_tags).items() if count > 1]
            if duplicates:
                findings.append(self._finding("warning", "duplicate_tags", "Caption contains duplicate tags.", item, resolved_route, duplicates, {"duplicates": duplicates}, "Run dedupe_tags.", 0.99))
            if item.caption_exists and not item.caption_text:
                findings.append(self._finding("warning", "empty_caption", "Caption file exists but is empty.", item, resolved_route, [], {}, "Fill in or remove the empty caption file.", 0.98))
            if not item.caption_exists and not item.caption_text:
                findings.append(self._finding("warning", "missing_caption", "Image has no caption file.", item, resolved_route, [], {}, "Add a caption or derive one before training.", 0.96))
            if token_info["over_limit"]:
                findings.append(self._finding("error", "over_token_caption", "Caption exceeds the token budget.", item, resolved_route, tags[:8], {"token_count": token_info["token_count"], "max_token_count": max_token_count}, "Trim or restructure the caption.", 0.95))
            if item.caption_text and token_info["token_count"] <= short_threshold:
                findings.append(self._finding("info", "unusually_short_caption", "Caption is unusually short for this dataset.", item, resolved_route, tags[:4], {"token_count": token_info["token_count"], "dataset_avg_tokens": round(avg_tokens, 2)}, "Check whether key concepts are missing.", 0.7))
            if item.caption_text and token_info["token_count"] >= long_threshold:
                findings.append(self._finding("info", "unusually_long_caption", "Caption is unusually long for this dataset.", item, resolved_route, tags[:8], {"token_count": token_info["token_count"], "dataset_avg_tokens": round(avg_tokens, 2)}, "Consider shortening or moving weak tags out.", 0.73))
            if item.caption_source == "filename":
                findings.append(self._finding("warning", "filename_derived_caption_risk", "Caption was derived from the filename instead of a stored caption file.", item, resolved_route, tags[:6], {}, "Review the derived text before training.", 0.84))
            if item.caption_text and self._caption_style(item.caption_text) != modal_style and modal_style not in {"empty", "mixed"}:
                findings.append(self._finding("info", "caption_style_inconsistency", "Caption style differs from the dataset's dominant style.", item, resolved_route, tags[:6], {"item_style": self._caption_style(item.caption_text), "dataset_style": modal_style}, "Normalize style for consistency.", 0.78))
            contradictions = self._find_contradictions(lower_tags)
            if contradictions:
                findings.append(self._finding("warning", "contradictory_tags", "Caption contains contradictory tags.", item, resolved_route, contradictions, {"pairs": contradictions}, "Review mutually exclusive tags.", 0.9))
            suspicious_rare = [tag for tag in tags if tag_counts.get(tag, 0) <= 1 and self._is_suspicious_rare_tag(tag)]
            if suspicious_rare:
                findings.append(self._finding("warning", "rare_suspicious_tags", "Rare tags look noisy or malformed.", item, resolved_route, suspicious_rare[:8], {"counts": {tag: tag_counts.get(tag, 0) for tag in suspicious_rare[:8]}}, "Verify typos or remove noisy one-off tags.", 0.8))
            if trigger_set:
                trigger_hits = [tag for tag in tags if tag.lower() in trigger_set]
                if trigger_hits and any(tag.lower() not in trigger_set for tag in tags[: max(1, len(trigger_hits))]):
                    findings.append(self._finding("info", "trigger_word_misuse", "Trigger words are present but not placed at the front of the caption.", item, resolved_route, trigger_hits, {"trigger_words": sorted(trigger_set)}, "Move trigger words closer to the front if your workflow expects that.", 0.7))
            if tags and len(tags) > max(30, int(avg_tag_count * 1.8)) and resolved_route == "sdxl":
                findings.append(self._finding("warning", "overlong_flat_tag_chain", "SDXL caption is a long flat tag chain.", item, resolved_route, tags[:10], {"tag_count": len(tags)}, "Break out weak tags or shorten the chain.", 0.83))
            if resolved_route == "sdxl" and tags:
                item_density = token_info["token_count"] / max(1, len(tags))
                if item_density < 1.2:
                    findings.append(self._finding("info", "token_density_low", "SDXL caption has very low token density per tag.", item, resolved_route, tags[:8], {"token_density": round(item_density, 2), "dataset_density": round(token_density, 2)}, "Check whether the caption is too flat or underspecified.", 0.68))
            if resolved_route == "anima" and modal_delimiters > 0 and item.caption_text and item.caption_text.count("|") != modal_delimiters:
                findings.append(self._finding("warning", "route_structure_drift", "Anima structured group count drifts from the dataset norm.", item, resolved_route, tags[:8], {"expected_delimiters": modal_delimiters, "actual_delimiters": item.caption_text.count("|")}, "Normalize structured group layout.", 0.77))
            if resolved_route == "anima" and "|" in item.caption_text:
                groups = [segment.strip() for segment in item.caption_text.split("|")]
                if any(not segment for segment in groups):
                    findings.append(self._finding("warning", "group_order_structure_invalid", "Anima caption contains empty structured groups.", item, resolved_route, tags[:8], {"group_count": len(groups), "groups": groups}, "Remove empty groups and keep a stable group order.", 0.86))
                elif len(groups) >= 2:
                    first_group_size = len(split_tags(groups[0]))
                    if first_group_size == 0:
                        findings.append(self._finding("warning", "fixed_group_drift", "Anima first structured group is empty or malformed.", item, resolved_route, tags[:8], {"groups": groups}, "Restore the expected leading group content.", 0.74))
            if resolved_route == "newbie" and self._caption_style(item.caption_text) == "tag" and modal_style == "natural":
                findings.append(self._finding("warning", "excessive_tag_fragments", "Newbie caption uses tag fragments inside a mostly natural-caption dataset.", item, resolved_route, tags[:8], {"dataset_style": modal_style}, "Rewrite as a more natural caption.", 0.8))
            if resolved_route == "newbie" and item.caption_text and modal_style in {"natural", "mixed"} and self._caption_style(item.caption_text) == "tag":
                findings.append(self._finding("warning", "natural_tag_drift", "Newbie caption drifts from natural language into tag-list form.", item, resolved_route, tags[:8], {"dataset_style": modal_style, "item_style": self._caption_style(item.caption_text)}, "Rewrite the caption into a sentence-style description.", 0.82))
            for source_tag, target_tag in typo_pairs.items():
                if source_tag in lower_tags:
                    findings.append(self._finding("warning", "likely_typo_near_duplicate", f"Tag '{source_tag}' looks like a typo of '{target_tag}'.", item, resolved_route, [source_tag, target_tag], {"candidate": target_tag}, "Merge the typo into the common spelling.", 0.82))

        if resolved_route == "sdxl" and coverage < 0.8:
            findings.append(
                TagFinding(
                    severity="warning",
                    code="weak_caption_coverage",
                    message="SDXL dataset caption coverage is weaker than recommended.",
                    route_family=resolved_route,
                    evidence={"caption_coverage": round(coverage, 4)},
                    suggested_action="Add captions for more images before training.",
                    confidence=0.88,
                )
            )

        review_queues: Dict[str, List[str]] = defaultdict(list)
        for finding in findings:
            if finding.image_path:
                review_queues[finding.code].append(finding.image_path)

        findings_dicts = [asdict(finding) for finding in findings]
        report = TagAnalysisReport(
            version=ANALYSIS_VERSION,
            dataset_path=str(dataset_dir),
            route_family=resolved_route,
            dataset_signature=self.compute_dataset_signature(str(dataset_dir), recursive=recursive, caption_extension=caption_extension),
            created_at=datetime.now().isoformat(),
            summary={
                "image_count": len(items),
                "captioned_count": sum(1 for item in items if item.caption_text),
                "missing_caption_count": sum(1 for item in items if not item.caption_exists),
                "empty_caption_count": sum(1 for item in items if item.caption_exists and not item.caption_text),
                "avg_token_count": round(avg_tokens, 2),
                "avg_tag_count": round(avg_tag_count, 2),
                "caption_style": modal_style,
            },
            findings=findings_dicts,
            findings_by_severity=dict(Counter(finding["severity"] for finding in findings_dicts)),
            review_queues=dict(review_queues),
            route_audit={
                "route_family": resolved_route,
                "dataset_style": modal_style,
                "caption_coverage": round(coverage, 4),
                "structured_group_mode": modal_delimiters,
                "token_density": round(token_density, 3),
                "checks": self._route_checks(resolved_route),
            },
            tag_distribution={
                "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(30)],
                "rare_tags": [{"tag": tag, "count": count} for tag, count in sorted(tag_counts.items(), key=lambda pair: (pair[1], pair[0].lower()))[:30]],
                "unique_tag_count": len(tag_counts),
            },
            token_budget={
                "max_token_count": max_token_count,
                "max_seen": max(token_counts) if token_counts else 0,
                "min_seen": min(token_counts) if token_counts else 0,
                "over_limit_count": sum(1 for count in token_counts if count > max_token_count),
            },
            items=serialized_items,
        )
        return asdict(report)

    def _finding(
        self,
        severity: str,
        code: str,
        message: str,
        item: DatasetCaptionItem,
        route_family: str,
        related_tags: Sequence[str],
        evidence: Dict[str, Any],
        suggested_action: str,
        confidence: float,
    ) -> TagFinding:
        return TagFinding(
            severity=severity,
            code=code,
            message=message,
            image_path=str(item.image_path),
            route_family=route_family,
            related_tags=list(related_tags),
            evidence=evidence,
            suggested_action=suggested_action,
            confidence=confidence,
        )

    def _normalize_route(self, route_family: str) -> str:
        value = str(route_family or "").strip().lower()
        if value in {"sdxl", "anima", "newbie"}:
            return value
        return "generic"

    def _caption_style(self, caption: str) -> str:
        text = str(caption or "").strip()
        if not text:
            return "empty"
        if text.count(",") >= 2 and not any(p in text for p in _SENTENCE_PUNCTUATION):
            return "tag"
        if any(p in text for p in _SENTENCE_PUNCTUATION) or len(text.split()) >= 10:
            return "natural"
        return "mixed"

    def _find_contradictions(self, lower_tags: Sequence[str]) -> List[str]:
        tag_set = set(lower_tags)
        found: List[str] = []
        for left, right in _CONTRADICTION_PAIRS:
            if left in tag_set and right in tag_set:
                found.extend([left, right])
        return found

    def _is_suspicious_rare_tag(self, tag: str) -> bool:
        text = str(tag or "").strip()
        if len(text) <= 1:
            return True
        if "__" in text or "  " in text:
            return True
        if re.search(r"[\\/]|[0-9]{4,}", text):
            return True
        if re.search(r"[^0-9A-Za-z _()\-]", text):
            return True
        return False

    def _find_typo_pairs(self, tag_counts: Counter[str]) -> Dict[str, str]:
        common = [tag.lower() for tag, count in tag_counts.items() if count >= 2]
        rare = [tag.lower() for tag, count in tag_counts.items() if count == 1]
        results: Dict[str, str] = {}
        for tag in rare:
            candidates = get_close_matches(tag, common, n=1, cutoff=0.8)
            if not candidates:
                continue
            candidate = candidates[0]
            if abs(len(tag) - len(candidate)) > 3:
                continue
            if SequenceMatcher(None, tag, candidate).ratio() < 0.8:
                continue
            results[tag] = candidate
        return results

    def _route_checks(self, route_family: str) -> List[str]:
        if route_family == "sdxl":
            return ["token_density", "flat_tag_chain", "caption_coverage"]
        if route_family == "anima":
            return ["structured_group_validation", "group_order_drift"]
        if route_family == "newbie":
            return ["natural_vs_tag_drift", "text_style_inconsistency", "tag_fragment_overuse"]
        return ["generic_caption_health", "tag_noise", "token_budget"]
