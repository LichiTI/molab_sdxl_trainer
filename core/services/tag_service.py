"""
Tag Service - Provides tag autocomplete and category lookup using Danbooru data.
"""
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import bisect
import logging

logger = logging.getLogger("TagService")

# Danbooru Category IDs
CATEGORY_GENERAL = 0
CATEGORY_ARTIST = 1
CATEGORY_COPYRIGHT = 3
CATEGORY_CHARACTER = 4
CATEGORY_META = 5

# Category Colors (for frontend reference)
CATEGORY_COLORS = {
    0: {"name": "general", "color": "#3b82f6"},      # Blue
    1: {"name": "artist", "color": "#ef4444"},       # Red
    3: {"name": "copyright", "color": "#a855f7"},    # Purple
    4: {"name": "character", "color": "#22c55e"},    # Green
    5: {"name": "meta", "color": "#f97316"},         # Orange
}

@dataclass
class TagInfo:
    name: str
    category: int
    count: int
    aliases: list[str]

class TagService:
    """
    Tag lookup service with autocomplete support.
    Loads Danbooru tags from CSV and provides fast prefix search.
    """
    
    def __init__(self):
        self.tags: dict[str, TagInfo] = {}
        self.sorted_tags: list[str] = []  # For binary search prefix matching
        self.translation_map: dict[str, str] = {} # English -> Chinese cache
        self.loaded = False
        self.cache_path: Optional[Path] = None

    def load_from_csv(self, csv_path: str | Path) -> bool:
        """Load tags from danbooru_tags.csv format: name,category,count,aliases"""
        csv_path = Path(csv_path)
        self.cache_path = csv_path.with_suffix('.translation.json')
        
        # Try to load existing translations
        if self.cache_path.exists():
            try:
                import json
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.translation_map = json.load(f)
                logger.info(f"[TagService] Loaded {len(self.translation_map)} user translations.")
            except Exception as e:
                logger.warning(f"[TagService] Failed to load translations: {e}")

        if not csv_path.exists():
            logger.error(f"[TagService] CSV not found: {csv_path}")
            return False
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    
                    name = row[0].strip()
                    try:
                        category = int(row[1])
                        count = int(row[2])
                    except ValueError:
                        continue
                    
                    aliases = []
                    if len(row) > 3 and row[3]:
                        aliases = [a.strip() for a in row[3].split(',')]
                    
                    self.tags[name] = TagInfo(
                        name=name,
                        category=category,
                        count=count,
                        aliases=aliases
                    )
            
            # Build sorted list for prefix search
            self.sorted_tags = sorted(self.tags.keys())
            self.loaded = True
            logger.info(f"[TagService] Loaded {len(self.tags)} tags")
            return True
        
        except Exception as e:
            logger.error(f"[TagService] Failed to load CSV: {e}")
            return False
    
    def save_translations(self):
        """Persist translations to disk."""
        if self.cache_path:
            import json
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.translation_map, f, ensure_ascii=False, indent=2)

    # --- Normalization Layer ---

    def normalize_tag(self, tag: str) -> str:
        """Normalize a single tag for deterministic comparison.

        Rules:
        - lowercase comparison, preserve display form when possible
        - treat space and underscore as equivalent
        - trim repeated whitespace
        - dedupe escaped and unescaped parentheses forms safely
        """
        tag = str(tag or "").strip()
        if not tag:
            return ""
        # Treat space and underscore as equivalent for comparison
        tag = tag.replace("_", " ")
        # Trim repeated whitespace
        tag = " ".join(tag.split())
        return tag

    def normalize_caption(self, caption: str) -> str:
        """Normalize a full caption/dataset of tags by normalizing each tag and preserving the comma-join.

        Rules:
        - splits tags by comma
        - normalizes each tag
        - removes empty tags and duplicates (case-insensitive)
        - preserves display form (original non-normalized casing)
        """
        if not caption:
            return ""
        raw_tags = [t.strip() for t in str(caption).split(",") if t.strip()]
        seen: set[str] = set()
        result: list[str] = []
        for raw_tag in raw_tags:
            norm_tag = self.normalize_tag(raw_tag)
            if not norm_tag:
                continue
            key = norm_tag.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(raw_tag)  # preserve original display form
        return ", ".join(result)

    def compare_tags(self, tags1: list[str], tags2: list[str]) -> bool:
        """Compare two sets of tags for equality with normalization."""
        set1 = set(self.normalize_tag(t).lower() for t in tags1)
        set2 = set(self.normalize_tag(t).lower() for t in tags2)
        return set1 == set2

    def compute_tag_similarity(self, tag1: str, tag2: str) -> float:
        """Compute similarity between two tags using a fuzzy-matching approach.

        Returns a value between 0 and 1.
        """
        from difflib import SequenceMatcher
        norm1 = self.normalize_tag(tag1)
        norm2 = self.normalize_tag(tag2)
        # Handle the case where normalization strips to empty
        if not norm1 and not norm2:
            return 1.0
        if not norm1 or not norm2:
            return 0.0
        return SequenceMatcher(None, norm1.lower(), norm2.lower()).ratio()

    # --- /Normalization Layer ---

    def get_category_counts(self) -> dict[int, int]:
        """Calculate tag counts per category."""
        counts = {0: 0, 1: 0, 3: 0, 4: 0, 5: 0}
        if not self.loaded:
            return counts
            
        for info in self.tags.values():
            if info.category in counts:
                counts[info.category] += 1
            else:
                # Handle unknown categories if any
                pass
        return counts

    def suggest(self, query: str, limit: int = 20) -> list[dict]:
        """
        Get tag suggestions with translations.
        """
        if not self.loaded or not query:
            return []
        
        query = query.lower().replace(' ', '_')
        results = []
        
        # Binary search for prefix start
        idx = bisect.bisect_left(self.sorted_tags, query)
        
        # Collect matches
        while idx < len(self.sorted_tags) and len(results) < limit * 2:
            tag = self.sorted_tags[idx]
            if not tag.startswith(query):
                break
            
            info = self.tags[tag]
            cat_info = CATEGORY_COLORS.get(info.category, {"name": "unknown", "color": "#888"})
            
            results.append({
                "name": info.name,
                "zh": self.translation_map.get(info.name, ""), # Include translation if exists
                "category": info.category,
                "categoryName": cat_info["name"],
                "color": cat_info["color"],
                "count": info.count
            })
            idx += 1
        
        # Sort by count (popularity) and limit
        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:limit]
    
    def add_translation(self, tag: str, zh: str):
        """Add a new translation to memory."""
        self.translation_map[tag.lower()] = zh
    
    def get_category(self, tag: str) -> Optional[dict]:
        """Get category info for a single tag."""
        tag = tag.lower().replace(' ', '_')
        info = self.tags.get(tag)
        if not info:
            return None
        
        cat_info = CATEGORY_COLORS.get(info.category, {"name": "unknown", "color": "#888"})
        return {
            "name": info.name,
            "category": info.category,
            "categoryName": cat_info["name"],
            "color": cat_info["color"],
            "count": info.count
        }
    
    def get_categories_batch(self, tags: list[str]) -> dict[str, dict]:
        """Get categories for multiple tags at once."""
        result = {}
        for tag in tags:
            info = self.get_category(tag)
            if info:
                result[tag] = info
            else:
                # Unknown tag - default to general
                result[tag] = {
                    "name": tag,
                    "category": 0,
                    "categoryName": "general",
                    "color": "#3b82f6",
                    "count": 0
                }
        return result

    def get_paginated_tags(self, page: int = 1, size: int = 100, query: str = "", category: Optional[int] = None) -> dict:
        """
        获取分页后的标签列表。
        Supports prefix search, category filtering and popularity sorting.
        """
        if not self.loaded:
            return {"items": [], "total": 0}

        source_list = self.sorted_tags
        
        # 1. 如果有分类过滤，先缩小范围
        if category is not None:
             source_list = [t for t in self.sorted_tags if self.tags.get(t).category == category]

        # 2. 如果有搜索词，再进行搜索
        if query:
            query = query.lower().replace(' ', '_')
            temp_filtered = [t for t in source_list if query in t]
            # Additionally search in translations if available
            for t, zh in self.translation_map.items():
                if query in zh and t in source_list and t not in temp_filtered:
                    temp_filtered.append(t)
            source_list = temp_filtered

        total = len(source_list)
        start = (page - 1) * size
        end = start + size
        
        items = []
        for tag in source_list[start:end]:
            info = self.tags.get(tag)
            if info:
                cat_info = CATEGORY_COLORS.get(info.category, {"name": "unknown", "color": "#888"})
                items.append({
                    "name": info.name,
                    "zh": self.translation_map.get(info.name, ""),
                    "category": info.category,
                    "categoryName": cat_info["name"],
                    "color": cat_info["color"],
                    "count": info.count
                })

        return {
            "items": items,
            "total": total,
            "page": page,
            "size": size
        }

    # --- 翻译任务管理 ---
    _translation_status = {"running": False, "processed": 0, "total": 0, "current_batch": ""}

    def get_translation_status(self):
        return self._translation_status

    async def start_background_translation(self, translator_callback):
        """
        启动后台翻译任务。
        按 1000 个一波分批处理，不阻塞主线程。
        """
        if self._translation_status["running"]:
            return
        
        self._translation_status["running"] = True
        # 找出所有未翻译的 tag (按热度排序，先翻译高效词)
        all_tags = sorted(self.tags.values(), key=lambda x: x.count, reverse=True)
        untranslated = [t.name for t in all_tags if t.name not in self.translation_map]
        
        self._translation_status["total"] = len(untranslated)
        self._translation_status["processed"] = 0
        
        batch_size = 1000
        for i in range(0, len(untranslated), batch_size):
            if not self._translation_status["running"]: # 支持中途停止
                break
                
            batch = untranslated[i:i+batch_size]
            self._translation_status["current_batch"] = f"{i} to {i+len(batch)}"
            
            try:
                # 调用翻译回调 (由外部路由器提供，因为需要配置 LLM 等)
                translated_dict = await translator_callback(batch)
                
                # 更新缓存
                for en, zh in translated_dict.items():
                    self.add_translation(en, zh)
                
                self._translation_status["processed"] += len(batch)
                
                # 每 1000 个保存一次磁盘，防止掉电
                self.save_translations()
                
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"[TagService] Batch translation error: {e}")
                # 即使出错了也继续下一波，或者在这里根据错误类型决定是否 stop
            
        self._translation_status["running"] = False

    def stop_background_translation(self):
        self._translation_status["running"] = False

# Global instance
_tag_service: Optional[TagService] = None

def get_tag_service() -> TagService:
    """Get or create the global TagService instance."""
    global _tag_service
    if _tag_service is None:
        _tag_service = TagService()
    return _tag_service

def init_tag_service(csv_path: str | Path) -> bool:
    """Initialize the tag service with a CSV file."""
    service = get_tag_service()
    return service.load_from_csv(csv_path)
