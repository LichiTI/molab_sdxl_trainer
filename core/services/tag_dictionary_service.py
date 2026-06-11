try:
    import yaml
except ImportError:
    yaml = None
import json
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Any, Optional

class TagDictionaryService:
    """
    聚合开源项目的精选 Tag 库，提供带汉化的分类查询服务。
    """
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.groups: Dict[str, List[Dict[str, Any]]] = {}
        self.all_curated_tags: Dict[str, Dict[str, Any]] = {}

    def sync(self):
        """扫描 YAML 目录并按文件夹分组聚合数据"""
        if not self.data_root.exists():
            logger.error(f"[TagDictionary] Data directory not found: {self.data_root}")
            return False

        self.groups = {}
        self.all_curated_tags = {}

        # 文件夹映射名（中文美化）
        group_names = {
            "human": "人物主体",
            "natural": "自然环境",
            "image-composition": "构图镜头",
            "artistic-license": "艺术风格",
            "characters": "作品角色",
            "items": "物品道具",
            "restricted": "限制级资源",
            "clothing": "服装服饰",
            "food": "饮食文化",
            "humanities": "人文景观",
            "others": "其他分类",
            "synced": "已同步资源"
        }

        # 遍历目录及子目录
        found_files = 0
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.data_root)
                    
                    # 识别大类 (Group)
                    group_key = self._detect_group(file_path, rel_path)
                    
                    group_name = group_names.get(group_key.lower(), group_key.replace('-', ' ').title())
                    if group_name not in self.groups:
                        self.groups[group_name] = []
                    
                    self._parse_yaml(file_path, self.groups[group_name])
                    found_files += 1
        
        logger.info(f"[TagDictionary] Found {found_files} files, grouped into {len(self.groups)} categories.")
        return True

    def _detect_group(self, file_path: Path, rel_path: Path) -> str:
        """智能识别文件所属的分组键"""
        stem = file_path.stem.lower()
        
        # 1. 基于文件名的强特征识别 (优先级最高)
        if any(k in stem for k in ['human', 'body', 'face', 'hair', 'head', 'hand', 'foot', 'anatomy', 'eye']): 
            return "human"
        if any(k in stem for k in ['clothing', 'dress', 'outfit', 'suit', 'apparel', 'wear', 'accessory', 'jewelry', 'shoes', 'socks']): 
            return "clothing"
        if any(k in stem for k in ['natural', 'sky', 'weather', 'flora', 'outdoors', 'environment', 'tree', 'flower', 'cloud']): 
            return "natural"
        if any(k in stem for k in ['composition', 'perspective', 'style', 'lighting', 'view', 'background']): 
            return "image-composition"
        if any(k in stem for k in ['art', 'artist', 'licence', 'media']): 
            return "artistic-license"
        if any(k in stem for k in ['character', 'game', 'anime', 'role']): 
            return "characters"
        if any(k in stem for k in ['r18', 'nude', 'nsfw', 'restricted', 'erotic']): 
            return "restricted"
        if any(k in stem for k in ['item', 'obj', 'weapon', 'tool', 'prop', 'vehicle']): 
            return "items"
        if any(k in stem for k in ['food', 'drink', 'fruit', 'vegetable', 'meat', 'candy', 'snack', 'spice']):
            return "food"
        if any(k in stem for k in ['building', 'city', 'indoor', 'outdoor', 'architecture']):
            return "humanities"

        # 2. 如果文件名无法判定，尝试基于文件夹结构 (策略：取最深层的有效文件夹名)
        # 避开像 'synced', 'tags', 'repo-main' 这种无意义的包装层
        ignore_names = {'synced', 'tags', 'repository', 'main', 'master', 'build'}
        valid_parts = [p for p in rel_path.parts[:-1] if p.lower() not in ignore_names and not p.startswith('.')]
        
        if valid_parts:
            # 优先检查最近的父目录
            potential_key = valid_parts[-1].lower()
            # 如果是随机生成的 repo 名(含连字符)，通常不是好的 Key，除非 valid_parts 就一个
            if len(valid_parts) > 1 and '-' in potential_key and len(potential_key) > 10:
                potential_key = valid_parts[0].lower()
            return potential_key
            
        # 3. 兜底策略
        return "others"

    def _parse_yaml(self, path: Path, target_list: List[Dict]):
        if yaml is None:
            logger.warning(f"[TagDictionary] Cannot parse {path}: PyYAML not installed.")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if not data: return

                category_name = data.get('name', path.stem)
                tags_source = data.get('content', data)
                
                tags_list = []
                reserved_keys = ['name', 'description', 'version', 'author']

                for key, info in tags_source.items():
                    if key in reserved_keys or not isinstance(info, dict):
                        continue
                        
                    eng_name = key
                    tag_data = {
                        "en": eng_name,
                        "zh": info.get('name', eng_name),
                        "category": category_name,
                        "alias": info.get('alias', []),
                        "wiki": info.get('wikiURL', ''),
                        "image": info.get('image', ''),
                        "restricted": info.get('restricted', False)
                    }
                    tags_list.append(tag_data)
                    self.all_curated_tags[eng_name.lower()] = tag_data

                if tags_list:
                    target_list.append({
                        "id": path.stem,
                        "name": category_name,
                        "tags": tags_list
                    })
        except Exception as e:
            logger.error(f"[TagDictionary] Error parsing {path}: {e}")

    def get_all_data(self):
        output = []
        # 按中文名排个序，或者按我们在 group_names 中定义的顺序
        sorted_groups = sorted(self.groups.items(), key=lambda x: x[0])
        for name, items in sorted_groups:
            output.append({
                "group": name,
                "items": items
            })
        return {
            "categories": output,
            "total_count": len(self.all_curated_tags)
        }

    def search(self, query: str):
        """在精选库中进行中英文模糊搜索"""
        query = query.lower()
        matches = []
        for en, data in self.all_curated_tags.items():
            if query in en or query in data['zh'].lower() or any(query in a.lower() for a in data['alias']):
                matches.append(data)
            if len(matches) >= 50: break
        return matches

# Global instance
_tag_dict_service: Optional[TagDictionaryService] = None

def get_tag_dict_service() -> TagDictionaryService:
    global _tag_dict_service
    if _tag_dict_service is None:
        # Default path relative to workspace if not initialized
        path = Path(__file__).parent.parent / "resources" / "curated_tags"
        _tag_dict_service = TagDictionaryService(path)
    return _tag_dict_service

def init_tag_dict_service(path: str) -> bool:
    service = get_tag_dict_service()
    service.data_root = Path(path)
    return service.sync()
