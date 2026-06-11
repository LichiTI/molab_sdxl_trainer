import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
SYSTEM_TAGS = {'image': '📷 图片', 'model': '🧠 模型', 'output': '📤 输出', 'input': '📥 输入', 'dataset': '📚 数据集', 'lora': '🎨 LoRA', 'any': '📁 通用'}

import logging
logger = logging.getLogger(__name__)

@dataclass
class ShortcutPath:
    name: str
    path: str
    tags: List[str] = field(default_factory=lambda: ['any'])

    def matches_filter(self, filter_tags: List[str]) -> bool:
        if not filter_tags or 'any' in filter_tags:
            return True
        return any((tag in self.tags for tag in filter_tags))

class ShortcutManager:

    def __init__(self, config_dir: Path=None):
        self.config_dir = config_dir or Path(__file__).parent.parent / 'config'
        self.config_file = self.config_dir / 'shortcuts.json'
        self._shortcuts: List[ShortcutPath] = []
        self._load()

    def _load(self):
        if not self.config_file.exists():
            self._shortcuts = []
            return
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._shortcuts = [ShortcutPath(**item) for item in data.get('shortcuts', [])]
        except Exception as e:
            logger.error(f'[ShortcutManager] Failed to load: {e}')
            self._shortcuts = []

    def _save(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = {'shortcuts': [asdict(s) for s in self._shortcuts]}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f'[ShortcutManager] Failed to save: {e}')

    def get_all(self) -> List[ShortcutPath]:
        return self._shortcuts.copy()

    def get_filtered(self, filter_tags: List[str]) -> List[ShortcutPath]:
        return [s for s in self._shortcuts if s.matches_filter(filter_tags)]

    def add(self, name: str, path: str, tags: List[str]=None) -> bool:
        if tags is None:
            tags = ['any']
        for s in self._shortcuts:
            if s.path == path:
                return False
        self._shortcuts.append(ShortcutPath(name=name, path=path, tags=tags))
        self._save()
        return True

    def remove(self, path: str) -> bool:
        for i, s in enumerate(self._shortcuts):
            if s.path == path:
                del self._shortcuts[i]
                self._save()
                return True
        return False

    def update(self, path: str, name: str=None, tags: List[str]=None) -> bool:
        for s in self._shortcuts:
            if s.path == path:
                if name is not None:
                    s.name = name
                if tags is not None:
                    s.tags = tags
                self._save()
                return True
        return False

    def migrate_from_old_config(self, old_shortcuts: List[str]):
        for path in old_shortcuts:
            if not any((s.path == path for s in self._shortcuts)):
                name = Path(path).name
                self._shortcuts.append(ShortcutPath(name=name, path=path, tags=['any']))
        self._save()
shortcut_manager = ShortcutManager()