"""Small JSON-backed aesthetic labeling service."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class AestheticLabelStore:
    def __init__(self, state_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.state_path = state_path or root / "data" / "aesthetic_labels.json"

    def list_sources(self) -> list[dict[str, Any]]:
        return list(self._read().get("sources", []))

    def add_source(self, path: str, name: str = "") -> dict[str, Any]:
        data = self._read()
        source = {
            "id": f"src_{int(time.time() * 1000)}",
            "name": name or Path(path).name,
            "path": path,
            "created_at": time.time(),
        }
        data.setdefault("sources", []).append(source)
        self._write(data)
        return source

    def samples(self, source_id: str = "", source_path: str = "", limit: int = 100) -> list[dict[str, Any]]:
        data = self._read()
        path = source_path
        if source_id:
            for source in data.get("sources", []):
                if source.get("id") == source_id:
                    path = source.get("path", "")
                    break
        root = Path(path)
        labels = data.get("labels", {})
        if not root.is_dir():
            return []
        items = []
        for img in sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[:limit]:
            key = str(img)
            items.append({
                "path": key,
                "name": img.name,
                "label": labels.get(key, {}).get("label"),
                "score": labels.get(key, {}).get("score"),
                "notes": labels.get(key, {}).get("notes", ""),
            })
        return items

    def set_label(self, image_path: str, label: str = "", score: float | None = None, notes: str = "") -> dict[str, Any]:
        data = self._read()
        entry = {"label": label, "score": score, "notes": notes, "updated_at": time.time()}
        data.setdefault("labels", {})[image_path] = entry
        self._write(data)
        return entry

    def stats(self) -> dict[str, Any]:
        data = self._read()
        labels = data.get("labels", {})
        return {
            "source_count": len(data.get("sources", [])),
            "label_count": len(labels),
            "positive_count": sum(1 for item in labels.values() if item.get("label") in {"good", "positive", "keep"}),
            "negative_count": sum(1 for item in labels.values() if item.get("label") in {"bad", "negative", "reject"}),
        }

    def settings(self, updates: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._read()
        settings = data.setdefault("settings", {})
        if updates:
            settings.update(updates)
            self._write(data)
        return settings

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"sources": [], "labels": {}, "settings": {}}

    def _write(self, data: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
