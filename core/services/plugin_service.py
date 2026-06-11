"""
Plugin Manager Service
Handles discovery, validation, and loading of extensions with permission management.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PluginPermission(BaseModel):
    id: str
    description: str
    severity: str = "normal"  # normal, warning, critical

# Define known permissions
KNOWN_PERMISSIONS = {
    "network": PluginPermission(id="network", description="Access the internet", severity="warning"),
    "filesystem_write": PluginPermission(id="filesystem_write", description="Write to file system", severity="critical"),
    "filesystem_read": PluginPermission(id="filesystem_read", description="Read files", severity="normal"),
    "gpu_compute": PluginPermission(id="gpu_compute", description="Use GPU resources", severity="normal"),
    "process_control": PluginPermission(id="process_control", description="Execute system commands", severity="critical"),
}

class PluginManifest(BaseModel):
    id: str
    name: str
    version: str
    description: str
    author: str
    website: Optional[str] = None
    permissions: List[str] = []
    
    # Validation helper
    def get_permission_details(self) -> List[PluginPermission]:
        return [KNOWN_PERMISSIONS.get(p, PluginPermission(id=p, description="Unknown permission", severity="warning")) 
                for p in self.permissions]

class PluginService:
    def __init__(self, extensions_dir: Path):
        self.extensions_dir = extensions_dir
        self.plugins: Dict[str, PluginManifest] = {}

    def scan_plugins(self) -> List[PluginManifest]:
        """Scan extensions directory for plugins with valid manifest.json"""
        self.plugins = {}
        
        if not self.extensions_dir.exists():
            self.extensions_dir.mkdir(parents=True, exist_ok=True)
            return []

        for item in self.extensions_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        # Basic validation could go here
                        if "id" not in data:
                            data["id"] = item.name
                        
                        manifest = PluginManifest(**data)
                        self.plugins[manifest.id] = manifest
                    except Exception as e:
                        logger.error(f"[PluginManager] Failed to load manifest for {item.name}: {e}")
        
        return list(self.plugins.values())

    def get_plugin_path(self, plugin_id: str) -> Optional[Path]:
        if plugin_id in self.plugins:
            return self.extensions_dir / self.plugins[plugin_id].id  # Assuming folder name matches ID or we store path
        # Fallback: scan to find folder
        for item in self.extensions_dir.iterdir():
            if item.is_dir():
                p = item / "manifest.json"
                if p.exists():
                    try:
                        d = json.loads(p.read_text(encoding="utf-8"))
                        if d.get("id") == plugin_id:
                            return item
                    except Exception:
                        pass
        return None

# Singleton instance will be initialized by main or router
