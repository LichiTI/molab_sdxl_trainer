import json
import logging
import os
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import List, Dict, Any, Optional

try:
    from core.services.native_module_loader import load_lulynx_native
except ImportError:
    from backend.core.services.native_module_loader import load_lulynx_native

try:
    from core.security import validate_path
except ImportError:
    # Fallback/Mock
    def validate_path(path: Any, allow_files=True, allow_dirs=True):
        pass

from .file_service import SafeJSONHandler
from ..presets import SmartPresetManager

class ConfigService:
    _write_lock = Lock()
    logger = logging.getLogger(__name__)

    def __init__(self, config_manager: Any):
        self.config = config_manager
        self.presets_dir = Path('customconfig/presets')
        self.workflows_dir = Path('customconfig/workflows')
        # [SECURITY] Prevent arbitrary directory creation
        try:
             validate_path(self.presets_dir, allow_dirs=True)
             self.presets_dir.mkdir(parents=True, exist_ok=True)
             validate_path(self.workflows_dir, allow_dirs=True)
             self.workflows_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to init config directories: {e}")

    def export_config(self, filepath: str) -> bool:
        try:
            export_data = {'version': '1.0', 'exported_at': datetime.now().isoformat(), 'config': self.config.config}
            with self._write_lock:
                # [SECURITY] Use SafeJSONHandler
                SafeJSONHandler.save(filepath, export_data)
            self.logger.info(f'[ConfigService] Exported config to: {filepath}')
            return True
        except Exception as e:
            self.logger.error(f'[ConfigService] Export failed: {e}')
            return False

    def import_config(self, filepath: str, merge: bool = False) -> bool:
        try:
            # [SECURITY] Use SafeJSONHandler
            import_data = SafeJSONHandler.load(filepath)
            
            imported_config = import_data.get('config', import_data)
            if merge:
                self._deep_merge(self.config.config, imported_config)
            else:
                version = self.config.config.get('version')
                self.config.config = imported_config
                self.config.config['version'] = version
            self.config.save_config()
            self.logger.info(f'[ConfigService] Imported config from: {filepath}')
            return True
        except Exception as e:
            self.logger.error(f'[ConfigService] Import failed: {e}')
            return False

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _native_config_lists_disabled(self) -> bool:
        return str(os.environ.get('LULYNX_DISABLE_NATIVE_CONFIG_LISTS', '')).strip().lower() in {
            '1',
            'true',
            'yes',
            'on',
        }

    def save_preset(self, name: str, description: str = '') -> bool:
        try:
            preset_data = {
                'name': name, 
                'description': description, 
                'created_at': datetime.now().isoformat(), 
                'config': {
                    'workflow': self.config.config.get('workflow', {}), 
                    'processing': self.config.config.get('processing', {})
                }
            }
            safe_name = ''.join((c for c in name if c.isalnum() or c in '_ -')).strip()
            preset_file = self.presets_dir / f'{safe_name}.json'
            with self._write_lock:
                SafeJSONHandler.save(str(preset_file), preset_data)
            self.logger.info(f'[ConfigService] Saved preset: {name}')
            return True
        except Exception as e:
            self.logger.error(f'[ConfigService] Save preset failed: {e}')
            return False

    def load_preset(self, name: str) -> bool:
        try:
            safe_name = ''.join((c for c in name if c.isalnum() or c in '_ -')).strip()
            preset_file = self.presets_dir / f'{safe_name}.json'
            if not preset_file.exists():
                self.logger.warning(f'[ConfigService] Preset not found: {name}')
                return False
            
            preset_data = SafeJSONHandler.load(str(preset_file))
            
            preset_config = preset_data.get('config', {})
            if 'workflow' in preset_config:
                self.config.config['workflow'] = preset_config['workflow']
            if 'processing' in preset_config:
                self.config.config['processing'] = preset_config['processing']
            self.config.save_config()
            self.logger.info(f'[ConfigService] Loaded preset: {name}')
            return True
        except Exception as e:
            self.logger.error(f'[ConfigService] Load preset failed: {e}')
            return False

    def list_presets(self) -> List[Dict[str, str]]:
        native_presets = self._list_presets_native()
        if native_presets is not None:
            return native_presets
        presets: List[Dict[str, str]] = []
        for file in self.presets_dir.glob('*.json'):
            try:
                data = SafeJSONHandler.load(str(file))
                presets.append({
                    'name': str(data.get('name', file.stem)), 
                    'description': str(data.get('description', '')), 
                    'created_at': str(data.get('created_at', ''))
                })
            except Exception as e:
                self.logger.warning(f'[ConfigService] Failed to load preset {file.name}: {e}')
                pass
        return presets

    def _list_presets_native(self) -> Optional[List[Dict[str, str]]]:
        if self._native_config_lists_disabled():
            return None
        native = load_lulynx_native()
        if native is None or not hasattr(native, 'list_config_presets'):
            return None
        try:
            payload = native.list_config_presets(str(self.presets_dir))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        presets: List[Dict[str, str]] = []
        for item in payload.get('presets', []) or []:
            if not isinstance(item, dict):
                continue
            presets.append({
                'name': str(item.get('name', '')),
                'description': str(item.get('description', '')),
                'created_at': str(item.get('created_at', '')),
            })
        return presets

    def delete_preset(self, name: str) -> bool:
        try:
            safe_name = ''.join((c for c in name if c.isalnum() or c in '_ -')).strip()
            preset_file = self.presets_dir / f'{safe_name}.json'
            with self._write_lock:
                if preset_file.exists():
                    preset_file.unlink()
                    self.logger.info(f'[ConfigService] Deleted preset: {name}')
                    return True
            return False
        except Exception as e:
            self.logger.error(f'[ConfigService] Delete preset failed: {e}')
            return False

    def save_workflow_template(self, name: str, blocks: list, description: str='') -> bool:
        try:
            template_data = {'name': name, 'description': description, 'created_at': datetime.now().isoformat(), 'blocks': blocks}
            safe_name = ''.join((c for c in name if c.isalnum() or c in '_ -')).strip()
            template_file = self.workflows_dir / f'{safe_name}.json'
            with self._write_lock:
                 SafeJSONHandler.save(str(template_file), template_data)
            return True
        except Exception as e:
            self.logger.error(f'[ConfigService] Save workflow template failed: {e}')
            return False

    def get_smart_presets(self) -> Dict[str, Any]:
        """获取所有智能预设定义"""
        return SmartPresetManager.PRESETS

    def list_workflow_templates(self) -> List[Dict[str, Any]]:
        native_templates = self._list_workflow_templates_native()
        if native_templates is not None:
            return native_templates
        templates: List[Dict[str, Any]] = []
        for file in self.workflows_dir.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data: Dict[str, Any] = json.load(f)
                templates.append({
                    'name': str(data.get('name', file.stem)), 
                    'description': str(data.get('description', '')), 
                    'blocks': list(data.get('blocks', []))
                })
            except Exception as e:
                self.logger.error(f'[ConfigService] Failed to load workflow template {file.name}: {e}')
                pass
        return templates

    def _list_workflow_templates_native(self) -> Optional[List[Dict[str, Any]]]:
        if self._native_config_lists_disabled():
            return None
        native = load_lulynx_native()
        if native is None or not hasattr(native, 'list_workflow_template_files'):
            return None
        try:
            payload = native.list_workflow_template_files(str(self.workflows_dir))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        templates: List[Dict[str, Any]] = []
        for item in payload.get('templates', []) or []:
            if not isinstance(item, dict):
                continue
            templates.append({
                'name': str(item.get('name', '')),
                'description': str(item.get('description', '')),
                'blocks': list(item.get('blocks', []) or []),
            })
        return templates
