import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PresetManager:
    DEFAULT_PRESETS = {
        'default': {
            'name': 'Default (初始化)',
            'created_at': '2024-01-01T00:00:00', 'updated_at': '2024-01-01T00:00:00',
            'modules': {}
        },
        'survival_8gb': {
            'name': 'Survival (极限生存 8GB)', 
            'description': '最大化显存节省 (LoRA-FA + LISA + Adafactor)',
            'created_at': '2024-01-01T00:00:00', 'updated_at': '2024-01-01T00:00:00',
            'modules': {
                'start_training': {  # Assuming module name in flow
                   'trainer_config': {
                        'lisa_enabled': True, 'lisa_active_ratio': 0.2, 'lisa_interval': 1,
                        'lora_fa_enabled': True,
                        'pissa_enabled': False, # PiSSA uses SVD init which might peak VRAM momentarily, risky for survival
                        'dora_enabled': False,
                        'hutchinson_auto_freeze': True, 'hutchinson_freeze_ratio': 0.3,
                        'optimizer_type': 'Adafactor',
                        'mixed_precision': 'bf16',
                        'use_gradient_checkpointing': True
                   }
                }
            }
        },
        'balanced_speed': {
            'name': 'Speedster (极速模式)',
            'description': '速度与质量的平衡 (PiSSA + LISA)',
            'created_at': '2024-01-01T00:00:00', 'updated_at': '2024-01-01T00:00:00',
            'modules': {
                'start_training': {
                    'trainer_config': {
                        'pissa_enabled': True, 'pissa_svd_algo': 'rsvd',
                        'lisa_enabled': True, 'lisa_active_ratio': 0.5, # Less aggressive LISA
                        'lora_fa_enabled': False,
                        'dora_enabled': False,
                        'optimizer_type': 'AdamW8bit',
                    }
                }
            }
        },
        'artist_quality': {
            'name': 'Artist (极致画质)',
            'description': '不计成本追求画质 (DoRA + Manifold + SoftPruning)',
            'created_at': '2024-01-01T00:00:00', 'updated_at': '2024-01-01T00:00:00',
            'modules': {
                'start_training': {
                    'trainer_config': {
                        'dora_enabled': True,
                        'manifold_constraint': True, # Assume geometric_lock maps to this
                        'pruning_enabled': True, 'pruning_target_ratio': 0.7,
                        'lisa_enabled': False, # Full training for quality
                        'optimizer_type': 'Prodigy', 
                        'network_dim': 64, 'network_alpha': 32
                    }
                }
            }
        }
    }

    def __init__(self, config_dir: Path=None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / 'config'
        self.config_dir = Path(config_dir)
        self.presets_file = self.config_dir / 'presets.json'
        self.presets: Dict[str, Dict] = {}
        self.current_preset: str = 'default'
        self._ensure_config_dir()
        self._load_presets()

    def _ensure_config_dir(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_presets(self):
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.presets = data.get('presets', {})
                    self.current_preset = data.get('current', 'default')
            except Exception as e:
                logger.error(f'[PresetManager] Error loading presets: {e}')
                self.presets = dict(self.DEFAULT_PRESETS)
        else:
            self.presets = dict(self.DEFAULT_PRESETS)
            self._save_presets()

    def _save_presets(self):
        try:
            data = {'current': self.current_preset, 'presets': self.presets}
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f'[PresetManager] Error saving presets: {e}')

    def get_preset_names(self) -> List[str]:
        return list(self.presets.keys())

    def get_preset_display_names(self) -> Dict[str, str]:
        return {key: preset.get('name', key) for key, preset in self.presets.items()}

    def get_current_preset(self) -> Dict:
        return self.presets.get(self.current_preset, self.presets.get('default', {}))

    def get_current_preset_name(self) -> str:
        return self.current_preset

    def set_current_preset(self, preset_id: str):
        if preset_id in self.presets:
            self.current_preset = preset_id
            self._save_presets()
            return True
        return False

    def get_module_config(self, module_name: str) -> Dict:
        preset = self.get_current_preset()
        modules = preset.get('modules', {})
        return modules.get(module_name, {})

    def set_module_config(self, module_name: str, config: Dict):
        if self.current_preset not in self.presets:
            return
        if 'modules' not in self.presets[self.current_preset]:
            self.presets[self.current_preset]['modules'] = {}
        self.presets[self.current_preset]['modules'][module_name] = config
        self.presets[self.current_preset]['updated_at'] = datetime.now().isoformat()
        self._save_presets()

    def create_preset(self, preset_id: str, display_name: str, copy_from: str=None) -> bool:
        if preset_id in self.presets:
            return False
        now = datetime.now().isoformat()
        if copy_from and copy_from in self.presets:
            new_preset = json.loads(json.dumps(self.presets[copy_from]))
            new_preset['name'] = display_name
            new_preset['created_at'] = now
            new_preset['updated_at'] = now
        else:
            new_preset = {'name': display_name, 'created_at': now, 'updated_at': now, 'modules': {}}
        self.presets[preset_id] = new_preset
        self._save_presets()
        return True

    def delete_preset(self, preset_id: str) -> bool:
        if preset_id == 'default' or preset_id not in self.presets:
            return False
        del self.presets[preset_id]
        if self.current_preset == preset_id:
            self.current_preset = 'default'
        self._save_presets()
        return True

    def rename_preset(self, preset_id: str, new_name: str) -> bool:
        if preset_id not in self.presets:
            return False
        self.presets[preset_id]['name'] = new_name
        self.presets[preset_id]['updated_at'] = datetime.now().isoformat()
        self._save_presets()
        return True

    def save_current_state(self, modules_config: Dict[str, Dict]):
        if self.current_preset not in self.presets:
            return
        self.presets[self.current_preset]['modules'] = modules_config
        self.presets[self.current_preset]['updated_at'] = datetime.now().isoformat()
        self._save_presets()
preset_manager = PresetManager()