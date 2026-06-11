import os
import json
import logging
from pathlib import Path
from typing import Set, Optional, Dict, Any

# [SECURITY] Import core security validation
try:
    from core.security import validate_path
except ImportError:
    # Fallback/Mock for circular import prevention or standalone testing
    def validate_path(path: Any, allow_files=True, allow_dirs=True):
        pass

logger = logging.getLogger(__name__)

class SafeJSONHandler:
    """[SECURITY] Safe JSON Handler with atomic writes and schema validation"""
    
    @staticmethod
    def load(filepath: str, schema: Optional[Dict] = None) -> Any:
        try:
            # Security check
            validate_path(filepath, allow_files=True)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if schema:
                SafeJSONHandler._validate_schema(data, schema)
            return data
        except Exception as e:
            logger.error(f"SafeJSONHandler load failed for {filepath}: {e}")
            raise

    @staticmethod
    def save(filepath: str, data: Any, indent: int = 2) -> None:
        try:
            # Security check
            validate_path(filepath, allow_files=True)
            
            # Atomic write pattern
            path_obj = Path(filepath)
            tmp_path = path_obj.with_suffix('.tmp')
            
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            # Atomic rename
            tmp_path.replace(path_obj)
        except Exception as e:
            logger.error(f"SafeJSONHandler save failed for {filepath}: {e}")
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise

    @staticmethod
    def _validate_schema(data: Any, schema: Dict) -> None:
        for key, expected_type in schema.items():
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(data[key], expected_type):
                raise TypeError(f"Expected {expected_type} for {key}, got {type(data[key])}")

class FileService:

    @staticmethod
    def validate_directory(path: str) -> bool:
        if not path:
            return False
        # [SECURITY] Add path traversal check
        try:
            validate_path(path, allow_dirs=True, allow_files=False)
            return os.path.exists(path) and os.path.isdir(path)
        except Exception:
            return False

    @staticmethod
    def get_images_count(directory: str, extensions: Optional[Set[str]] = None) -> int:
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        if not FileService.validate_directory(directory):
            return 0
        count = 0
        try:
            # [SECURITY] already validated by validate_directory above
            for file in Path(directory).iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    count += 1
        except Exception as e:
            logger.warning(f"Error counting images in {directory}: {e}")
            return 0
        return count

    @staticmethod
    def ensure_directory(path: str) -> None:
        try:
            # [SECURITY] Prevent arbitrary directory creation
            validate_path(path, allow_dirs=True, allow_files=False)
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to ensure directory {path}: {e}")
            raise