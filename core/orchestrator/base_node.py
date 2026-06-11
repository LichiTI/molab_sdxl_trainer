from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from PIL import Image
from .io_types import IOSchema, IOField, DataType

@dataclass
class NodeConfig:
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any=None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any):
        self.values[key] = value

@dataclass
class NodeResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConfigField:

    def __init__(self, field_type: str, default: Any=None, label: str='', description: str='', range: Optional[tuple]=None, choices: Optional[List[str]]=None, required: bool=False):
        self.field_type = field_type
        self.default = default
        self.label = label
        self.description = description
        self.range = range
        self.choices = choices
        self.required = required

    def to_dict(self) -> Dict:
        return {'type': self.field_type, 'default': self.default, 'label': self.label, 'description': self.description, 'range': self.range, 'choices': self.choices, 'required': self.required}

class BaseNode(ABC):
    TYPE_INPUT = 'input'
    TYPE_OUTPUT = 'output'
    TYPE_FILTER = 'filter'
    TYPE_TRANSFORM = 'transform'
    TYPE_ANALYZE = 'analyze'
    TYPE_EXPORT = 'export'

    def __init__(self):
        self.config = NodeConfig()
        self._is_loaded = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    def get_description() -> str:
        return ''

    @staticmethod
    def get_icon() -> str:
        return '📦'

    @staticmethod
    def get_category() -> str:
        return 'builtin'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_TRANSFORM

    @classmethod
    def get_input_schema(cls) -> IOSchema:
        """
        Define the strict input contract for this node.
        Defaults to accepting a generic image and filename for backward compatibility.
        """
        return IOSchema(fields=[
            IOField(name="image", type=DataType.IMAGE, required=False),
            IOField(name="filename", type=DataType.STRING, required=False),
            IOField(name="metadata", type=DataType.DICT, required=False),
            IOField(name="data", type=DataType.ANY, required=False) # Fallback
        ])

    @classmethod
    def get_output_schema(cls) -> IOSchema:
        """
        Define the strict output contract for this node.
        """
        return IOSchema(fields=[
            IOField(name="image", type=DataType.IMAGE),
            IOField(name="filename", type=DataType.STRING),
            IOField(name="metadata", type=DataType.DICT)
        ])

    @staticmethod
    @abstractmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        pass

    def configure(self, config: Dict[str, Any]):
        schema = self.get_config_schema()
        for key, field in schema.items():
            if key in config:
                self.config[key] = config[key]
            else:
                self.config[key] = field.default

    def on_load(self):
        self._is_loaded = True

    def on_unload(self):
        self._is_loaded = False

    @abstractmethod
    def process(self, data: Any, progress_callback: Optional[Callable[[str], None]]=None) -> NodeResult:
        pass

    def process_batch(self, data_list: List[Any], progress_callback: Optional[Callable[[int, int, str], None]]=None) -> List[NodeResult]:
        results = []
        total = len(data_list)
        for i, data in enumerate(data_list):
            if progress_callback:
                progress_callback(i, total, f'处理中 {i + 1}/{total}')
            result = self.process(data)
            results.append(result)
        if progress_callback:
            progress_callback(total, total, '完成')
        return results

    def validate_config(self) -> List[str]:
        errors = []
        schema = self.get_config_schema()
        for key, field in schema.items():
            value = self.config.get(key)
            if field.required and value is None:
                errors.append(f'缺少必填项: {field.label or key}')
            if value is not None and field.range:
                min_val, max_val = field.range
                if not min_val <= value <= max_val:
                    errors.append(f'{field.label or key} 超出范围 [{min_val}, {max_val}]')
            if value is not None and field.choices:
                if value not in field.choices:
                    errors.append(f'{field.label or key} 无效选项: {value}')
        return errors

    def to_dict(self) -> Dict:
        return {'node_type': self.__class__.__name__, 'config': self.config.values}