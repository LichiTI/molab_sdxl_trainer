from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

class DataType(Enum):
    """Supported data types for node IO"""
    ANY = "any"
    IMAGE = "image"          # PIL.Image
    LATENTS = "latents"      # Latent Tensor
    STRING = "string"        # str
    NUMBER = "number"        # int/float
    BOOLEAN = "boolean"      # bool
    DICT = "dict"            # dict
    LIST = "list"            # list
    FILE_PATH = "file_path"  # Path/str
    NONE = "none"            # NoneType

@dataclass
class IOField:
    """Definition of a single input/output field"""
    name: str
    type: DataType
    description: str = ""
    required: bool = True
    active: bool = True # Can be toggled in dynamic scenarios

@dataclass
class IOSchema:
    """Collection of fields defining an input or output interface"""
    fields: List[IOField] = field(default_factory=list)

    def get_field(self, name: str) -> Optional[IOField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": f.name,
                "type": f.type.value,
                "description": f.description,
                "required": f.required
            }
            for f in self.fields
        ]
