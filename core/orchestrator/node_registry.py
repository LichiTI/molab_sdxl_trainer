import logging
from typing import Dict, List, Type, Optional
from .base_node import BaseNode

class NodeRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NodeRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.logger = logging.getLogger('NodeRegistry')
        self._nodes: Dict[str, Type[BaseNode]] = {}
        self._initialized = True

    def register(self, node_class: Type[BaseNode], node_id: Optional[str]=None):
        if not issubclass(node_class, BaseNode):
            raise TypeError(f'{node_class} must be a subclass of BaseNode')
        nid = node_id or node_class.__name__
        self._nodes[nid] = node_class
        self.logger.info(f'Registered node: {nid} ({node_class.get_name()})')

    def unregister(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self.logger.info(f'Unregistered node: {node_id}')

    def get(self, node_id: str) -> Optional[Type[BaseNode]]:
        return self._nodes.get(node_id)

    def create_instance(self, node_id: str) -> Optional[BaseNode]:
        node_class = self.get(node_id)
        if node_class:
            return node_class()
        return None

    def list_all(self) -> List[Dict]:
        result = []
        for node_id, node_class in self._nodes.items():
            result.append({'id': node_id, 'name': node_class.get_name(), 'description': node_class.get_description(), 'icon': node_class.get_icon(), 'category': node_class.get_category(), 'type': node_class.get_node_type()})
        return result

    def list_by_category(self, category: str) -> List[Dict]:
        return [n for n in self.list_all() if n['category'] == category]

    def get_builtin_nodes(self) -> List[Dict]:
        return self.list_by_category('builtin')

    def get_community_nodes(self) -> List[Dict]:
        return self.list_by_category('community')
node_registry = NodeRegistry()

def register_node(node_id: Optional[str]=None):

    def decorator(cls):
        node_registry.register(cls, node_id)
        return cls
    return decorator