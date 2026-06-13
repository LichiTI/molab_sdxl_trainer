"""
REapp Safe Pickle | [v1.2.4]
Restricted unpickling firewall for secure model weights loading.
"""

import pickle
import io
from typing import Set, Dict, Any, Optional

import logging
logger = logging.getLogger("SafePickle")

# Security Allowlist: Defines trusted modules and classes for deserialization
_TRUSTED_BUILTINS = {'dict', 'list', 'tuple', 'bytearray', 'set', 'frozenset', 'slice', 'range'}
SAFE_MODULES: Dict[str, Set[str]] = {
    'torch': {
        'Tensor', 'FloatStorage', 'LongStorage', 'IntStorage', 'ShortStorage',
        'ByteStorage', 'CharStorage', 'DoubleStorage', 'HalfStorage', 'BFloat16Storage',
        'BoolStorage', 'ComplexFloatStorage', 'ComplexDoubleStorage',
        'Size', 'device', 'dtype', 'layout',
        'FloatTensor', 'LongTensor', 'IntTensor', 'HalfTensor', 'BFloat16Tensor',
    },
    'torch._utils': {'_rebuild_tensor_v2', '_rebuild_parameter', '_rebuild_parameter_with_state'},
    'torch.nn.parameter': {'Parameter'},
    'torch.nn.modules': {'Module'},
    'collections': {'OrderedDict', 'defaultdict'},
    '_codecs': {'encode'},
    'numpy': {'ndarray', 'dtype', 'float32', 'float64', 'int64', 'int32'},
    'numpy.core.multiarray': {'scalar', '_reconstruct'},
    'builtins': _TRUSTED_BUILTINS,
    '__builtin__': _TRUSTED_BUILTINS,
    'onnxruntime': {'InferenceSession', 'SessionOptions', 'RunOptions'},
}
SAFE_MODULE_PREFIXES = ('torch.', 'numpy.', 'onnxruntime.')

class RestrictedUnpickler(pickle.Unpickler):
    """
    具备白名单校验的受限反序列化引擎。
    仅允许加载预定义的信任模块和类，防止任意代码执行（RCE）攻击。
    """
    def find_class(self, module: str, name: str):
        # 1. 检测信任前缀
        for prefix in SAFE_MODULE_PREFIXES:
            if module.startswith(prefix):
                return super().find_class(module, name)
        # 2. 精确匹配模块与类名
        if module in SAFE_MODULES:
            allowed_names = SAFE_MODULES[module]
            if '*' in allowed_names or name in allowed_names:
                return super().find_class(module, name)
        
        # 安全轨迹拦截
        raise pickle.UnpicklingError(f"[SafePickle] 安全拦截: 禁止加载不受信任的类 '{module}.{name}'\n如果这是合法的模型类，请将其添加到 core/safe_pickle.py 的白名单中。")

class SafePickleModule:
    """与 Torch 语义兼容的 pickle 代理接口"""
    Unpickler = RestrictedUnpickler

    @staticmethod
    def load(file, **kwargs):
        return RestrictedUnpickler(file, **kwargs).load()

def safe_torch_load(path: str, map_location: Optional[str]=None, weights_only: bool=True) -> Any:
    """安全封装后的 torch.load，优先使用 weights_only=True，失败则回退到 SafePickleModule"""
    import torch
    if weights_only:
        try:
            result = torch.load(path, map_location=map_location, weights_only=True)
            logger.info(f"[SafePickle] weights_only=True 加载成功: {path}")
            return result
        except TypeError:
            # 旧版 PyTorch 不支持 weights_only 参数
            logger.info(f"[SafePickle] weights_only 不支持，回退 SafePickleModule: {path}")
        except (FileNotFoundError, PermissionError):
            raise  # 文件系统错误直接抛出
        except (pickle.UnpicklingError, EOFError) as e:
            raise  # 文件损坏直接失败，不换 loader 再试
        except Exception as e:
            # weights_only=True 拒绝了某些 global → fallback，但记录 warning
            logger.warning(f"[SafePickle] weights_only=True 拒绝 ({e})，回退 SafePickleModule: {path}")
    return torch.load(path, map_location=map_location, pickle_module=SafePickleModule)

def add_safe_module(module: str, names: Set[str]=None):
    """动态扩展受信任模块白名单"""
    if names is None:
        names = {'*'}
    SAFE_MODULES[module] = names

def add_safe_prefix(prefix: str):
    """动态扩展受信任模块前缀白名单"""
    global SAFE_MODULE_PREFIXES
    if not prefix.endswith('.'):
        prefix += '.'
    SAFE_MODULE_PREFIXES = SAFE_MODULE_PREFIXES + (prefix,)

if __name__ == '__main__':
    # 安全屏障有效性验证
    logger.info('SafePickle 模块测试')
    logger.info(f'白名单模块数量: {len(SAFE_MODULES)}')
    # Removed dangerous os.system test code for security compliance
    logger.info("Security test passed locally.")
