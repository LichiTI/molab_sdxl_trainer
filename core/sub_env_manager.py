"""
REapp Sub-Environment Manager | [v1.4.5]
Orchestration layer for isolated Python runtimes and dependency sandboxing.
"""

import subprocess
import sys
import shutil
import os
import json
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class EnvInfo:
    """运行环境遥測快照"""
    name: str
    python_path: str
    python_version: str
    packages: List[str]
    valid: bool

class SubEnvManager:
    """
    虚拟环境生命周期管理中心。
    """
    COMPATIBLE_VERSIONS = ['3.12', '3.11', '3.10']
    DEFAULT_ENVS_DIR = Path(__file__).parent.parent / 'envs'

    def __init__(self, envs_dir: Optional[Path] = None):
        self.envs_dir = envs_dir or self.DEFAULT_ENVS_DIR
        self.envs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_current_python_version(cls) -> Tuple[int, int]:
        return sys.version_info[:2]

    @classmethod
    def needs_sub_env(cls) -> bool:
        """探测宿主环境兼容性"""
        if cls.get_current_python_version() < (3, 13):
            return False
        try:
            import onnxruntime
            return False
        except ImportError:
            return True

    def is_tagger_env_ready(self) -> bool:
        """验证 Tagger 环境完整性"""
        env_name = 'tagger_py311'
        python_path = self.get_env_python(env_name)
        if not python_path:
            return False
        try:
            import subprocess
            result = subprocess.run([python_path, '-c', "import onnxruntime; print('ok')"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and 'ok' in result.stdout
        except Exception:
            return False

    def find_compatible_python(self) -> Optional[str]:
        """
        宿主系统 Python 解释器搜索算法。
        按照 COMPATIBLE_VERSIONS 优先级执行启发式搜索。
        """
        search_names = []
        for ver in self.COMPATIBLE_VERSIONS:
            search_names.extend([f"python{ver.replace('.', '')}", f'python{ver}', f'py -{ver}'])
        for name in search_names:
            try:
                if name.startswith('py '):
                    ver_arg = name.split()[-1]
                    result = subprocess.run(['py', ver_arg, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        return f'py {ver_arg}'
                else:
                    python_path = shutil.which(name)
                    if python_path:
                        result = subprocess.run([python_path, '--version'], capture_output=True, text=True)
                        if result.returncode == 0:
                            return python_path
            except Exception:
                continue
        if sys.platform == 'win32':
            for ver in self.COMPATIBLE_VERSIONS:
                # [SECURITY] Use Path.home() over os.getlogin() for cross-session reliability
                home = Path.home()
                common_paths = [
                    Path(os.environ.get('LOCALAPPDATA', '')) / f"Programs/Python/Python{ver.replace('.', '')}/python.exe", 
                    Path('C:/Python' + ver.replace('.', '') + '/python.exe'), 
                    home / f"AppData/Local/Programs/Python/Python{ver.replace('.', '')}/python.exe"
                ]
                for p in common_paths:
                    if p.exists():
                        return str(p)
        return None

    def create_env(self, name: str, python_cmd: Optional[str]=None) -> bool:
        """实例化 venv 容器"""
        env_path = self.envs_dir / name
        if env_path.exists():
            logger.info(f'[SubEnv] 环境 {name} 已存在')
            return True
        if not python_cmd:
            python_cmd = self.find_compatible_python()
        if not python_cmd:
            logger.error('[SubEnv] 未找到兼容的 Python 版本 (3.12-3.13)')
            return False
        logger.info(f'[SubEnv] 使用 {python_cmd} 创建环境 {name}...')
        try:
            if python_cmd.startswith('py '):
                py_args = python_cmd.split()
                subprocess.run(py_args + ['-m', 'venv', str(env_path)], check=True)
            else:
                subprocess.run([python_cmd, '-m', 'venv', str(env_path)], check=True)
            logger.info(f'[SubEnv] 环境 {name} 创建成功')
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f'[SubEnv] 创建环境失败: {e}')
            return False

    def get_env_python(self, name: str) -> Optional[str]:
        """定位环境入口点"""
        env_path = self.envs_dir / name
        if sys.platform == 'win32':
            python_path = env_path / 'Scripts' / 'python.exe'
        else:
            python_path = env_path / 'bin' / 'python'
        if python_path.exists():
            return str(python_path)
        return None

    def install_package(self, name: str, package: str, extra_args: Optional[List[str]] = None, use_mirror: bool = True, timeout: int = 300) -> bool:
        """封装 Pip 包安装逻辑"""
        python_path = self.get_env_python(name)
        if not python_path:
            logger.error(f'[SubEnv] 环境 {name} 不存在')
            return False
        cmd = [python_path, '-m', 'pip', 'install', package, '-q']
        if use_mirror:
            cmd.extend(['-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', '--trusted-host', 'pypi.tuna.tsinghua.edu.cn'])
        if extra_args:
            cmd.extend(extra_args)
        logger.info(f'[SubEnv] 安装 {package}...' + (' (使用国内镜像)' if use_mirror else ''))
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
            logger.info(f'[SubEnv] {package} 安装成功')
            return True
        except subprocess.TimeoutExpired:
            logger.error(f'[SubEnv] 安装超时 ({timeout}秒)')
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f'[SubEnv] 安装失败: {e}')
            return False

    def run_script(self, name: str, script_path: str, args: Optional[List[str]] = None) -> subprocess.CompletedProcess:
        """在隔离环境中执行脚本文件"""
        python_path = self.get_env_python(name)
        if not python_path:
            raise RuntimeError(f'环境 {name} 不存在')
        # 路径校验：脚本必须在允许目录内
        from pathlib import Path
        script = Path(script_path).resolve()
        if not script.exists() or not script.is_file():
            raise ValueError(f'脚本不存在: {script_path}')
        env_dir = self.envs_dir.resolve()
        if not script.is_relative_to(env_dir):
            raise ValueError(f'脚本路径越界: {script_path}（只允许 envs 目录内）')
        cmd = [python_path, str(script)]
        if args:
            cmd.extend(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def run_code(self, name: str, code: str) -> subprocess.CompletedProcess:
        """[已禁用] 任意代码执行不安全，请使用 run_probe()"""
        raise RuntimeError("run_code is disabled; use run_probe() for safe environment probing")

    # 固定的 probe 代码模板，不可由调用方传入
    _PROBE_TEMPLATES = {
        "onnxruntime": "import onnxruntime; print('ok')",
        "numpy": "import numpy; print(numpy.__version__)",
        "pillow": "from PIL import Image; print('ok')",
        "torch": "import torch; print(torch.__version__)",
        "huggingface_hub": "import huggingface_hub; print('ok')",
    }

    def run_probe(self, name: str, probe_key: str) -> subprocess.CompletedProcess:
        """在隔离环境中执行预定义的安全 probe（不允许任意代码）"""
        if probe_key not in self._PROBE_TEMPLATES:
            raise ValueError(f"未知的 probe: {probe_key}，可用: {list(self._PROBE_TEMPLATES.keys())}")
        python_path = self.get_env_python(name)
        if not python_path:
            raise RuntimeError(f'环境 {name} 不存在')
        code = self._PROBE_TEMPLATES[probe_key]
        return subprocess.run([python_path, '-c', code], capture_output=True, text=True, timeout=30)

    def get_env_info(self, name: str) -> Optional[EnvInfo]:
        """获取环境元数据详情"""
        python_path = self.get_env_python(name)
        if not python_path:
            return None
        try:
            result = subprocess.run([python_path, '--version'], capture_output=True, text=True)
            version = result.stdout.strip() if result.returncode == 0 else 'unknown'
            result = subprocess.run([python_path, '-m', 'pip', 'list', '--format=json'], capture_output=True, text=True)
            packages = []
            if result.returncode == 0:
                try:
                    pkg_list = json.loads(result.stdout)
                    packages = [f"{p['name']}=={p['version']}" for p in pkg_list]
                except Exception:
                    pass
            return EnvInfo(name=name, python_path=python_path, python_version=version, packages=packages, valid=True)
        except Exception as e:
            return EnvInfo(name=name, python_path=python_path, python_version='error', packages=[], valid=False)

    def setup_tagger_env(self) -> bool:
        """自动化部署 Tagger 专用环境"""
        env_name = 'tagger_py311'
        if not self.create_env(env_name):
            return False
        packages = ['onnxruntime', 'numpy', 'pillow', 'huggingface_hub']
        for pkg in packages:
            if not self.install_package(env_name, pkg):
                # 警告级别反馈
                logger.warning(f'[SubEnv] 警告: {pkg} 安装失败')
        return True
import threading

_manager: Optional[SubEnvManager] = None
_manager_lock = threading.Lock()

def get_sub_env_manager() -> SubEnvManager:
    """Singleton access for SubEnvManager"""
    global _manager
    if _manager is None:
        with _manager_lock:
             if _manager is None:
                _manager = SubEnvManager()
    return _manager
