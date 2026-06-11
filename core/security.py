"""
REapp Security Core
提供路径遍历防御和 API 认证机制
"""

import os
import json
import secrets
import logging
import hmac
from pathlib import Path
from typing import Optional, List, Union, Set
from fastapi import HTTPException, Header, Depends, status
from fastapi.security import APIKeyHeader
from project_roots import resolve_project_root

logger = logging.getLogger("SecurityCore")

_PROJECT_ROOT = resolve_project_root(source_file=__file__).resolve()
_DATA_ROOT = _PROJECT_ROOT / "data"

# ==========================================
# 1. 路径安全 (Path Traversal Protection)
# ==========================================

# 读 roots：包含项目根（可读源码、配置等）
READ_ROOTS: List[Path] = [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "output",
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "logs",
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "temp",
]

# 写 roots：不包含项目根（不允许覆盖源码）
WRITE_ROOTS: List[Path] = [
    _PROJECT_ROOT / "output",
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "logs",
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "temp",
]

# EXTRA_SAFE_ROOTS：用户显式选择的目录（从 native file dialog 写入）
# 不可通过普通 HTTP API 修改，配置文件放在 launcher 私有目录（不在 WRITE_ROOTS 内）
_LAUNCHER_CONFIG_DIR = _PROJECT_ROOT / "backend" / "lulynx_launcher" / "config"
_EXTRA_ROOTS_FILE = _LAUNCHER_CONFIG_DIR / ".extra_safe_roots.json"
_EXTRA_SAFE_ROOTS: List[Path] = []

# 内部配置/源码路径，HTTP API 不可读也不可写
_INTERNAL_DENY_PATHS: List[Path] = [
    _LAUNCHER_CONFIG_DIR.resolve(),
]

_PROJECT_SOURCE_DENY_TOPLEVEL: Set[str] = {
    "backend",
    "resources",
    "plugin",
    "ref",
    ".git",
}

# 系统级根目录，不允许加入 EXTRA_SAFE_ROOTS
_FORBIDDEN_EXTRA_ROOTS: Set[str] = {
    "/", "C:\\", "D:\\", "E:\\", "F:\\",
}

# Windows 系统目录（运行时动态收集，避免硬编码盘符）
_FORBIDDEN_SYSTEM_DIRS: Set[Path] = set()
try:
    _windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if _windir:
        _FORBIDDEN_SYSTEM_DIRS.add(Path(_windir).resolve())
    for sysdir in ("C:\\Program Files", "C:\\Program Files (x86)", "C:\\ProgramData", "C:\\Users"):
        _FORBIDDEN_SYSTEM_DIRS.add(Path(sysdir).resolve())
except Exception:
    pass

# WRITE_ROOTS 中相对于 _PROJECT_ROOT 的子目录名（允许用户通过 native dialog 加入的）
_WRITE_ROOT_SUBDIRS: Set[str] = {p.name for p in WRITE_ROOTS}

def _is_safe_extra_root(path: Path) -> bool:
    """
    校验候选目录是否可以加入 EXTRA_SAFE_ROOTS。
    拒绝：系统盘根、用户主目录、项目根、项目下非 WRITE_ROOTS 的子目录（含 backend/、resources/、native/ 等源码目录）、
    以及 _INTERNAL_DENY_PATHS 的父目录。
    """
    resolved = path.resolve()
    # 系统级根目录
    if str(resolved) in _FORBIDDEN_EXTRA_ROOTS or len(resolved.parts) <= 1:
        logger.warning(f"拒绝系统级根目录: {resolved}")
        return False
    # 用户主目录和 Users 根目录
    if resolved == Path.home().resolve():
        logger.warning(f"拒绝用户主目录: {resolved}")
        return False
    # Windows 系统目录
    for sysdir in _FORBIDDEN_SYSTEM_DIRS:
        if resolved == sysdir or sysdir in resolved.parents:
            logger.warning(f"拒绝系统目录: {resolved}")
            return False
    # 项目根目录本身
    if resolved == _PROJECT_ROOT.resolve():
        logger.warning(f"拒绝项目根目录: {resolved}")
        return False
    # 项目根下的子目录：只允许 WRITE_ROOTS 中列出的（output/data/logs/models/temp）
    if _PROJECT_ROOT.resolve() in resolved.parents:
        rel = resolved.relative_to(_PROJECT_ROOT.resolve())
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir not in _WRITE_ROOT_SUBDIRS:
            logger.warning(f"拒绝项目源码/配置目录: {resolved}（只允许 {_WRITE_ROOT_SUBDIRS}）")
            return False
    # 不允许是 _INTERNAL_DENY_PATHS 的父目录（防止把 launcher/config/ 暴露为可写）
    for deny_path in _INTERNAL_DENY_PATHS:
        if deny_path == resolved or deny_path.is_relative_to(resolved):
            logger.warning(f"拒绝覆盖内部路径的目录: {resolved}")
            return False
    return True

def _load_extra_roots() -> List[Path]:
    """从可信配置文件加载额外安全根目录"""
    if not _EXTRA_ROOTS_FILE.exists():
        return []
    try:
        data = json.loads(_EXTRA_ROOTS_FILE.read_text("utf-8"))
        roots = []
        for entry in data.get("roots", []):
            p = Path(entry).resolve()
            if _is_safe_extra_root(p):
                roots.append(p)
        return roots
    except Exception as e:
        logger.warning(f"加载 EXTRA_SAFE_ROOTS 失败: {e}")
        return []


def _is_under_project_source(path: Path) -> bool:
    resolved = path.resolve()
    if _PROJECT_ROOT.resolve() not in resolved.parents:
        return False
    rel = resolved.relative_to(_PROJECT_ROOT.resolve())
    if not rel.parts:
        return False
    return rel.parts[0] in _PROJECT_SOURCE_DENY_TOPLEVEL


def _is_forbidden_external_path(path: Path) -> bool:
    resolved = path.resolve()
    if str(resolved) in _FORBIDDEN_EXTRA_ROOTS or len(resolved.parts) <= 1:
        return True
    if resolved == Path.home().resolve():
        return True
    for sysdir in _FORBIDDEN_SYSTEM_DIRS:
        if resolved == sysdir or sysdir in resolved.parents:
            return True
    return False

def _get_extra_roots() -> List[Path]:
    """获取额外安全根目录（带缓存）"""
    global _EXTRA_SAFE_ROOTS
    if not _EXTRA_SAFE_ROOTS:
        _EXTRA_SAFE_ROOTS = _load_extra_roots()
    return _EXTRA_SAFE_ROOTS

def add_extra_safe_root(path: Path) -> bool:
    """
    添加额外安全根目录。仅允许从 native bridge 调用。
    返回 True 表示成功，False 表示被拒绝。
    """
    resolved = path.resolve()
    if not _is_safe_extra_root(resolved):
        return False
    if not resolved.exists() or not resolved.is_dir():
        logger.warning(f"目录不存在: {resolved}")
        return False
    # 写入配置文件（同时清洗旧的不安全 entries）
    try:
        raw = []
        if _EXTRA_ROOTS_FILE.exists():
            raw = json.loads(_EXTRA_ROOTS_FILE.read_text("utf-8")).get("roots", [])
        # 清洗：只保留通过 _is_safe_extra_root 的条目
        existing = [e for e in raw if _is_safe_extra_root(Path(e).resolve())]
        if str(resolved) not in existing:
            existing.append(str(resolved))
        # 如果清洗后有变更（删掉了脏条目），也写回
        if existing != raw:
            logger.info(f"清洗 EXTRA_SAFE_ROOTS: {len(raw)} → {len(existing)} 条")
        _EXTRA_ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _EXTRA_ROOTS_FILE.write_text(json.dumps({"roots": existing}, indent=2), "utf-8")
        global _EXTRA_SAFE_ROOTS
        _EXTRA_SAFE_ROOTS = []  # 清除缓存，下次加载时刷新
        return True
    except Exception as e:
        logger.error(f"保存 EXTRA_SAFE_ROOTS 失败: {e}")
        return False

# 保持向后兼容
SAFE_ROOTS = READ_ROOTS

logger.info(f"READ_ROOTS: {[str(r) for r in READ_ROOTS]}")
logger.info(f"WRITE_ROOTS: {[str(r) for r in WRITE_ROOTS]}")


def _resolve_and_check(path: Union[str, Path], roots: List[Path]) -> Path:
    """解析路径并检查是否在允许的 root 内"""
    if isinstance(path, str):
        path = Path(path)
    resolved = path.resolve()
    # 内部路径拒绝（读写都拦）
    for deny_path in _INTERNAL_DENY_PATHS:
        if resolved == deny_path or deny_path in resolved.parents:
            logger.warning(f"Security Block: 内部路径拒绝: {path}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: Internal configuration path."
            )
    if _is_under_project_source(resolved):
        logger.warning(f"Security Block: 项目源码路径拒绝: {path}")
        raise HTTPException(
            status_code=403,
            detail="Access denied: Project source paths are not allowed."
        )
    for root in roots:
        resolved_root = root.resolve()
        if resolved == resolved_root or resolved_root in resolved.parents:
            return resolved
    logger.warning(f"Security Block: 路径越界: {path}")
    raise HTTPException(
        status_code=403,
        detail="Access denied: Path is outside allowed directories."
    )


def validate_read_path(
    path: Union[str, Path],
    *,
    allow_files: bool = True,
    allow_dirs: bool = True,
    allowed_extensions: Optional[Set[str]] = None,
    safe_roots: Optional[List[Path]] = None,
) -> Path:
    """
    校验读取路径。默认使用 READ_ROOTS + EXTRA_SAFE_ROOTS。
    """
    if safe_roots is None:
        if isinstance(path, str):
            resolved = Path(path).resolve()
        else:
            resolved = path.resolve()
        for deny_path in _INTERNAL_DENY_PATHS:
            if resolved == deny_path or deny_path in resolved.parents:
                logger.warning(f"Security Block: 内部路径拒绝: {path}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: Internal configuration path."
                )
        if _is_under_project_source(resolved):
            logger.warning(f"Security Block: 项目源码路径拒绝: {path}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: Project source paths are not allowed."
            )
        if _is_forbidden_external_path(resolved):
            logger.warning(f"Security Block: 系统路径拒绝: {path}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: System paths are not allowed."
            )
    else:
        resolved = _resolve_and_check(path, safe_roots)

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")

    if resolved.is_file() and not allow_files:
        raise HTTPException(status_code=403, detail="不允许读取文件")
    if resolved.is_dir() and not allow_dirs:
        raise HTTPException(status_code=403, detail="不允许读取目录")
    if allowed_extensions and resolved.is_file():
        if resolved.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=403, detail=f"不允许的扩展名: {resolved.suffix}")

    return resolved


def validate_write_path(
    path: Union[str, Path],
    *,
    allow_files: bool = True,
    allow_dirs: bool = True,
    allowed_extensions: Optional[Set[str]] = None,
    safe_roots: Optional[List[Path]] = None,
) -> Path:
    """
    校验写入路径。默认使用 WRITE_ROOTS + EXTRA_SAFE_ROOTS。
    - 文件写入：parent 必须已存在且在 write root 内
    - 目录写入：如果目标不存在，只允许创建最后一级，父目录必须已存在
    """
    roots = safe_roots if safe_roots is not None else (WRITE_ROOTS + _get_extra_roots())
    default_mode = safe_roots is None
    if safe_roots is None:
        if isinstance(path, str):
            resolved = Path(path).resolve()
        else:
            resolved = path.resolve()
        for deny_path in _INTERNAL_DENY_PATHS:
            if resolved == deny_path or deny_path in resolved.parents:
                logger.warning(f"Security Block: 内部路径拒绝: {path}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: Internal configuration path."
                )
        if _is_under_project_source(resolved):
            logger.warning(f"Security Block: 项目源码路径拒绝: {path}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: Project source paths are not allowed."
            )
        if _is_forbidden_external_path(resolved):
            logger.warning(f"Security Block: 系统路径拒绝: {path}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: System paths are not allowed."
            )
    else:
        resolved = _resolve_and_check(path, roots)

    # 已存在路径的类型检查
    if resolved.exists():
        if resolved.is_file() and not allow_files:
            raise HTTPException(status_code=403, detail="目标是文件，不允许（期望目录）")
        if resolved.is_dir() and not allow_dirs:
            raise HTTPException(status_code=403, detail="目标是目录，不允许（期望文件）")

    if allow_files and not allow_dirs:
        # 文件写入模式：parent 必须已存在且在 root 内
        parent = resolved.parent
        if not parent.exists():
            raise HTTPException(status_code=403, detail="父目录不存在，不允许创建")
        if default_mode:
            for deny_path in _INTERNAL_DENY_PATHS:
                if parent == deny_path or deny_path in parent.parents:
                    logger.warning(f"Security Block: 内部路径拒绝: {parent}")
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: Internal configuration path."
                    )
            if _is_under_project_source(parent):
                logger.warning(f"Security Block: 项目源码路径拒绝: {parent}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: Project source paths are not allowed."
                )
            if _is_forbidden_external_path(parent):
                logger.warning(f"Security Block: 系统路径拒绝: {parent}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: System paths are not allowed."
                )
        else:
            _resolve_and_check(parent, roots)
        if allowed_extensions and resolved.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=403, detail=f"不允许的扩展名: {resolved.suffix}")

    if allow_dirs and not resolved.exists():
        # 目录写入模式：只允许创建最后一级
        parent = resolved.parent
        if not parent.exists():
            raise HTTPException(status_code=403, detail="不允许一次性创建多级目录")
        if default_mode:
            for deny_path in _INTERNAL_DENY_PATHS:
                if parent == deny_path or deny_path in parent.parents:
                    logger.warning(f"Security Block: 内部路径拒绝: {parent}")
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: Internal configuration path."
                    )
            if _is_under_project_source(parent):
                logger.warning(f"Security Block: 项目源码路径拒绝: {parent}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: Project source paths are not allowed."
                )
            if _is_forbidden_external_path(parent):
                logger.warning(f"Security Block: 系统路径拒绝: {parent}")
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: System paths are not allowed."
                )
        else:
            _resolve_and_check(parent, roots)

    return resolved


def validate_path(
    path: Union[str, Path],
    allow_files: bool = True,
    allow_dirs: bool = True,
    must_exist: bool = False,
    safe_roots: Optional[List[Path]] = None
) -> Path:
    """
    向后兼容的路径校验。内部委托给 validate_read_path / validate_write_path。
    新代码请直接使用 validate_read_path 或 validate_write_path。
    """
    if safe_roots is not None:
        # 调用方指定了自定义 roots，直接使用
        roots = safe_roots
        resolved = _resolve_and_check(path, roots)
        if must_exist and not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")
        if resolved.exists():
            if not allow_files and resolved.is_file():
                raise HTTPException(status_code=400, detail="File path not allowed")
            if not allow_dirs and resolved.is_dir():
                raise HTTPException(status_code=400, detail="Directory path not allowed")
        return resolved

    if must_exist:
        return validate_read_path(path, allow_files=allow_files, allow_dirs=allow_dirs)
    else:
        return validate_write_path(path, allow_files=allow_files, allow_dirs=allow_dirs)

# ==========================================
# 2. API 认证 (API Key Authentication)
# TODO: 为本地部署环境默认关闭鉴权，未来支持云端/多用户时再启用
# ==========================================

API_KEY_NAME = "X-REAPP-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 存储 API Key 的文件 (仅本机可见)
AUTH_FILE = _DATA_ROOT / ".auth"

def get_or_create_api_key() -> str:
    """获取或创建持久化的 API Key"""
    # TODO: 实现更加安全的 Key 存储方案 (如系统钥匙串)
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if AUTH_FILE.exists():
        try:
            return AUTH_FILE.read_text("utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to read auth file: {e}")
            
    # 生成新 Key
    new_key = secrets.token_urlsafe(32)
    try:
        AUTH_FILE.write_text(new_key, "utf-8")
        # 设置仅所有者可读写 (Linux/Mac有效，Windows需要额外处理但此处略过)
        os.chmod(AUTH_FILE, 0o600)
    except Exception as e:
        logger.warning(f"Failed to persist API Key: {e}")
        
    return new_key

# 内存中的 Key 缓存
_CURRENT_API_KEY = get_or_create_api_key()

def _api_auth_required() -> bool:
    """Return True when API key auth should be enforced."""
    value = os.environ.get("LULYNX_REQUIRE_API_KEY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency: 验证 API Key
    如果 Key 不匹配，抛出 401
    """
    # [INTENTIONAL DESIGN]
    # Local Desktop Mode: Default to OPEN access.
    # Set LULYNX_REQUIRE_API_KEY=1 to enforce this guard for LAN/remote deployments.
    if not _api_auth_required():
        return True

    if not api_key or not hmac.compare_digest(str(api_key), str(_CURRENT_API_KEY)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True

def get_current_api_key():
    """获取当前有效的 API Key (供前端注入使用)"""
    return _CURRENT_API_KEY
