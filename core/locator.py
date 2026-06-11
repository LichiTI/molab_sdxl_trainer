from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from config.settings import ConfigManager
    from core.services.i18n import I18nService
    from core.job_manager import JobManager

import logging

logger = logging.getLogger("Locator")

class Locator:
    _config: Optional['ConfigManager'] = None
    _i18n: Optional['I18nService'] = None
    _jobs: Optional['JobManager'] = None
    jobs: Optional['JobManager'] = None # 公共访问属性
    _initialized: bool = False

    @classmethod
    def init(cls, config: 'ConfigManager', i18n: 'I18nService'):
        cls._config = config
        cls._i18n = i18n
        cls._initialized = True
        logger.info('[Locator] 服务定位器已初始化')

    @classmethod
    def init_jobs(cls, jobs: 'JobManager'):
        cls._jobs = jobs
        cls.jobs = jobs
        logger.info('[Locator] JobManager已注册')

    @classmethod
    def get_config(cls) -> 'ConfigManager':
        """获取 ConfigManager 实例（替代已弃用的 config 属性）"""
        if cls._config is None:
            raise RuntimeError('Locator未初始化，请先调用Locator.init()')
        return cls._config

    @classmethod
    def get_i18n(cls) -> 'I18nService':
        """获取 I18nService 实例（替代已弃用的 i18n 属性）"""
        if cls._i18n is None:
            raise RuntimeError('Locator未初始化，请先调用Locator.init()')
        return cls._i18n

    @classmethod
    def get_jobs(cls) -> Optional['JobManager']:
        """获取 JobManager 实例（替代已弃用的 jobs 属性）"""
        return cls._jobs

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized

    @classmethod
    def reset(cls):
        cls._config = None
        cls._i18n = None
        cls._jobs = None
        cls._initialized = False

def get_config() -> 'ConfigManager':
    return Locator.get_config()

def get_i18n() -> 'I18nService':
    return Locator.get_i18n()

def t(key: str, **kwargs) -> str:
    return Locator.get_i18n().t(key, **kwargs)
