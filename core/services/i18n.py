import json
import os
import locale
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)

class I18nService:
    SUPPORTED_LANGUAGES = {'zh': '简体中文', 'zh_CN': '简体中文', 'en': 'English'}

    def __init__(self, config_manager: Optional[Any] = None):
        self.config = config_manager
        self.current_language: Optional[str] = None
        self.translations: Dict[str, Any] = {}
        backend_root = Path(__file__).resolve().parents[2]
        self.language_dirs = [
            backend_root / 'lulynx_launcher' / 'i18n',
            backend_root / 'language',
        ]
        self.external_translator: Optional[Any] = None
        self._initialize_language()

    def _normalize_language(self, language: Optional[str]) -> str:
        value = str(language or '').strip()
        if value in {'zh', 'zh_CN', 'zh-CN', 'zh_Hans', 'zh-Hans'}:
            return 'zh'
        if value.startswith('zh'):
            return 'zh'
        return 'en'

    def _initialize_language(self):
        if self.config:
            saved_lang = self.config.get_ui_config('language')
            if saved_lang:
                normalized = self._normalize_language(saved_lang)
                self.current_language = normalized
                self._load_translations(normalized)
                return
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                if system_locale.startswith('zh'):
                    detected_lang = 'zh'
                else:
                    detected_lang = 'en'
                self.current_language = detected_lang
                self._load_translations(detected_lang)
                return
        except Exception as e:
            logger.warning(f'[i18n] System locale detection failed: {e}')
            pass
        self.current_language = 'en'
        self._load_translations('en')

    def _load_translations(self, language):
        language = self._normalize_language(language)
        candidates = []
        for language_dir in self.language_dirs:
            candidates.append(language_dir / f'{language}.json')
            if language == 'zh':
                candidates.append(language_dir / 'zh_CN.json')
        if language != 'en':
            for language_dir in self.language_dirs:
                candidates.append(language_dir / 'en.json')

        translation_file = next((path for path in candidates if path.exists()), None)
        if translation_file is None:
            logger.info('[i18n] No translation file found; using built-in key fallback.')
            self.translations = {}
            return
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            logger.info(f'[i18n] Loaded translations for: {language} ({translation_file})')
        except Exception as e:
            logger.warning(f'[i18n] Error loading translations, using key fallback: {e}')
            self.translations = {}

    def t(self, key: str, **kwargs: Any) -> str:
        keys = key.split('.')
        value = self.translations
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return key
        if kwargs and isinstance(value, str):
            try:
                return value.format(**kwargs)
            except Exception as e:
                logger.error(f'[i18n] String formatting failed for key "{key}": {e}')
                return value
        return str(value)

    def set_language(self, language: str) -> bool:
        normalized = self._normalize_language(language)
        if normalized in self.SUPPORTED_LANGUAGES:
            self.current_language = normalized
            self._load_translations(normalized)
            if self.config:
                self.config.set_ui_config('language', normalized)
            return True
        return False

    def get_current_language(self):
        return self.current_language

    def get_supported_languages(self):
        return self.SUPPORTED_LANGUAGES

    def set_external_translator(self, translator_func):
        self.external_translator = translator_func
        logger.info(f'[i18n] External translator registered: {translator_func}')

    def translate_text(self, text, target_lang=None):
        if target_lang is None:
            target_lang = self.current_language
        if self.external_translator:
            try:
                return self.external_translator(text, target_lang)
            except Exception as e:
                logger.error(f'[i18n] External translation failed: {e}')
                return text
        return text


# ... (omitting class content change for brevity, focusing on the singleton function) ...

_i18n_instance = None
_i18n_lock = threading.Lock()

def get_i18n(config_manager=None):
    global _i18n_instance
    if _i18n_instance is None:
        with _i18n_lock:
            if _i18n_instance is None:
                _i18n_instance = I18nService(config_manager)
    return _i18n_instance

def t(key, **kwargs):
    return get_i18n().t(key, **kwargs)
