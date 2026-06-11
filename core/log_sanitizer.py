"""
REapp Log Sanitizer | [v1.3.8]
Regex-driven PII redaction engine for secure telemetry.
"""

import re
import os
import logging
from pathlib import Path
from typing import List, Tuple

USER_HOME = str(Path.home())

# Transformation rules for sensitive data scrubbing
SANITIZE_RULES: List[Tuple[re.Pattern, str]] = [
    # Path Sanitization: Matches local home directories across platforms
    (re.compile(
        r'(' + re.escape(USER_HOME) + r')|'  
        r'(C:[/\\]Users[/\\][^/\\]+)|'       
        r'(/[/\\]?home[/\\][^/\\]+)|'        
        r'(/[/\\]?Users[/\\][^/\\]+)',       
        re.IGNORECASE
    ), '<HOME>'),
    (re.compile('sk-[a-zA-Z0-9]{20,}'), '<API_KEY>'), 
    (re.compile('api[_-]?key[=:]\\s*["\\\']?[a-zA-Z0-9-_]{16,}["\\\']?', re.IGNORECASE), 'api_key=<REDACTED>'), 
    (re.compile('token[=:]\\s*["\\\']?[a-zA-Z0-9-_]{20,}["\\\']?', re.IGNORECASE), 'token=<REDACTED>'), 
    (re.compile('password[=:]\\s*["\\\']?[^\\s"\\\']{4,}["\\\']?', re.IGNORECASE), 'password=<REDACTED>')
]

def sanitize_log(message: str) -> str:
    """脱敏核心引擎：通过规则集进行流式替换"""
    if not message:
        return message
    result = str(message)
    for pattern, replacement in SANITIZE_RULES:
        result = pattern.sub(replacement, result)
    return result

def sanitize_path(path: str) -> str:
    """路径脱敏：将用户私有路径前缀转换为标准化占位符"""
    if not path:
        return path
    norm_path = str(Path(path).resolve())
    norm_home = str(Path.home().resolve())
    if norm_path.startswith(norm_home):
        return '<HOME>' + norm_path[len(norm_home):]
    return path

class SanitizedFormatter(logging.Formatter):
    """支持脱敏功能的日志格式化器"""
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return sanitize_log(formatted)

class SanitizedLogHandler(logging.Handler):
    """
    脱敏日志处理器拦截器。
    在向上游 Handler 传递前对 LogRecord 进行深度拷贝与静默脱敏。
    """
    def __init__(self, wrapped_handler: logging.Handler):
        super().__init__()
        self.wrapped_handler = wrapped_handler

    def emit(self, record: logging.LogRecord):
        # 实例化脱敏快照副本，防止副作用污染原始日志流
        sanitized_record = logging.makeLogRecord(record.__dict__)
        sanitized_record.msg = sanitize_log(str(record.msg))
        if record.args:
            sanitized_record.args = tuple((sanitize_log(str(arg)) if isinstance(arg, str) else arg for arg in record.args))
        self.wrapped_handler.emit(sanitized_record)

    def flush(self):
        self.wrapped_handler.flush()

    def close(self):
        self.wrapped_handler.close()

def add_sensitive_path(path: str):
    """动态注入自定义脱敏路径规则"""
    if path:
        pattern = re.compile(re.escape(str(path)), re.IGNORECASE)
        SANITIZE_RULES.insert(0, (pattern, '<SENSITIVE>'))

if __name__ == '__main__':
    # 开发环境脱敏逻辑验证
    test_messages = [f'Loading model from {USER_HOME}/models/test.pth', 'C:\\Users\\Ruilynx\\Documents\\secret.txt', 'API key: sk-abcdefghijklmnopqrstuvwxyz123456', 'Setting password=mysecretpass123', 'token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9', '/home/user/.config/settings.json']
    logging.info('日志脱敏测试:')
    logging.info('-' * 50)
    for msg in test_messages:
        logging.info(f'原始: {msg}')
        logging.info(f'脱敏: {sanitize_log(msg)}')
        logging.info('')
