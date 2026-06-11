"""
REapp Hugging Face Downloader
Handles GGUF model downloads with mirror (hf-mirror.com) and proxy support.
"""

import os
import logging
import asyncio
import requests
from pathlib import Path, PurePosixPath
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

# Default model directory (absolute path)
_PROJECT_ROOT = Path(__file__).parent.parent  # h:/REapp
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "llm"


def _safe_filename(filename: str, base_dir: Path) -> Path:
    """
    安全构造下载文件路径。拒绝路径遍历、绝对路径、Windows drive 路径。
    允许 HuggingFace 常见的子目录结构（如 model/layer.safetensors）。
    """
    if not filename or not filename.strip():
        raise ValueError("文件名不能为空")
    if len(filename) > 512:
        raise ValueError(f"文件名过长: {len(filename)} > 512")
    if '\\' in filename:
        raise ValueError(f"反斜杠不允许: {filename}")

    pure = PurePosixPath(filename)
    if pure.is_absolute():
        raise ValueError(f"绝对路径不允许: {filename}")

    for part in pure.parts:
        if part == '..':
            raise ValueError(f"路径遍历不允许: {filename}")
        if ':' in part:
            raise ValueError(f"drive 路径不允许: {filename}")

    # 用 parts join 避免 Windows 反斜杠问题
    target = base_dir
    for part in pure.parts:
        target = target / part
    target = target.resolve()

    if not target.is_relative_to(base_dir.resolve()):
        raise ValueError(f"路径越界: {filename}")

    return target

class HFDownloader:
    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._active_downloads: Dict[str, bool] = {}

    async def download_gguf(
        self, 
        repo_id: str, 
        filename: str, 
        progress_callback: Optional[Callable[[float, str], None]] = None,
        use_mirror: bool = False,
        mirror_url: str = "https://hf-mirror.com",
        proxy_url: Optional[str] = None
    ) -> bool:
        """
        Download a GGUF file from Hugging Face.
        """
        download_id = f"{repo_id}/{filename}"
        if self._active_downloads.get(download_id):
            logger.warning(f"[HFDownloader] already downloading {download_id}")
            return False

        self._active_downloads[download_id] = True
        
        try:
            # Construct URL
            base_url = mirror_url if use_mirror else "https://huggingface.co"
            # Standard HF direct download URL: https://huggingface.co/repo/resolve/main/file
            url = f"{base_url.rstrip('/')}/{repo_id}/resolve/main/{filename}"

            # 安全路径构造：拒绝路径遍历
            target_path = _safe_filename(filename, self.model_dir)
            
            logger.info(f"[HFDownloader] Starting download: {url} -> {target_path}")
            if progress_callback:
                progress_callback(0, f"开始下载 {filename}...")

            # Run blocking download in a thread
            return await asyncio.to_thread(
                self._download_sync, 
                url, 
                target_path, 
                progress_callback, 
                proxy_url
            )
            
        except Exception as e:
            logger.error(f"[HFDownloader] Download failed: {e}")
            if progress_callback:
                progress_callback(-1, f"下载失败: {str(e)}")
            return False
        finally:
            self._active_downloads[download_id] = False

    def _download_sync(self, url: str, target_path: Path, progress_callback, proxy_url):
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        
        try:
            # Use stream=True to get content in chunks
            # Increased timeout: (connect_timeout, read_timeout) - mirrors can be slow
            logger.info(f"[HFDownloader] Connecting to: {url}")
            response = requests.get(url, stream=True, proxies=proxies, timeout=(30, 120))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            last_report_percent = -1  # Throttle: only report every 1%
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8*1024*1024):  # 8MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0 and progress_callback:
                            percent = (downloaded_size / total_size) * 100
                            # Throttle: only update every 1%
                            if int(percent) > last_report_percent:
                                last_report_percent = int(percent)
                                progress_callback(percent, f"正在下载: {percent:.1f}% ({downloaded_size//1024**2}MB / {total_size//1024**2}MB)")
            
            logger.info(f"[HFDownloader] Download complete: {target_path}")
            if progress_callback:
                progress_callback(100, "下载完成！")
            return True
            
        except Exception as e:
            if target_path.exists():
                target_path.unlink() # Cleanup partial file
            raise e

# Global instance
hf_downloader = HFDownloader()
