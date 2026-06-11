# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""CivitAI model downloader — public REST API client."""

from __future__ import annotations
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

CIVITAI_API_BASE = "https://civitai.com/api/v1"


class CivitAIDownloader:
    """Download models from CivitAI via their public REST API."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("CIVITAI_API_TOKEN")

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _api_get(self, endpoint: str) -> dict:
        url = f"{CIVITAI_API_BASE}/{endpoint}"
        req = Request(url, headers=self._build_headers())
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            raise RuntimeError(f"CivitAI API error {e.code}: {e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    def get_model_info(self, model_id: int) -> dict:
        """Fetch model metadata from CivitAI."""
        if not isinstance(model_id, int) or model_id <= 0:
            raise ValueError("model_id must be a positive integer")
        return self._api_get(f"models/{model_id}")

    def get_download_url(
        self,
        model_id: int,
        version_id: Optional[int] = None,
    ) -> tuple:
        """Get download URL and filename for a model.

        Returns (url, filename, size_kb).
        """
        info = self.get_model_info(model_id)
        versions = info.get("modelVersions", [])
        if not versions:
            raise ValueError(f"No versions found for model {model_id}")

        version = versions[0]  # latest by default
        if version_id:
            for v in versions:
                if v.get("id") == version_id:
                    version = v
                    break

        files = version.get("files", [])
        if not files:
            raise ValueError(f"No files in version {version.get('id')}")

        # Pick primary file (usually first, or largest)
        primary = files[0]
        for f in files:
            if f.get("primary", False):
                primary = f
                break

        url = primary.get("downloadUrl", "")
        name = primary.get("name", f"model_{model_id}.safetensors")
        size = primary.get("sizeKB", 0)

        return url, name, size

    def download(
        self,
        model_id: int,
        output_dir: str,
        version_id: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Download a model to output_dir. Returns the saved file path."""
        url, filename, size_kb = self.get_download_url(model_id, version_id)
        if not url:
            raise ValueError("No download URL available")

        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(output_path, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)
        except HTTPError as e:
            raise RuntimeError(f"Download failed: {e.code} {e.reason}") from e

        logger.info(f"Downloaded {filename} to {output_path}")
        return str(output_path)


def build_api_url(model_id: int) -> str:
    """Build the API URL for a model. Useful for testing."""
    return f"{CIVITAI_API_BASE}/models/{model_id}"


def build_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build request headers. Useful for testing."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
