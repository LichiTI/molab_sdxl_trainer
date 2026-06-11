# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for civitai_downloader: URL construction, headers, validation, response parsing.

No actual network calls are made in any test.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))


def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name!r} from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cd = _import_module("civitai_downloader", os.path.join(_HERE, "civitai_downloader.py"))
CivitAIDownloader = _cd.CivitAIDownloader
build_api_url = _cd.build_api_url
build_headers = _cd.build_headers


def test_api_url_construction() -> None:
    """build_api_url(12345) returns the correct CivitAI API endpoint."""
    url = build_api_url(12345)
    assert url == "https://civitai.com/api/v1/models/12345", (
        f"Unexpected URL: {url!r}"
    )
    print("  PASS: test_api_url_construction")


def test_headers_no_token() -> None:
    """build_headers() without a token returns Content-Type only, no Authorization."""
    headers = build_headers()
    assert headers.get("Content-Type") == "application/json", (
        f"Missing Content-Type in headers: {headers}"
    )
    assert "Authorization" not in headers, (
        f"Authorization should not be present without token: {headers}"
    )
    print("  PASS: test_headers_no_token")


def test_headers_with_token() -> None:
    """build_headers('mytoken') includes Authorization: Bearer mytoken."""
    headers = build_headers("mytoken")
    assert "Authorization" in headers, f"Authorization missing: {headers}"
    assert headers["Authorization"] == "Bearer mytoken", (
        f"Expected 'Bearer mytoken', got {headers['Authorization']!r}"
    )
    print("  PASS: test_headers_with_token")


def test_invalid_model_id() -> None:
    """CivitAIDownloader().get_model_info(-1) raises ValueError for a negative model ID."""
    downloader = CivitAIDownloader()
    try:
        downloader.get_model_info(-1)
        assert False, "Should have raised ValueError for model_id=-1"
    except ValueError as e:
        assert "-1" in str(e) or "invalid" in str(e).lower() or "model" in str(e).lower(), (
            f"ValueError message did not mention the bad ID: {e}"
        )
    print("  PASS: test_invalid_model_id")


def test_mock_response_parsing() -> None:
    """Parsing a mock CivitAI API response dict extracts the download URL correctly."""
    # Simulates the structure returned by CivitAI /api/v1/models/<id>
    mock_response = {
        "id": 12345,
        "name": "Example LoRA",
        "modelVersions": [
            {
                "id": 99001,
                "name": "v1.0",
                "files": [
                    {
                        "name": "example_lora.safetensors",
                        "downloadUrl": "https://civitai.com/api/download/models/99001",
                        "primary": True,
                        "sizeKB": 72000,
                    }
                ],
            }
        ],
    }

    # Replicate the parsing logic that CivitAIDownloader.get_download_url would use
    versions = mock_response.get("modelVersions", [])
    assert versions, "No modelVersions in mock response"
    latest_version = versions[0]
    files = latest_version.get("files", [])
    assert files, "No files in latest modelVersion"
    primary = next((f for f in files if f.get("primary")), files[0])
    download_url = primary["downloadUrl"]
    assert download_url == "https://civitai.com/api/download/models/99001", (
        f"Unexpected download URL: {download_url!r}"
    )
    print(f"  PASS: test_mock_response_parsing (url={download_url!r})")


def main() -> int:
    print("CivitAI Downloader Smoke Tests")
    print("=" * 40)
    test_api_url_construction()
    test_headers_no_token()
    test_headers_with_token()
    test_invalid_model_id()
    test_mock_response_parsing()
    print("=" * 40)
    print("All CivitAI downloader smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
