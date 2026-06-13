"""Provenance stamping, runtime banners, and metadata signing.

Provides tools for:
- Rendering runtime startup banners with project metadata
- Stamping model exports with provenance fields (lulynx_* keys)
- HMAC-SHA256 signing of metadata payloads for integrity verification
"""

from lulynx_compliance.banner import render_banner, print_banner
from lulynx_compliance.provenance import build_provenance_fields, write_export_notice
from lulynx_compliance.signing import sign_payload, verify_signature

__all__ = [
    "build_provenance_fields",
    "print_banner",
    "render_banner",
    "sign_payload",
    "verify_signature",
    "write_export_notice",
]
