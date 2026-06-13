"""HMAC-SHA256 metadata signing and verification.

Provides lightweight payload signing for provenance metadata, using
HMAC-SHA256 with a caller-supplied secret.  No asymmetric crypto is
used; this is intended for attestation, not tamper-proof security.
"""

from __future__ import annotations

import hashlib
import hmac


def sign_payload(payload: str, secret: str) -> str:
    """Return a hex-encoded HMAC-SHA256 signature of *payload*.

    Parameters
    ----------
    payload:
        The canonical string to sign (typically JSON with sorted keys).
    secret:
        The HMAC shared secret.

    Returns
    -------
    str
        Lowercase hex digest of the HMAC-SHA256.
    """
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload: str, secret: str, expected_hex: str) -> bool:
    """Verify that *expected_hex* matches the HMAC-SHA256 of *payload*.

    Uses constant-time comparison to avoid timing side-channels.

    Returns
    -------
    bool
        ``True`` if the signature matches, ``False`` otherwise.
    """
    actual = sign_payload(payload, secret)
    return hmac.compare_digest(actual, expected_hex.lower().strip())
