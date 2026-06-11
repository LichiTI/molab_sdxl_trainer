# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
# Warehouse implementation.

import json
import random
from typing import Optional


def try_dual_caption(
    json_text: str,
    short_key: str = "short",
    long_key: str = "long",
) -> Optional[str]:
    if not json_text or not json_text.strip().startswith("{"):
        return None
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    has_short = short_key in data and data[short_key]
    has_long = long_key in data and data[long_key]
    if has_short and has_long:
        return str(data[short_key] if random.random() < 0.5 else data[long_key])
    if has_short:
        return str(data[short_key])
    if has_long:
        return str(data[long_key])
    return None


def try_dual_caption_deterministic(
    json_text: str,
    short_key: str = "short",
    long_key: str = "long",
    use_short: bool = True,
) -> Optional[str]:
    if not json_text or not json_text.strip().startswith("{"):
        return None
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    has_short = short_key in data and data[short_key]
    has_long = long_key in data and data[long_key]
    if has_short and has_long:
        return str(data[short_key] if use_short else data[long_key])
    if has_short:
        return str(data[short_key])
    if has_long:
        return str(data[long_key])
    return None

