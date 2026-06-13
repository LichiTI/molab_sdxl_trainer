"""Training route classification and metadata contract.

Classifies training types (e.g. "anima-lora", "sdxl-finetune") into route
families, resolves capability flags, and produces immutable route contract
objects for downstream metadata stamping.
"""

from lulynx_route_contract.contract import (
    RouteContract,
    classify_route,
    contract_to_metadata,
    extract_metadata_from_mapping,
    label_from_mapping,
    kind_from_mapping,
)
from lulynx_route_contract.families import (
    FAMILY_TABLE,
    LABEL_TABLE,
    resolve_family,
    resolve_label,
)

__all__ = [
    "FAMILY_TABLE",
    "LABEL_TABLE",
    "RouteContract",
    "classify_route",
    "contract_to_metadata",
    "extract_metadata_from_mapping",
    "kind_from_mapping",
    "label_from_mapping",
    "resolve_family",
    "resolve_label",
]
