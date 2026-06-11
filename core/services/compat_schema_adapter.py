"""Legacy training schema response adapter.

The old WebUI stores executable schema snippets from ``/api/schemas/all`` and
checks ``/api/schemas/hashes`` before refreshing them.  The new launcher keeps
training fields in the typed registry, so this module derives a small legacy
schema view from that registry instead of maintaining a second schema source.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from backend.lulynx_launcher.domain.training_models import (
    LulynxFieldType,
    LulynxTrainingField,
    LulynxTrainingSchema,
)
from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry


_ADAPTER_VERSION = 1


def build_schema_hashes_payload(
    registry: LulynxTrainingRegistry | None = None,
) -> dict[str, Any]:
    """Return the old ``/api/schemas/hashes`` payload shape."""

    return {
        "schemas": [
            {"name": record["name"], "hash": record["hash"]}
            for record in build_legacy_schema_records(registry)
        ]
    }


def build_all_schemas_payload(
    registry: LulynxTrainingRegistry | None = None,
) -> dict[str, Any]:
    """Return the old ``/api/schemas/all`` payload shape."""

    return {"schemas": build_legacy_schema_records(registry)}


def build_legacy_schema_records(
    registry: LulynxTrainingRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build old WebUI schema records from the current training registry."""

    registry = registry or LulynxTrainingRegistry.default()
    records = [_shared_schema_record()]
    records.extend(_schema_record(schema) for schema in registry.get_all())
    return records


def _shared_schema_record() -> dict[str, Any]:
    source = '({ RAW: {}, metadata: { source: "lulynx-registry", version: 1 } })'
    return _record(
        name="shared",
        source=source,
        metadata={
            "source": "lulynx-registry",
            "adapter_version": _ADAPTER_VERSION,
            "purpose": "legacy-shared-placeholder",
        },
    )


def _schema_record(schema: LulynxTrainingSchema) -> dict[str, Any]:
    source = _schema_source(schema)
    return _record(
        name=schema.id,
        source=source,
        metadata={
            "train_type": schema.id,
            "label_zh": schema.label_zh,
            "label_en": schema.label_en,
            "group": schema.group,
            "runtime_hint": schema.runtime_hint,
            "trainer_entry": schema.trainer_entry,
            "experimental": schema.experimental,
            "status": schema.status.value,
            "source": "lulynx-training-registry",
            "adapter_version": _ADAPTER_VERSION,
        },
    )


def _record(name: str, source: str, metadata: dict[str, Any]) -> dict[str, Any]:
    digest_source = {
        "adapter_version": _ADAPTER_VERSION,
        "name": name,
        "schema": source,
        "metadata": metadata,
    }
    digest = hashlib.sha256(
        json.dumps(digest_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"name": name, "hash": digest, "schema": source, "metadata": metadata}


def _schema_source(schema: LulynxTrainingSchema) -> str:
    objects: list[str] = []
    seen: set[str] = set()
    if not _schema_has_field(schema, "model_train_type"):
        objects.append(
            'Schema.object({\n'
            f'  "model_train_type": Schema.string().default({_js(schema.id)}).disabled().description("model_train_type")\n'
            '}).description("metadata")'
        )
        seen.add("model_train_type")

    for section in schema.sections:
        fields = [field for field in _iter_section_fields(section.fields) if field.key not in seen]
        if not fields:
            continue
        seen.update(field.key for field in fields)
        entries = [f"  {_js(field.key)}: {_field_source(field)}" for field in fields]
        title = section.title_zh or section.title_en or section.id
        objects.append(
            "Schema.object({\n"
            + ",\n".join(entries)
            + f"\n}}).description({_js(title)})"
        )

    if not objects:
        objects.append(
            'Schema.object({\n'
            f'  "model_train_type": Schema.string().default({_js(schema.id)}).disabled().description("model_train_type")\n'
            '}).description("metadata")'
        )

    if len(objects) == 1:
        return objects[0]
    return "Schema.intersect([\n" + ",\n".join(f"  {item}" for item in objects) + "\n])"


def _schema_has_field(schema: LulynxTrainingSchema, key: str) -> bool:
    return any(field.key == key for field in _iter_fields(schema))


def _iter_fields(schema: LulynxTrainingSchema) -> Iterable[LulynxTrainingField]:
    for section in schema.sections:
        yield from _iter_section_fields(section.fields)


def _iter_section_fields(items: Iterable[Any]) -> Iterable[LulynxTrainingField]:
    for item in items:
        if isinstance(item, LulynxTrainingField):
            yield item
        elif isinstance(item, (list, tuple)):
            yield from _iter_section_fields(item)


def _field_source(field: LulynxTrainingField) -> str:
    expr = _field_base(field)
    expr = _apply_role(expr, field)
    if _should_emit_default(field):
        expr += f".default({_js(field.default_value)})"
    if field.type in {LulynxFieldType.number, LulynxFieldType.slider}:
        if field.min is not None:
            expr += f".min({_number(field.min)})"
        if field.max is not None:
            expr += f".max({_number(field.max)})"
        if field.step is not None:
            expr += f".step({_number(field.step)})"
    if field.type == LulynxFieldType.hidden:
        expr += ".disabled()"
    description = _field_description(field)
    if description:
        expr += f".description({_js(description)})"
    return expr


def _field_base(field: LulynxTrainingField) -> str:
    if field.type == LulynxFieldType.boolean:
        return "Schema.boolean()"
    if field.type in {LulynxFieldType.number, LulynxFieldType.slider}:
        return "Schema.number()"
    if field.type == LulynxFieldType.select and field.options:
        return f"Schema.union({_js(list(field.options))})"
    if isinstance(field.default_value, bool):
        return "Schema.boolean()"
    if isinstance(field.default_value, (int, float)) and not isinstance(field.default_value, bool):
        return "Schema.number()"
    return "Schema.string()"


def _apply_role(expr: str, field: LulynxTrainingField) -> str:
    if field.type == LulynxFieldType.textarea:
        return expr + '.role("textarea")'
    if field.type == LulynxFieldType.folder:
        return expr + '.role("filepicker", { type: "folder" })'
    if field.type == LulynxFieldType.file:
        return expr + '.role("filepicker", { type: "model-file" })'
    return expr


def _should_emit_default(field: LulynxTrainingField) -> bool:
    if field.default_value is None:
        return False
    if field.type in {LulynxFieldType.number, LulynxFieldType.slider} and field.default_value == "":
        return False
    return True


def _field_description(field: LulynxTrainingField) -> str:
    parts = [
        field.label_zh or field.label_en or field.key,
        field.description_zh or field.description_en,
    ]
    return "\n".join(part for part in parts if part)


def _number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
