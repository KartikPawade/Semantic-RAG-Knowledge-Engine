"""
Schema Registry: The "System Map" for multi-collection metadata.

Maps each collection to:
- Allowed metadata fields and types (for ingestion + filter extraction)
- Schema hint for the AI (how to use filters in search/ask)
- Primary filter strategy description

Decouples metadata schema from LLM logic so the right filters (city, department,
product_id, region) are used per collection.
"""
from typing import Any, Optional

from pydantic import BaseModel, create_model


# ----- Collection schema definition -----

class CollectionSchemaDef(BaseModel):
    """Definition of metadata schema for one collection."""
    collection_name: str
    fields: dict[str, str]  # field_name -> type hint "string" | "number"
    schema_hint: str       # Instruction for AI: when to use which filter
    filter_strategy: str   # Short description: e.g. "Scoping by geography and department"


# Standardized field names (use consistent keys across collections where possible)
# location -> city/region; team -> department

SCHEMA_REGISTRY: dict[str, CollectionSchemaDef] = {
    "policy_collection": CollectionSchemaDef(
        collection_name="policy_collection",
        fields={
            "city": "string",
            "department": "string",
        },
        schema_hint=(
            "Use city and department whenever the user mentions a location or a specific team "
            "to ensure they don't see another office's policy."
        ),
        filter_strategy="Scoping by geography and department to avoid Policy Overlap.",
    ),
    "product_catalog_collection": CollectionSchemaDef(
        collection_name="product_catalog_collection",
        fields={
            "product_id": "string",
            "region": "string",
        },
        schema_hint=(
            "If a product code is mentioned (e.g. A99), extract it into product_id. "
            "Always filter by region if the user specifies their location."
        ),
        filter_strategy="Deterministic ID matching (product_id) and region for 100% accuracy.",
    ),
}

# Fallback: unclassified or unknown collections have no required filters
DEFAULT_SCHEMA = CollectionSchemaDef(
    collection_name="unclassified_knowledge",
    fields={},
    schema_hint="No specific filters; use semantic search only.",
    filter_strategy="None",
)

# Optional: normalize extracted values to match stored metadata (avoid zero-result)
# e.g. "New York" -> "NY". Extend per collection/field as needed.
VALUE_NORMALIZERS: dict[str, dict[str, dict[str, str]]] = {
    "policy_collection": {
        "city": {"new york": "NY", "new york city": "NY", "los angeles": "LA", "san francisco": "SF"},
    },
    "product_catalog_collection": {},
}


def get_collection_schema(collection_name: str) -> CollectionSchemaDef:
    """Return the schema for a collection, or the default (no filters) if unknown."""
    return SCHEMA_REGISTRY.get(collection_name, DEFAULT_SCHEMA)


def get_schema_hint_for_rag(collection_name: str) -> str:
    """Return the schema hint for a single collection for injection into RAG system prompt."""
    schema = get_collection_schema(collection_name)
    if not schema.fields:
        return ""
    filters = ", ".join(schema.fields.keys())
    return (
        f"Schema hints (use these filters when the user query implies them):\n"
        f"- {collection_name}: filters [{filters}]. {schema.schema_hint}"
    )


def build_filter_model(collection_name: str) -> type[BaseModel]:
    """
    Dynamic Pydantic model for the chosen collection's filter fields.
    All fields optional so we only filter on what the user mentioned.
    """
    schema = get_collection_schema(collection_name)
    if not schema.fields:
        return create_model("EmptyFilter", __base__=BaseModel)
    field_defs: dict[str, tuple[type, Any]] = {}
    for fname, ftype in schema.fields.items():
        if ftype == "number":
            field_defs[fname] = (Optional[float], None)
        else:
            field_defs[fname] = (Optional[str], None)
    return create_model("ExtractedFilter", **field_defs)


def normalize_filter_values(collection_name: str, filters: dict[str, Any]) -> dict[str, Any]:
    """Apply VALUE_NORMALIZERS so extracted values match stored metadata (None fallback)."""
    norm = VALUE_NORMALIZERS.get(collection_name, {})
    out = {}
    for k, v in filters.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        v_str = str(v).strip().lower()
        if k in norm and v_str in norm[k]:
            out[k] = norm[k][v_str]
        else:
            out[k] = v
    return out


def filters_to_chroma_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert extracted filter dict to Chroma where clause.
    Drops None/empty; if no valid filters, returns None (no filter).
    Chroma format: {"key": {"$eq": value}} or {"$and": [{"key": {"$eq": value}}, ...]}.
    """
    where: list[dict[str, Any]] = []
    for k, v in filters.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        where.append({k: {"$eq": v}})
    if not where:
        return None
    if len(where) == 1:
        return where[0]
    return {"$and": where}
