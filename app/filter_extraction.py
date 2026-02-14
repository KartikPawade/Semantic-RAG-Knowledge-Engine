"""
Schema-aware filter extraction from user query (Step 2 of two-step extraction).

After collection routing, this module uses the collection's schema from the
registry to extract filter values (city, department, product_id, region) from
the user query via LLM. Validates/normalizes and returns Chroma where clause
or None (None fallback to avoid zero-result errors).
"""
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.prompts import EXTRACT_FILTER_PROMPT
from app.schema_registry import (
    build_filter_model,
    filters_to_chroma_where,
    get_collection_schema,
    normalize_filter_values,
)


def extract_filters_from_query(
    user_query: str,
    collection_name: str,
    llm: BaseChatModel,
) -> dict[str, Any] | None:
    """
    Extract metadata filters from the user query using the collection's schema.
    Returns a Chroma where clause dict, or None if no valid filters (so search
    runs without filter and avoids zero-result errors).
    """
    schema = get_collection_schema(collection_name)
    if not schema.fields:
        return None
    field_names = ", ".join(schema.fields.keys())
    chain = EXTRACT_FILTER_PROMPT | llm | StrOutputParser()
    raw = (chain.invoke({
        "field_names": field_names,
        "schema_hint": schema.schema_hint,
        "user_query": (user_query or "").strip()[:2000],
    }) or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw).strip()
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
    except json.JSONDecodeError:
        return None
    # Validate with dynamic Pydantic model (drops unknown keys, validates types)
    try:
        model_cls = build_filter_model(collection_name)
        # Only pass keys that exist in schema
        payload = {k: v for k, v in data.items() if k in schema.fields}
        if not payload:
            return None
        instance = model_cls(**payload)
        filters = instance.model_dump(exclude_none=True)
    except Exception:
        return None
    filters = normalize_filter_values(collection_name, filters)
    return filters_to_chroma_where(filters)
