import pytest
from app.schema_registry import (
    build_filter_model,
    filters_to_chroma_where,
    get_collection_schema,
    get_schema_hint_for_rag,
    normalize_filter_values,
)


def test_filters_to_chroma_where_single():
    result = filters_to_chroma_where({"city": "NY"})
    assert result == {"city": {"$eq": "NY"}}


def test_filters_to_chroma_where_multi():
    result = filters_to_chroma_where({"city": "NY", "department": "HR"})
    assert result == {"$and": [{"city": {"$eq": "NY"}}, {"department": {"$eq": "HR"}}]}


def test_filters_to_chroma_where_empty():
    assert filters_to_chroma_where({}) is None


def test_filters_to_chroma_where_drops_none():
    assert filters_to_chroma_where({"city": None, "department": "HR"}) == {
        "department": {"$eq": "HR"}
    }


def test_normalize_new_york():
    result = normalize_filter_values("policy_collection", {"city": "New York"})
    assert result["city"] == "NY"


def test_normalize_unknown_value_unchanged():
    result = normalize_filter_values("policy_collection", {"city": "Boston"})
    assert result["city"] == "Boston"


def test_build_filter_model_policy():
    model_cls = build_filter_model("policy_collection")
    instance = model_cls(city="NY", department="HR")
    assert instance.model_dump(exclude_none=True) == {"city": "NY", "department": "HR"}


def test_build_filter_model_drops_unknown_keys():
    model_cls = build_filter_model("policy_collection")
    instance = model_cls(city="NY", department="HR")
    dump = instance.model_dump(exclude_none=True)
    assert "city" in dump
    assert "department" in dump
    assert len(dump) == 2


def test_build_filter_model_unknown_collection():
    model_cls = build_filter_model("nonexistent_collection")
    instance = model_cls()
    assert instance.model_dump(exclude_none=True) == {}


def test_get_schema_hint_for_rag_unknown_collection():
    hint = get_schema_hint_for_rag("nonexistent_collection")
    assert hint == ""


def test_get_schema_hint_for_rag_policy():
    hint = get_schema_hint_for_rag("policy_collection")
    assert "policy_collection" in hint
    assert "city" in hint
    assert "department" in hint
    assert "Schema hints" in hint


def test_get_collection_schema_unknown():
    schema = get_collection_schema("unknown")
    assert schema.collection_name == "unclassified_knowledge"
    assert schema.fields == {}
