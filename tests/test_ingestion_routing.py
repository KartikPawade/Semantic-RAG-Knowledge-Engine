"""Test collection name normalization used in classify_query_to_collection."""
import pytest
from app.ingestion import _normalize_collection_name


def test_normalize_lowercase():
    assert _normalize_collection_name("Policy_Collection") == "policy_collection"


def test_normalize_spaces_to_underscores():
    assert _normalize_collection_name("Policy Collection") == "policy_collection"


def test_normalize_hyphens_to_underscores():
    assert _normalize_collection_name("policy-collection") == "policycollection"


def test_normalize_strips_punctuation():
    assert _normalize_collection_name("policy_collection!") == "policy_collection"


def test_normalize_mixed_space_and_underscore():
    assert _normalize_collection_name("Policy  Collection") == "policy_collection"


def test_normalize_empty():
    assert _normalize_collection_name("") == ""
    assert _normalize_collection_name("   ") == ""


def test_normalize_leading_trailing_underscores_stripped():
    assert _normalize_collection_name("_policy_collection_") == "policy_collection"
