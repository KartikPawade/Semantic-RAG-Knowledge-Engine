"""Filter extraction: empty schema returns None; mock LLM returning JSON produces where clause."""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.filter_extraction import extract_filters_from_query


def test_extract_filters_empty_schema_returns_none():
    """When collection has no schema fields, we return None and do not call LLM."""
    llm = MagicMock()
    result = extract_filters_from_query("policy in NY", "unclassified_knowledge", llm)
    assert result is None
    llm.invoke.assert_not_called()


def test_extract_filters_with_mock_llm_returning_json():
    """When LLM returns valid JSON for schema fields, we get a Chroma where clause."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"city": "NY", "department": "HR"}')
    result = extract_filters_from_query("HR policy in New York", "policy_collection", llm)
    assert result is not None
    assert result == {"$and": [{"city": {"$eq": "NY"}}, {"department": {"$eq": "HR"}}]}
