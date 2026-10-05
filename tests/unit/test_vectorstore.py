"""
Unit tests for merged multi-query retrieval.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.retrieval.vectorstore import VectorStoreManager


def manager_with_results(results: dict[str, list[str]]) -> VectorStoreManager:
    """A VectorStoreManager whose similarity search returns the given chunk texts per query."""
    manager = VectorStoreManager()
    manager._vectorstore = MagicMock()
    manager._vectorstore.similarity_search.side_effect = lambda q, k: [
        Document(page_content=text) for text in results[q][:k]
    ]
    return manager


class TestRetrieveForQueries:
    """Tests for retrieve_for_queries."""

    @pytest.mark.unit
    def test_interleaves_by_rank(self):
        manager = manager_with_results({"a": ["a1", "a2"], "b": ["b1", "b2"]})

        docs = manager.retrieve_for_queries(["a", "b"], k=2)

        assert [d.page_content for d in docs] == ["a1", "b1", "a2", "b2"]

    @pytest.mark.unit
    def test_removes_duplicates_keeping_first(self):
        manager = manager_with_results({"a": ["x", "a2"], "b": ["x", "b2"], "c": ["a2", "c2"]})

        docs = manager.retrieve_for_queries(["a", "b", "c"], k=2)

        assert [d.page_content for d in docs] == ["x", "a2", "b2", "c2"]

    @pytest.mark.unit
    def test_at_most_k_per_query(self):
        results = {q: [f"{q}{i}" for i in range(10)] for q in ("a", "b", "c")}
        manager = manager_with_results(results)

        docs = manager.retrieve_for_queries(["a", "b", "c"], k=4)

        assert len(docs) == 12
        for q in ("a", "b", "c"):
            manager._vectorstore.similarity_search.assert_any_call(q, k=4)

    @pytest.mark.unit
    def test_defaults_to_retrieval_k(self):
        manager = manager_with_results({"a": [f"a{i}" for i in range(10)]})

        docs = manager.retrieve_for_queries(["a"])

        assert len(docs) == manager.settings.retrieval_k
