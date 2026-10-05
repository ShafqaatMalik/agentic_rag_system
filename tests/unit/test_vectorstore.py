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


class TestRetrieveForQuestion:
    """Complex path: the original question is retrieved for as well as its sub-queries."""

    @pytest.mark.unit
    def test_question_hits_come_first(self):
        manager = manager_with_results({"Q": ["q1", "q2"], "a": ["a1", "a2"], "b": ["b1", "b2"]})

        docs = manager.retrieve_for_question("Q", ["a", "b"], k=2)

        assert [d.page_content for d in docs] == ["q1", "a1", "b1", "q2", "a2", "b2"]

    @pytest.mark.unit
    def test_question_is_not_searched_twice(self):
        manager = manager_with_results({"Q": ["q1"], "a": ["a1"]})

        manager.retrieve_for_question("Q", [" q ", "a"], k=1)

        searched = [c.args[0] for c in manager._vectorstore.similarity_search.call_args_list]
        assert searched == ["Q", "a"]

    @pytest.mark.unit
    def test_capped_keeping_top_ranks_of_every_query(self):
        from app.retrieval.vectorstore import MAX_MERGED_CHUNKS

        results = {q: [f"{q}{i}" for i in range(1, 5)] for q in ("Q", "a", "b", "c")}
        manager = manager_with_results(results)

        docs = manager.retrieve_for_question("Q", ["a", "b", "c"], k=4)

        texts = [d.page_content for d in docs]
        assert MAX_MERGED_CHUNKS == 12
        assert len(texts) == 12  # 16 candidates, capped
        for q in ("Q", "a", "b", "c"):
            assert {f"{q}1", f"{q}2", f"{q}3"} <= set(texts)
            assert f"{q}4" not in texts

    @pytest.mark.unit
    def test_max_docs_cap(self):
        manager = manager_with_results({"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"]})

        docs = manager.retrieve_for_queries(["a", "b"], k=3, max_docs=4)

        assert [d.page_content for d in docs] == ["a1", "b1", "a2", "b2"]
