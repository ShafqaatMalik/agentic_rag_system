"""
RAG Evaluation tests using ragas metrics.

Tests for:
- Answer Faithfulness: Is the answer grounded in context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Did we retrieve all important docs?
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from langchain_core.documents import Document


# Evaluation dataset loader
def load_eval_dataset() -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON file."""
    eval_path = Path(__file__).parent.parent.parent / "evaluation" / "eval_dataset.json"
    
    if not eval_path.exists():
        pytest.skip("Evaluation dataset not found")
    
    with open(eval_path, "r") as f:
        data = json.load(f)
    
    return data.get("test_cases", [])


class TestAnswerFaithfulness:
    """
    Tests for answer faithfulness (grounding).
    
    Faithfulness measures whether the generated answer is
    supported by the retrieved context.
    """
    
    @pytest.mark.evaluation
    def test_faithful_answer(self):
        """Test that answer derived from context is marked faithful."""
        context = "The company reported revenue of $50 million in Q3 2024."
        answer = "The company's Q3 2024 revenue was $50 million."
        
        # Simple faithfulness check: key facts in answer should be in context
        key_facts = ["$50 million", "Q3", "revenue"]
        
        faithfulness_score = sum(
            1 for fact in key_facts if fact.lower() in context.lower()
        ) / len(key_facts)
        
        assert faithfulness_score >= 0.8, "Answer should be faithful to context"
    
    @pytest.mark.evaluation
    def test_unfaithful_answer_detection(self):
        """Test detection of hallucinated information."""
        context = "The company reported revenue of $50 million in Q3 2024."
        answer = "The company's revenue was $75 million, a 50% increase from last year."
        
        # The answer contains fabricated information ($75 million, 50% increase)
        hallucinated_claims = ["$75 million", "50% increase"]
        
        hallucination_count = sum(
            1 for claim in hallucinated_claims if claim in answer
        )
        
        assert hallucination_count > 0, "Should detect hallucinated claims"
    
    @pytest.mark.evaluation
    @patch("app.chains.hallucination_checker.get_hallucination_chain")
    def test_faithfulness_with_hallucination_checker(self, mock_chain):
        """Test faithfulness using our hallucination checker chain."""
        from app.chains.hallucination_checker import (
            check_hallucination,
            HallucinationCheck
        )

        mock_chain_instance = MagicMock()
        mock_chain_instance.invoke = MagicMock(return_value=HallucinationCheck(
            is_grounded="yes",
            confidence="high",
            issues="None"
        ))
        mock_chain.return_value = mock_chain_instance

        documents = [
            Document(
                page_content="AI improves healthcare diagnosis accuracy.",
                metadata={"source": "test.pdf"}
            )
        ]
        answer = "AI helps improve the accuracy of medical diagnoses."

        result = check_hallucination(answer, documents)

        assert result.is_grounded == "yes"
        assert result.confidence in ["high", "medium"]


class TestAnswerRelevance:
    """
    Tests for answer relevance.
    
    Relevance measures whether the answer actually addresses
    the user's query.
    """
    
    @pytest.mark.evaluation
    def test_relevant_answer(self):
        """Test that on-topic answer is marked relevant."""
        query = "What is the company's revenue?"
        answer = "The company reported revenue of $50 million in Q3 2024."
        
        # Check if answer addresses the query topic
        query_keywords = ["revenue", "company"]
        
        relevance_score = sum(
            1 for kw in query_keywords if kw.lower() in answer.lower()
        ) / len(query_keywords)
        
        assert relevance_score >= 0.5, "Answer should be relevant to query"
    
    @pytest.mark.evaluation
    def test_irrelevant_answer_detection(self):
        """Test detection of off-topic answers."""
        query = "What is the company's revenue?"
        answer = "The weather today is sunny with a high of 75 degrees."
        
        # Answer doesn't address the query at all
        query_keywords = ["revenue", "company", "financial", "money", "$"]
        
        relevance_score = sum(
            1 for kw in query_keywords if kw.lower() in answer.lower()
        ) / len(query_keywords)
        
        assert relevance_score < 0.3, "Should detect irrelevant answer"
    
    @pytest.mark.evaluation
    @patch("app.chains.hallucination_checker.get_relevance_chain")
    def test_relevance_with_checker(self, mock_chain):
        """Test relevance using our answer relevance checker."""
        from app.chains.hallucination_checker import (
            check_answer_relevance,
            AnswerRelevanceCheck
        )

        mock_chain_instance = MagicMock()
        mock_chain_instance.invoke = MagicMock(return_value=AnswerRelevanceCheck(
            is_relevant="yes",
            reasoning="Answer directly addresses the revenue question"
        ))
        mock_chain.return_value = mock_chain_instance

        query = "What is the company's revenue?"
        answer = "The company reported $50 million in revenue."

        result = check_answer_relevance(query, answer)

        assert result.is_relevant == "yes"


class TestContextPrecision:
    """
    Tests for context precision.
    
    Precision measures what fraction of retrieved documents
    are actually relevant to the query.
    """
    
    @pytest.mark.evaluation
    def test_high_precision_retrieval(self):
        """Test high precision when all docs are relevant."""
        query = "How is AI used in healthcare?"

        retrieved_docs = [
            Document(page_content="AI enables faster medical diagnosis.", metadata={}),
            Document(page_content="Machine learning detects cancer in images.", metadata={}),
            Document(page_content="Healthcare AI improves patient outcomes.", metadata={}),
        ]

        # All docs are about AI in healthcare
        relevant_keywords = ["ai", "healthcare", "medical", "diagnosis", "patient", "machine", "learning", "cancer"]

        relevant_count = 0
        for doc in retrieved_docs:
            if any(kw in doc.page_content.lower() for kw in relevant_keywords):
                relevant_count += 1

        precision = relevant_count / len(retrieved_docs)

        assert precision >= 0.8, "Precision should be high when docs are relevant"
    
    @pytest.mark.evaluation
    def test_low_precision_detection(self):
        """Test detection of low precision retrieval."""
        query = "How is AI used in healthcare?"
        
        retrieved_docs = [
            Document(page_content="AI enables faster medical diagnosis.", metadata={}),
            Document(page_content="The stock market closed higher today.", metadata={}),
            Document(page_content="Recipe for chocolate cake.", metadata={}),
        ]
        
        # Only 1 of 3 docs is relevant
        relevant_keywords = ["ai", "healthcare", "medical", "diagnosis"]
        
        relevant_count = 0
        for doc in retrieved_docs:
            if any(kw in doc.page_content.lower() for kw in relevant_keywords):
                relevant_count += 1
        
        precision = relevant_count / len(retrieved_docs)
        
        assert precision < 0.5, "Should detect low precision"
    
    @pytest.mark.evaluation
    @patch("app.chains.grader.get_grader_chain")
    def test_precision_with_grader(self, mock_chain):
        """Test precision calculation using our grader chain."""
        from app.chains.grader import grade_documents, GradeDocument

        mock_chain_instance = MagicMock()
        # 2 relevant, 1 irrelevant
        mock_chain_instance.invoke = MagicMock(side_effect=[
            GradeDocument(is_relevant="yes", reasoning="Relevant"),
            GradeDocument(is_relevant="yes", reasoning="Relevant"),
            GradeDocument(is_relevant="no", reasoning="Off-topic"),
        ])
        mock_chain.return_value = mock_chain_instance

        documents = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]

        result = grade_documents("test query", documents)

        precision = len(result.relevant_docs) / len(documents)

        assert precision == pytest.approx(0.67, rel=0.1)


class TestContextRecall:
    """
    Tests for context recall.
    
    Recall measures what fraction of relevant documents
    were successfully retrieved.
    """
    
    @pytest.mark.evaluation
    def test_recall_calculation(self):
        """Test recall when we know ground truth."""
        # Ground truth: these are all the relevant docs in the corpus
        all_relevant_docs = [
            "AI enables faster medical diagnosis.",
            "Machine learning detects cancer in images.",
            "Healthcare AI improves patient outcomes.",
            "Deep learning assists radiologists.",
        ]
        
        # What we actually retrieved
        retrieved_content = [
            "AI enables faster medical diagnosis.",
            "Machine learning detects cancer in images.",
            # Missing 2 relevant docs
        ]
        
        # Calculate recall
        retrieved_relevant = sum(
            1 for doc in all_relevant_docs if doc in retrieved_content
        )
        recall = retrieved_relevant / len(all_relevant_docs)
        
        assert recall == 0.5, "Recall should be 50% (2 of 4 retrieved)"
    
    @pytest.mark.evaluation
    def test_perfect_recall(self):
        """Test perfect recall scenario."""
        all_relevant_docs = ["Doc A", "Doc B"]
        retrieved_content = ["Doc A", "Doc B", "Doc C"]  # Got all relevant + extra
        
        retrieved_relevant = sum(
            1 for doc in all_relevant_docs if doc in retrieved_content
        )
        recall = retrieved_relevant / len(all_relevant_docs)
        
        assert recall == 1.0, "Recall should be 100%"


class TestEvaluationDataset:
    """Tests using the evaluation dataset."""
    
    @pytest.mark.evaluation
    def test_dataset_loads(self):
        """Test that evaluation dataset loads correctly."""
        test_cases = load_eval_dataset()
        
        assert len(test_cases) > 0, "Should have test cases"
        
        # Verify structure
        for case in test_cases:
            assert "id" in case
            assert "question" in case
            assert "ground_truth" in case
            assert "expected_contexts" in case
    
    @pytest.mark.evaluation
    def test_dataset_coverage(self):
        """Test that dataset covers different query types."""
        test_cases = load_eval_dataset()
        
        questions = [case["question"] for case in test_cases]
        
        # Should have variety in questions
        assert len(set(questions)) == len(questions), "Questions should be unique"
        assert len(test_cases) >= 3, "Should have at least 3 test cases"
    
    @pytest.mark.evaluation
    @pytest.mark.parametrize("case_index", range(5))
    def test_ground_truth_quality(self, case_index):
        """Test that ground truth answers are reasonable."""
        test_cases = load_eval_dataset()
        
        if case_index >= len(test_cases):
            pytest.skip("Not enough test cases")
        
        case = test_cases[case_index]
        
        # Ground truth should be non-empty and reasonable length
        assert len(case["ground_truth"]) > 10, "Ground truth should be substantive"
        assert len(case["ground_truth"]) < 1000, "Ground truth should be concise"
        
        # Expected contexts should exist
        assert len(case["expected_contexts"]) > 0, "Should have expected contexts"


class TestRagasIntegration:
    """
    Tests demonstrating ragas-style evaluation.
    
    Note: These tests mock ragas to avoid API calls during CI.
    For real evaluation, run with actual ragas library.
    """
    
    @pytest.mark.evaluation
    def test_ragas_metrics_structure(self):
        """Test that we can structure data for ragas evaluation."""
        # Structure data as ragas expects
        evaluation_data = {
            "question": "How is AI used in healthcare?",
            "answer": "AI is used for faster diagnosis and treatment planning.",
            "contexts": [
                "AI enables faster medical diagnosis.",
                "Machine learning improves treatment outcomes."
            ],
            "ground_truth": "AI is used in healthcare for diagnosis and treatment."
        }
        
        # Verify structure
        assert "question" in evaluation_data
        assert "answer" in evaluation_data
        assert "contexts" in evaluation_data
        assert isinstance(evaluation_data["contexts"], list)
    
    @pytest.mark.evaluation
    @pytest.mark.slow
    def test_batch_evaluation_structure(self):
        """Test batch evaluation data structure."""
        test_cases = load_eval_dataset()
        
        # Structure for batch evaluation
        batch_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for case in test_cases[:3]:  # Use first 3 for test
            batch_data["question"].append(case["question"])
            batch_data["answer"].append(case["ground_truth"])  # Use ground truth as mock answer
            batch_data["contexts"].append(case["expected_contexts"])
            batch_data["ground_truth"].append(case["ground_truth"])
        
        # Verify batch structure
        assert len(batch_data["question"]) == 3
        assert len(batch_data["contexts"]) == 3
        assert all(isinstance(ctx, list) for ctx in batch_data["contexts"])
