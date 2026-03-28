"""
RAGAS evaluation test suite.
Run: pytest tests/test_ragas.py -v --alluredir=reports/allure-results
"""

import json
import pytest
import allure
from evaluators.ragas_evaluator import RAGASEvaluator, RAGSample


@pytest.fixture(scope="module")
def evaluator():
    return RAGASEvaluator(provider="openai")


@pytest.fixture(scope="module")
def rag_samples():
    with open("data/sample_datasets/rag_samples.json") as f:
        raw = json.load(f)
    return [RAGSample(**s) for s in raw]


@allure.feature("RAG Evaluation")
@allure.story("RAGAS Metrics")
@allure.severity(allure.severity_level.CRITICAL)
class TestRAGASEvaluation:

    def test_faithfulness_above_threshold(self, evaluator, rag_samples):
        """All RAG responses must be faithful to retrieved contexts."""
        results = evaluator.evaluate(rag_samples)
        failures = [
            f"Sample {i}: {r.failures}"
            for i, r in enumerate(results)
            if r.faithfulness < evaluator.thresholds.faithfulness
        ]
        assert not failures, f"Faithfulness threshold violations:\n" + "\n".join(failures)

    def test_answer_relevance_above_threshold(self, evaluator, rag_samples):
        """Answers must be relevant to the posed questions."""
        results = evaluator.evaluate(rag_samples)
        failures = [
            f"Sample {i}: relevance={r.answer_relevance:.2f}"
            for i, r in enumerate(results)
            if r.answer_relevance < evaluator.thresholds.answer_relevance
        ]
        assert not failures, f"Answer relevance violations:\n" + "\n".join(failures)

    def test_context_precision_above_threshold(self, evaluator, rag_samples):
        """Retrieved contexts must be precise and on-topic."""
        results = evaluator.evaluate(rag_samples)
        failures = [
            f"Sample {i}: context_precision={r.context_precision:.2f}"
            for i, r in enumerate(results)
            if r.context_precision < evaluator.thresholds.context_precision
        ]
        assert not failures, f"Context precision violations:\n" + "\n".join(failures)

    def test_no_harmful_content(self, evaluator, rag_samples):
        """RAG outputs must not contain harmful content."""
        results = evaluator.evaluate(rag_samples)
        failures = [
            f"Sample {i}: harmfulness={r.harmfulness:.2f}"
            for i, r in enumerate(results)
            if r.harmfulness > evaluator.thresholds.toxicity
        ]
        assert not failures, f"Harmful content detected:\n" + "\n".join(failures)

    def test_overall_pass_rate(self, evaluator, rag_samples):
        """At least 80% of RAG samples must pass all metrics."""
        results = evaluator.evaluate(rag_samples)
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        assert pass_rate >= 0.8, f"Pass rate {pass_rate:.0%} below 80% threshold"
