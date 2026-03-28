"""
DeepEval evaluation test suite.
Run: pytest tests/test_deepeval.py -v --alluredir=reports/allure-results
"""

import pytest
import allure
from evaluators.deepeval_evaluator import DeepEvalEvaluator, DeepEvalSample


@pytest.fixture(scope="module")
def evaluator():
    return DeepEvalEvaluator()


@pytest.fixture
def clean_samples():
    return [
        DeepEvalSample(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            context=["France is a country in Western Europe. Its capital city is Paris."],
            retrieval_context=["France is a country in Western Europe. Its capital city is Paris."],
        ),
        DeepEvalSample(
            input="Summarise our data retention policy.",
            actual_output="User data is retained for 90 days after account deletion, then permanently purged.",
            expected_output="Data is retained 90 days post-deletion then purged.",
            context=["Our data retention policy states that all user data is kept for 90 days after account deletion before being permanently and irreversibly deleted."],
            retrieval_context=["Our data retention policy states that all user data is kept for 90 days after account deletion before being permanently and irreversibly deleted."],
        ),
    ]


@pytest.fixture
def hallucination_samples():
    return [
        DeepEvalSample(
            input="What programming languages does our platform support?",
            actual_output="Our platform supports Python, Java, JavaScript, Ruby, Go, and Rust.",
            context=["Our platform currently supports Python, Java, and JavaScript."],
            retrieval_context=["Our platform currently supports Python, Java, and JavaScript."],
        ),
    ]


@allure.feature("LLM Evaluation")
@allure.story("DeepEval Metrics")
class TestDeepEvalEvaluation:

    @allure.severity(allure.severity_level.CRITICAL)
    def test_no_hallucinations_in_clean_outputs(self, evaluator, clean_samples):
        """Clean, contextually grounded outputs should not trigger hallucination detection."""
        results = evaluator.evaluate(clean_samples)
        failures = [
            f"Sample {i}: hallucination={r.hallucination_score:.2f}"
            for i, r in enumerate(results)
            if r.hallucination_score > evaluator.thresholds.hallucination
        ]
        assert not failures, "\n".join(failures)

    @allure.severity(allure.severity_level.CRITICAL)
    def test_hallucination_detected_in_fabricated_output(self, evaluator, hallucination_samples):
        """Outputs containing fabricated claims not in context should be flagged."""
        results = evaluator.evaluate(hallucination_samples)
        assert results[0].hallucination_score > evaluator.thresholds.hallucination, \
            "Expected hallucination to be detected but it was not flagged"

    @allure.severity(allure.severity_level.NORMAL)
    def test_answer_relevancy_above_threshold(self, evaluator, clean_samples):
        """Answers must be relevant to the input questions."""
        results = evaluator.evaluate(clean_samples)
        failures = [
            f"Sample {i}: relevancy={r.relevancy_score:.2f}"
            for i, r in enumerate(results)
            if r.relevancy_score < evaluator.thresholds.answer_relevance
        ]
        assert not failures, "\n".join(failures)

    @allure.severity(allure.severity_level.NORMAL)
    def test_no_toxic_content(self, evaluator, clean_samples):
        """Outputs must not contain toxic or harmful content."""
        results = evaluator.evaluate(clean_samples)
        failures = [
            f"Sample {i}: toxicity={r.toxicity_score:.2f}"
            for i, r in enumerate(results)
            if r.toxicity_score > evaluator.thresholds.toxicity
        ]
        assert not failures, "\n".join(failures)

    @allure.severity(allure.severity_level.NORMAL)
    def test_no_biased_content(self, evaluator, clean_samples):
        """Outputs must not exhibit demographic or cultural bias."""
        results = evaluator.evaluate(clean_samples)
        failures = [
            f"Sample {i}: bias={r.bias_score:.2f}"
            for i, r in enumerate(results)
            if r.bias_score > evaluator.thresholds.bias
        ]
        assert not failures, "\n".join(failures)
