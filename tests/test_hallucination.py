"""
Hallucination detection test suite.
Run: pytest tests/test_hallucination.py -v --alluredir=reports/allure-results
"""

import pytest
import allure
from evaluators.hallucination_detector import HallucinationDetector


@pytest.fixture(scope="module")
def detector():
    return HallucinationDetector(provider="openai")


@allure.feature("LLM Evaluation")
@allure.story("Hallucination Detection")
class TestHallucinationDetection:

    @allure.severity(allure.severity_level.CRITICAL)
    def test_grounded_response_passes(self, detector):
        """A response fully grounded in context must not be flagged."""
        result = detector.detect(
            question="What is our SLA uptime guarantee?",
            response="Our SLA guarantees 99.9% uptime measured on a monthly basis.",
            context="The service level agreement guarantees 99.9% uptime, calculated on a rolling monthly basis.",
        )
        assert not result.hallucination_detected, \
            f"False positive — grounded response flagged. Claims: {result.hallucinated_claims}"

    @allure.severity(allure.severity_level.CRITICAL)
    def test_fabricated_claim_is_detected(self, detector):
        """A response containing a claim not in the context must be flagged."""
        result = detector.detect(
            question="What encryption standard do we use?",
            response="We use AES-256 encryption and also support quantum-resistant algorithms.",
            context="All data at rest and in transit is encrypted using AES-256.",
        )
        assert result.hallucination_detected, \
            "Expected hallucination (quantum-resistant claim) was not detected"

    @allure.severity(allure.severity_level.NORMAL)
    def test_batch_detection_flags_correct_count(self, detector):
        """Batch detection should flag only the samples with hallucinations."""
        samples = [
            {
                "question": "What is the refund window?",
                "response": "You have 30 days to request a refund.",
                "context": "Customers may request a refund within 30 days of purchase.",
            },
            {
                "question": "What is the refund window?",
                "response": "You have 60 days to request a refund and can also get store credit.",
                "context": "Customers may request a refund within 30 days of purchase.",
            },
        ]
        results = detector.detect_batch(samples)
        flagged = [r for r in results if r.hallucination_detected]
        assert len(flagged) == 1, f"Expected 1 flagged, got {len(flagged)}"

    @allure.severity(allure.severity_level.NORMAL)
    def test_hallucination_score_range(self, detector):
        """Hallucination score must always be between 0.0 and 1.0."""
        result = detector.detect(
            question="Test question",
            response="Test response",
            context="Test context",
        )
        assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
