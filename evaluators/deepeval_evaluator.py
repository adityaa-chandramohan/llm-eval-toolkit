"""
DeepEval Evaluator
------------------
Evaluates LLM outputs using DeepEval metrics.
Metrics: Hallucination, Answer Relevancy, Faithfulness, Toxicity, Bias, Summarisation
"""

from dataclasses import dataclass, field
from typing import Optional
from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric,
    BiasMetric,
    SummarizationMetric,
    GEval,
)

from config.eval_config import config
from utils.logger import get_logger
from utils.result_writer import write_results

logger = get_logger(__name__)


@dataclass
class DeepEvalSample:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[list[str]] = None
    retrieval_context: Optional[list[str]] = None


@dataclass
class DeepEvalResult:
    hallucination_score: float
    relevancy_score: float
    faithfulness_score: float
    toxicity_score: float
    bias_score: float
    passed: bool
    failures: list[str] = field(default_factory=list)


class DeepEvalEvaluator:
    """
    Evaluates LLM outputs using DeepEval.
    Supports hallucination detection, relevancy, faithfulness, toxicity, and bias.
    """

    def __init__(self, model: str = None):
        self.model = GPTModel(model=model or config.llm.openai_model)
        self.thresholds = config.thresholds

    def _build_metrics(self):
        return [
            HallucinationMetric(threshold=self.thresholds.hallucination,   model=self.model),
            AnswerRelevancyMetric(threshold=self.thresholds.answer_relevance, model=self.model),
            FaithfulnessMetric(threshold=self.thresholds.faithfulness,     model=self.model),
            ToxicityMetric(threshold=self.thresholds.toxicity,             model=self.model),
            BiasMetric(threshold=self.thresholds.bias,                     model=self.model),
        ]

    def _build_test_cases(self, samples: list[DeepEvalSample]) -> list[LLMTestCase]:
        return [
            LLMTestCase(
                input=s.input,
                actual_output=s.actual_output,
                expected_output=s.expected_output,
                context=s.context,
                retrieval_context=s.retrieval_context,
            )
            for s in samples
        ]

    def evaluate(self, samples: list[DeepEvalSample]) -> list[DeepEvalResult]:
        logger.info(f"Running DeepEval evaluation on {len(samples)} samples...")

        test_cases = self._build_test_cases(samples)
        metrics    = self._build_metrics()

        evaluate(test_cases, metrics)

        results = []
        for tc in test_cases:
            failures = []
            scores = {m.name: m.score for m in tc.metrics_data} if tc.metrics_data else {}

            h  = scores.get("Hallucination",    0.0)
            ar = scores.get("Answer Relevancy", 0.0)
            f  = scores.get("Faithfulness",     0.0)
            t  = scores.get("Toxicity",         0.0)
            b  = scores.get("Bias",             0.0)

            if h  > self.thresholds.hallucination:    failures.append(f"Hallucination {h:.2f} > {self.thresholds.hallucination}")
            if ar < self.thresholds.answer_relevance: failures.append(f"Answer Relevancy {ar:.2f} < {self.thresholds.answer_relevance}")
            if f  < self.thresholds.faithfulness:     failures.append(f"Faithfulness {f:.2f} < {self.thresholds.faithfulness}")
            if t  > self.thresholds.toxicity:         failures.append(f"Toxicity {t:.2f} > {self.thresholds.toxicity}")
            if b  > self.thresholds.bias:             failures.append(f"Bias {b:.2f} > {self.thresholds.bias}")

            results.append(DeepEvalResult(
                hallucination_score=h,
                relevancy_score=ar,
                faithfulness_score=f,
                toxicity_score=t,
                bias_score=b,
                passed=len(failures) == 0,
                failures=failures,
            ))

        passed = sum(1 for r in results if r.passed)
        logger.info(f"DeepEval complete — {passed}/{len(results)} passed")
        return results

    def custom_geval(
        self,
        samples: list[DeepEvalSample],
        name: str,
        criteria: str,
        evaluation_params: list[LLMTestCaseParams],
        threshold: float = 0.7,
    ) -> list[DeepEvalResult]:
        """Run a custom G-Eval metric — useful for domain-specific criteria."""
        metric = GEval(
            name=name,
            criteria=criteria,
            evaluation_params=evaluation_params,
            threshold=threshold,
            model=self.model,
        )
        test_cases = self._build_test_cases(samples)
        evaluate(test_cases, [metric])

        results = []
        for tc in test_cases:
            score = tc.metrics_data[0].score if tc.metrics_data else 0.0
            passed = score >= threshold
            results.append(DeepEvalResult(
                hallucination_score=0.0,
                relevancy_score=score,
                faithfulness_score=0.0,
                toxicity_score=0.0,
                bias_score=0.0,
                passed=passed,
                failures=[] if passed else [f"{name} score {score:.2f} < {threshold}"],
            ))
        return results
