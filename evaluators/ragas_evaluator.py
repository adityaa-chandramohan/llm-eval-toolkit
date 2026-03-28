"""
RAGAS Evaluator
---------------
Evaluates RAG pipelines using the RAGAS framework.
Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall
"""

from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_vertexai import ChatVertexAI

from config.eval_config import config
from utils.logger import get_logger
from utils.result_writer import write_results

logger = get_logger(__name__)


@dataclass
class RAGSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    harmfulness: float
    passed: bool
    failures: list[str] = field(default_factory=list)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.
    Supports OpenAI and Vertex AI as judge LLMs.
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm = self._init_llm()
        self.embeddings = OpenAIEmbeddings()
        self.thresholds = config.thresholds

    def _init_llm(self):
        if self.provider == "vertex":
            return ChatVertexAI(
                model_name=config.llm.vertex_model,
                project=config.llm.gcp_project,
                location=config.llm.gcp_location,
            )
        return ChatOpenAI(model=config.llm.openai_model, temperature=0)

    def _build_dataset(self, samples: list[RAGSample]) -> Dataset:
        return Dataset.from_dict({
            "question":    [s.question     for s in samples],
            "answer":      [s.answer       for s in samples],
            "contexts":    [s.contexts     for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        })

    def evaluate(self, samples: list[RAGSample]) -> list[RAGASResult]:
        logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")

        dataset = self._build_dataset(samples)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, harmfulness],
            llm=self.llm,
            embeddings=self.embeddings,
        )

        df = scores.to_pandas()
        results = []

        for _, row in df.iterrows():
            failures = []
            f  = row.get("faithfulness",        0.0)
            ar = row.get("answer_relevancy",     0.0)
            cp = row.get("context_precision",    0.0)
            cr = row.get("context_recall",       0.0)
            h  = row.get("harmfulness",          0.0)

            if f  < self.thresholds.faithfulness:      failures.append(f"Faithfulness {f:.2f} < {self.thresholds.faithfulness}")
            if ar < self.thresholds.answer_relevance:  failures.append(f"Answer Relevance {ar:.2f} < {self.thresholds.answer_relevance}")
            if cp < self.thresholds.context_precision: failures.append(f"Context Precision {cp:.2f} < {self.thresholds.context_precision}")
            if cr < self.thresholds.context_recall:    failures.append(f"Context Recall {cr:.2f} < {self.thresholds.context_recall}")
            if h  > self.thresholds.toxicity:          failures.append(f"Harmfulness {h:.2f} > {self.thresholds.toxicity}")

            results.append(RAGASResult(
                faithfulness=f,
                answer_relevance=ar,
                context_precision=cp,
                context_recall=cr,
                harmfulness=h,
                passed=len(failures) == 0,
                failures=failures,
            ))

        if config.save_results:
            write_results("ragas", df)

        passed = sum(1 for r in results if r.passed)
        logger.info(f"RAGAS complete — {passed}/{len(results)} passed")
        return results
