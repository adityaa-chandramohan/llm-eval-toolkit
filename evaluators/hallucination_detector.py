"""
Hallucination Detector
-----------------------
Standalone hallucination detection using LLM-as-judge pattern.
Works without a full RAG pipeline — useful for spot-checking any LLM output.
"""

from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

from config.eval_config import config
from utils.logger import get_logger

logger = get_logger(__name__)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert fact-checker evaluating whether an AI response contains hallucinations.

CONTEXT (ground truth information):
{context}

AI RESPONSE to evaluate:
{response}

QUESTION that was asked:
{question}

Evaluate the response strictly against the context provided.
Respond in JSON format:
{{
  "hallucination_detected": true/false,
  "confidence": 0.0-1.0,
  "hallucinated_claims": ["claim1", "claim2"],
  "explanation": "brief explanation"
}}

Only flag claims that contradict or are entirely absent from the context.
""")


@dataclass
class HallucinationResult:
    hallucination_detected: bool
    confidence: float
    hallucinated_claims: list[str]
    explanation: str
    score: float  # 0.0 = fully hallucinated, 1.0 = no hallucination


class HallucinationDetector:
    """
    LLM-as-judge hallucination detector.
    Supports OpenAI GPT and Google Vertex AI as judge models.
    """

    def __init__(self, provider: str = "openai"):
        self.llm = self._init_llm(provider)
        self.chain = HALLUCINATION_PROMPT | self.llm
        self.threshold = config.thresholds.hallucination

    def _init_llm(self, provider: str):
        if provider == "vertex":
            return ChatVertexAI(
                model_name=config.llm.vertex_model,
                project=config.llm.gcp_project,
                location=config.llm.gcp_location,
                temperature=0,
            )
        return ChatOpenAI(model=config.llm.openai_model, temperature=0)

    def detect(self, question: str, response: str, context: str) -> HallucinationResult:
        import json

        logger.debug(f"Checking hallucination for: {question[:60]}...")

        raw = self.chain.invoke({
            "question": question,
            "response": response,
            "context": context,
        })

        try:
            content = raw.content if hasattr(raw, "content") else str(raw)
            # Strip markdown code fences if present
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse judge response: {e}")
            data = {"hallucination_detected": False, "confidence": 0.0,
                    "hallucinated_claims": [], "explanation": "Parse error"}

        detected = data.get("hallucination_detected", False)
        claims   = data.get("hallucinated_claims", [])
        score    = 1.0 - (len(claims) * 0.2) if detected else 1.0
        score    = max(0.0, min(1.0, score))

        return HallucinationResult(
            hallucination_detected=detected,
            confidence=data.get("confidence", 0.0),
            hallucinated_claims=claims,
            explanation=data.get("explanation", ""),
            score=score,
        )

    def detect_batch(
        self,
        samples: list[dict],  # each: {question, response, context}
    ) -> list[HallucinationResult]:
        results = [
            self.detect(s["question"], s["response"], s["context"])
            for s in samples
        ]
        flagged = sum(1 for r in results if r.hallucination_detected)
        logger.info(f"Hallucination check — {flagged}/{len(results)} flagged")
        return results
