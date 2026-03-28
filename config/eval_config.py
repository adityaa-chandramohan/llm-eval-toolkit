from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class ThresholdConfig(BaseModel):
    faithfulness: float = float(os.getenv("FAITHFULNESS_THRESHOLD", 0.80))
    answer_relevance: float = float(os.getenv("RELEVANCE_THRESHOLD", 0.75))
    context_precision: float = 0.75
    context_recall: float = 0.70
    hallucination: float = float(os.getenv("HALLUCINATION_THRESHOLD", 0.15))  # lower = better
    toxicity: float = 0.10                                                      # lower = better
    bias: float = 0.10


class LLMConfig(BaseModel):
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    vertex_model: str = os.getenv("VERTEX_MODEL", "gemini-1.5-pro")
    gcp_project: Optional[str] = os.getenv("GCP_PROJECT_ID")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")


class EvalConfig(BaseModel):
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", 10))
    output_dir: str = os.getenv("EVAL_OUTPUT_DIR", "reports/output")
    save_results: bool = True
    verbose: bool = True


# Singleton config instance
config = EvalConfig()
