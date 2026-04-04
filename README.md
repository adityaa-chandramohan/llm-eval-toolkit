# llm-eval-toolkit

A modular Python toolkit for evaluating large language models in production — covering faithfulness, answer relevance, context precision, hallucination detection, and bias/fairness assessment.

Designed for CI/CD integration and enterprise RAG pipeline validation. Supports both **RAGAS** and **DeepEval** backends so teams can run the evaluation suite that fits their existing stack.

---

## What it evaluates

| Metric | Backend | Description |
|--------|---------|-------------|
| Faithfulness | RAGAS | Is the answer grounded in the retrieved context? |
| Answer Relevance | RAGAS | Does the answer actually address the question? |
| Context Precision | RAGAS | Are the retrieved chunks relevant to the question? |
| Context Recall | RAGAS | Did retrieval surface all the necessary information? |
| Hallucination | Custom + DeepEval | Does the answer contain claims not supported by context? |
| Bias & Fairness | Custom | Does output vary inappropriately across demographic groups? |

---

## Project Structure

```
llm-eval-toolkit/
├── evaluators/
│   ├── ragas_evaluator.py         # RAGAS metrics pipeline
│   ├── deepeval_evaluator.py      # DeepEval metrics pipeline
│   ├── hallucination_detector.py  # Hallucination scoring
│   └── bias_fairness_evaluator.py # Bias & fairness checks
├── tests/
│   ├── test_ragas.py
│   ├── test_deepeval.py
│   └── test_hallucination.py
├── config/                        # Evaluation configuration
├── data/                          # Sample datasets and test cases
├── utils/                         # Shared helpers
├── reports/                       # Allure + HTML output
├── run_eval.py                    # CLI entry point
├── pytest.ini
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI or Vertex AI credentials (set in `.env`)

### Install

```bash
git clone https://github.com/adityaa-chandramohan/llm-eval-toolkit.git
cd llm-eval-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

### Run evaluations

```bash
# Full suite
pytest tests/ -v --alluredir=reports/allure-results

# RAGAS only
pytest tests/test_ragas.py -v

# Hallucination only
pytest tests/test_hallucination.py -v

# Generate Allure report
allure serve reports/allure-results
```

---

## CI/CD Integration

Add to your pipeline to gate releases on evaluation quality thresholds:

```yaml
# GitHub Actions example
- name: Run LLM Evaluations
  run: |
    pip install -r requirements.txt
    pytest tests/ --tb=short -q

- name: Upload Allure Results
  uses: actions/upload-artifact@v4
  with:
    name: allure-results
    path: reports/allure-results/
```

---

## Configuration

Create a `.env` file with your credentials:

```env
OPENAI_API_KEY=sk-...
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
VERTEX_PROJECT=your-gcp-project
VERTEX_LOCATION=us-central1
```

Evaluation thresholds and model selection are configured in `config/`.

---

## Stack

- **RAGAS** — RAG evaluation metrics (faithfulness, relevance, precision, recall)
- **DeepEval** — LLM unit testing framework with pytest integration
- **LangChain** — LLM abstraction layer for evaluation pipelines
- **pytest + allure-pytest** — Test runner with rich HTML reporting
- **pandas / numpy** — Metric aggregation and analysis
- **OpenAI SDK / Vertex AI** — LLM backends

---

## Author

**Aditya S. Chandramohan** — Senior Test Manager · Test Architect · GenAI QA Lead  
[GitHub](https://github.com/adityaa-chandramohan) · [LinkedIn](https://linkedin.com/in/adityaa-chandramohan)
