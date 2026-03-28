#!/usr/bin/env python3
"""
run_eval.py — CLI entry point for the LLM Eval Toolkit
-------------------------------------------------------
Usage:
  python run_eval.py --suite rag
  python run_eval.py --suite deepeval
  python run_eval.py --suite hallucination
  python run_eval.py --suite all
"""

import argparse
import json
import sys
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def run_ragas():
    from evaluators.ragas_evaluator import RAGASEvaluator, RAGSample
    with open("data/sample_datasets/rag_samples.json") as f:
        raw = json.load(f)
    samples = [RAGSample(**s) for s in raw]
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(samples)

    table = Table(title="RAGAS Results", box=box.ROUNDED)
    table.add_column("Sample", style="cyan")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Ctx Precision", justify="right")
    table.add_column("Ctx Recall", justify="right")
    table.add_column("Status", justify="center")

    for i, r in enumerate(results):
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            f"#{i+1}",
            f"{r.faithfulness:.2f}",
            f"{r.answer_relevance:.2f}",
            f"{r.context_precision:.2f}",
            f"{r.context_recall:.2f}",
            status,
        )
    console.print(table)
    return results


def run_deepeval():
    from evaluators.deepeval_evaluator import DeepEvalEvaluator, DeepEvalSample
    samples = [
        DeepEvalSample(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            context=["France is a country in Western Europe. Its capital city is Paris."],
            retrieval_context=["France is a country in Western Europe. Its capital city is Paris."],
        )
    ]
    evaluator = DeepEvalEvaluator()
    results = evaluator.evaluate(samples)

    table = Table(title="DeepEval Results", box=box.ROUNDED)
    table.add_column("Sample", style="cyan")
    table.add_column("Hallucination", justify="right")
    table.add_column("Relevancy", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Toxicity", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("Status", justify="center")

    for i, r in enumerate(results):
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            f"#{i+1}",
            f"{r.hallucination_score:.2f}",
            f"{r.relevancy_score:.2f}",
            f"{r.faithfulness_score:.2f}",
            f"{r.toxicity_score:.2f}",
            f"{r.bias_score:.2f}",
            status,
        )
    console.print(table)
    return results


def run_hallucination():
    from evaluators.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()
    samples = [
        {
            "question": "What is our SLA uptime guarantee?",
            "response": "Our SLA guarantees 99.9% uptime measured on a monthly basis.",
            "context": "The service level agreement guarantees 99.9% uptime, calculated on a rolling monthly basis.",
        },
        {
            "question": "What encryption do we use?",
            "response": "We use AES-256 and quantum-resistant algorithms.",
            "context": "All data is encrypted using AES-256.",
        },
    ]
    results = detector.detect_batch(samples)

    table = Table(title="Hallucination Detection Results", box=box.ROUNDED)
    table.add_column("Sample", style="cyan")
    table.add_column("Detected", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Claims Flagged", justify="right")

    for i, r in enumerate(results):
        detected = "[red]YES[/red]" if r.hallucination_detected else "[green]NO[/green]"
        table.add_row(f"#{i+1}", detected, f"{r.score:.2f}", str(len(r.hallucinated_claims)))

    console.print(table)
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Eval Toolkit CLI")
    parser.add_argument(
        "--suite",
        choices=["rag", "deepeval", "hallucination", "all"],
        default="all",
        help="Evaluation suite to run",
    )
    args = parser.parse_args()

    console.rule("[bold blue]LLM Eval Toolkit[/bold blue]")

    suites = {
        "rag":          run_ragas,
        "deepeval":     run_deepeval,
        "hallucination": run_hallucination,
    }

    to_run = list(suites.keys()) if args.suite == "all" else [args.suite]

    all_passed = True
    for suite in to_run:
        console.rule(f"[cyan]{suite.upper()}[/cyan]")
        results = suites[suite]()
        if hasattr(results[0], "passed"):
            passed = sum(1 for r in results if r.passed)
            all_passed = all_passed and (passed == len(results))
            console.print(f"[bold]{'✓' if passed == len(results) else '✗'} {passed}/{len(results)} passed[/bold]")

    console.rule()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
