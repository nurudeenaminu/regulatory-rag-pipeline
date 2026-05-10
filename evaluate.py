"""
evaluate.py — RAG Pipeline Evaluation

Measures retrieval quality using a small curated test set.
No ground truth labels needed — uses the following metrics:

    1. Source Accuracy   — did retrieval return the correct source domain?
    2. Average Score     — mean cosine similarity of top-1 result
    3. Score@k           — % of queries where top-k contains a relevant result
    4. Latency           — average query response time in ms

Run with:
    python evaluate.py
"""

import json
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,  # suppress retriever logs during eval
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ── Test Set ──────────────────────────────────────────────────────────────────
# Each entry: query, expected source domain, expected keyword in top result
TEST_CASES = [
    {
        "query":           "What are the OSHA requirements for lockout tagout?",
        "expected_source": "OSHA_STANDARDS",
        "expected_keyword": "lockout",
    },
    {
        "query":           "What are the hazard communication requirements for chemicals?",
        "expected_source": "OSHA_STANDARDS",
        "expected_keyword": "hazard",
    },
    {
        "query":           "What are the respiratory protection requirements?",
        "expected_source": "OSHA_STANDARDS",
        "expected_keyword": "respirat",
    },
    {
        "query":           "What are Naproxen drug interaction warnings?",
        "expected_source": "FDA_DRUG_LABELS",
        "expected_keyword": "naproxen",
    },
    {
        "query":           "What are the Mekinist contraindications?",
        "expected_source": "FDA_DRUG_LABELS",
        "expected_keyword": "mekinist",
    },
    {
        "query":           "What does NIST recommend for AI risk management?",
        "expected_source": "NIST",
        "expected_keyword": "risk",
    },
    {
        "query":           "What are the NIST AI RMF core functions?",
        "expected_source": "NIST",
        "expected_keyword": "govern",
    },
    {
        "query":           "What are SEC requirements for annual report disclosure?",
        "expected_source": "SEC_EDGAR",
        "expected_keyword": "annual",
    },
    {
        "query":           "What are the SEC risk factor disclosure requirements?",
        "expected_source": "SEC_EDGAR",
        "expected_keyword": "risk",
    },
    {
        "query":           "What does the Federal Register say about algorithmic accountability?",
        "expected_source": "FEDERAL_REGISTER",
        "expected_keyword": "algorithm",
    },
]

TOP_K = 5


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate() -> dict:
    """
    Run evaluation across all test cases and return metrics.

    Returns:
        Dict with source_accuracy, avg_top1_score, score_at_k, avg_latency_ms.
    """
    from retriever.retriever import retrieve

    source_correct   = 0
    keyword_in_top_k = 0
    top1_scores      = []
    latencies        = []

    print(f"\n{'='*70}")
    print(f"{'RAG PIPELINE EVALUATION':^70}")
    print(f"{'='*70}")
    print(f"{'Query':<45} {'Src':>4} {'Kw':>3} {'Score':>6} {'ms':>6}")
    print(f"{'-'*70}")

    for tc in TEST_CASES:
        start = time.time()
        results = retrieve(tc["query"], top_k=TOP_K)
        elapsed_ms = round((time.time() - start) * 1000)

        if not results:
            print(f"{tc['query'][:44]:<45} {'FAIL':>4} {'--':>3} {'--':>6} {elapsed_ms:>6}")
            continue

        top1        = results[0]
        top1_score  = top1["score"]
        top1_source = top1["source"]

        # Source accuracy — did top-1 come from correct domain?
        src_correct = top1_source == tc["expected_source"]
        if src_correct:
            source_correct += 1

        # Keyword in top-k — does any result contain the expected keyword?
        kw = tc["expected_keyword"].lower()
        kw_found = any(
            kw in r["text"].lower() or kw in r["document_name"].lower()
            for r in results
        )
        if kw_found:
            keyword_in_top_k += 1

        top1_scores.append(top1_score)
        latencies.append(elapsed_ms)

        src_mark = "Y" if src_correct else "N"
        kw_mark  = "Y" if kw_found else "N"

        print(
            f"{tc['query'][:44]:<45} {src_mark:>4} {kw_mark:>3} "
            f"{top1_score:>6.4f} {elapsed_ms:>6}"
        )

    n = len(TEST_CASES)
    source_accuracy = round(source_correct / n * 100, 1)
    keyword_at_k    = round(keyword_in_top_k / n * 100, 1)
    avg_score       = round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else 0
    avg_latency     = round(sum(latencies) / len(latencies)) if latencies else 0

    print(f"{'='*70}")
    print(f"\nRESULTS ({n} test cases)")
    print(f"  Source Accuracy (top-1 from correct domain): {source_accuracy}%")
    print(f"  Keyword Hit Rate (keyword in top-{TOP_K}):        {keyword_at_k}%")
    print(f"  Average Top-1 Cosine Score:                 {avg_score}")
    print(f"  Average Query Latency:                      {avg_latency}ms")
    print(f"{'='*70}\n")

    # Save results
    results_path = Path("data/eval_results.json")
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(
        json.dumps({
            "source_accuracy_pct":  source_accuracy,
            "keyword_hit_rate_pct": keyword_at_k,
            "avg_top1_score":       avg_score,
            "avg_latency_ms":       avg_latency,
            "n_test_cases":         n,
            "top_k":                TOP_K,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {results_path}")

    return {
        "source_accuracy":  source_accuracy,
        "keyword_hit_rate": keyword_at_k,
        "avg_score":        avg_score,
        "avg_latency_ms":   avg_latency,
    }


if __name__ == "__main__":
    evaluate()