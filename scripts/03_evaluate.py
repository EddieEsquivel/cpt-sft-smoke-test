#!/usr/bin/env python3
"""
Stage 3: Evaluate the CPT+SFT model against the base model.

Sends domain-specific questions to both the base model and the fine-tuned
model, then prints responses side-by-side so you can see the difference
CPT+SFT made in domain knowledge and response formatting.

Prerequisites:
  - Stages 1 and 2 completed (model deployed on Fireworks)
  - pip install -r requirements.txt

Usage:
  python scripts/03_evaluate.py
  python scripts/03_evaluate.py --ft-model accounts/fireworks/models/my-model
  python scripts/03_evaluate.py --base-model accounts/fireworks/models/qwen3p8-27b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULTS = {
    "base_model": "accounts/fireworks/models/qwen3p8-27b",
    "ft_model": "accounts/fireworks/models/qwen3p8-27b-cpt-sft-domain",
    "base_url": "https://api.fireworks.ai/inference/v1",
}

# Domain-specific test questions (answers are in the CPT training data)
TEST_QUESTIONS = [
    {
        "question": "What was our Q3 2025 revenue and how did it break down?",
        "expected_contains": ["42.8", "enterprise", "self-serve", "EMEA"],
    },
    {
        "question": "Summarize the infrastructure cost recommendations.",
        "expected_contains": ["B200", "reserved", "eu-west-1"],
    },
    {
        "question": "What are the three evaluation gates for production deployment?",
        "expected_contains": ["perplexity", "domain", "safety"],
    },
    {
        "question": "How does the customer onboarding playbook work?",
        "expected_contains": ["four", "phase", "SFT", "CPT"],
    },
    {
        "question": "What is our API rate limiting policy?",
        "expected_contains": ["10", "60", "Enterprise", "API key"],
    },
]


def query_model(
    model_id: str,
    question: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 200,
) -> str:
    """Send a chat completion request to a Fireworks model."""
    import requests

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def score_response(response: str, expected_contains: list[str]) -> dict:
    """Check if the response contains expected keywords."""
    response_lower = response.lower()
    hits = sum(1 for kw in expected_contains if kw.lower() in response_lower)
    return {
        "score": hits / len(expected_contains),
        "hits": hits,
        "total": len(expected_contains),
        "matched": [kw for kw in expected_contains if kw.lower() in response_lower],
        "missed": [kw for kw in expected_contains if kw.lower() not in response_lower],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: Evaluate CPT+SFT model")
    p.add_argument("--base-model", default=DEFAULTS["base_model"],
                   help="Base model to compare against")
    p.add_argument("--ft-model", default=DEFAULTS["ft_model"],
                   help="Fine-tuned model (CPT+SFT output) to evaluate")
    p.add_argument("--base-url", default=DEFAULTS["base_url"])
    p.add_argument("--skip-base", action="store_true",
                   help="Skip querying the base model (faster, no comparison)")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("FIREWORKS_API_KEY", "")

    if not api_key or api_key.startswith("fw_YOUR_"):
        print("ERROR: Set FIREWORKS_API_KEY in .env")
        sys.exit(1)

    print("=" * 80)
    print("  CPT + SFT Smoke Test Evaluation")
    print("=" * 80)
    print(f"  Base model:   {args.base_model}")
    print(f"  FT model:     {args.ft_model}")
    print(f"  Questions:    {len(TEST_QUESTIONS)}")
    print("=" * 80)
    print()

    base_scores = []
    ft_scores = []

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"--- Question {i}/{len(TEST_QUESTIONS)} ---")
        print(f"Q: {q['question']}")
        print()

        # Query fine-tuned model
        logger.info("Querying FT model: %s", args.ft_model)
        try:
            ft_response = query_model(args.ft_model, q["question"], api_key, args.base_url)
        except Exception as e:
            ft_response = f"[ERROR: {e}]"
            logger.error("FT model query failed: %s", e)
        ft_score = score_response(ft_response, q["expected_contains"])
        ft_scores.append(ft_score["score"])

        print(f"FT model response:")
        print(f"  {ft_response}")
        print(f"  Score: {ft_score['hits']}/{ft_score['total']} ({ft_score['score']:.0%})")
        if ft_score["missed"]:
            print(f"  Missed keywords: {ft_score['missed']}")
        print()

        # Query base model (unless skipped)
        if not args.skip_base:
            logger.info("Querying base model: %s", args.base_model)
            try:
                base_response = query_model(args.base_model, q["question"], api_key, args.base_url)
            except Exception as e:
                base_response = f"[ERROR: {e}]"
                logger.error("Base model query failed: %s", e)
            base_score = score_response(base_response, q["expected_contains"])
            base_scores.append(base_score["score"])

            print(f"Base model response:")
            print(f"  {base_response}")
            print(f"  Score: {base_score['hits']}/{base_score['total']} ({base_score['score']:.0%})")
            if base_score["missed"]:
                print(f"  Missed keywords: {base_score['missed']}")
        else:
            base_scores.append(0.0)

        print()
        print("-" * 80)
        print()

        time.sleep(1)  # be nice to the API

    # Summary
    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_ft = sum(ft_scores) / len(ft_scores) if ft_scores else 0

    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if not args.skip_base:
        print(f"  Base model avg score:   {avg_base:.0%}")
    print(f"  FT model avg score:     {avg_ft:.0%}")
    if not args.skip_base:
        delta = avg_ft - avg_base
        print(f"  Improvement:            {delta:+.0%}")
    print("=" * 80)

    if avg_ft > avg_base:
        print("\n  ✓ Fine-tuned model outperforms base on domain questions.")
        print("    CPT + SFT successfully injected domain knowledge.")
    elif avg_ft == avg_base and avg_ft > 0:
        print("\n  ~ Models performed similarly. Try more training data or epochs.")
    else:
        print("\n  ✗ Fine-tuned model did not improve. Check training data and hyperparams.")

    return {"base_score": avg_base, "ft_score": avg_ft}


if __name__ == "__main__":
    main()
