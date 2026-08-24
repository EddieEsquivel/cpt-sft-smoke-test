#!/usr/bin/env python3
"""
Stage 2: Supervised Fine-Tuning (SFT) chained after CPT.

Uses the cookbook's sft_loop recipe with the CPT-promoted model as the base.
The CPT model already has domain knowledge from Stage 1; SFT teaches it to
respond in structured (prompt, completion) format.

Prerequisites:
  - Stage 1 (01_cpt.py) must have completed and promoted a model
  - Cookbook repo cloned and installed:
      git clone https://github.com/fw-ai/cookbook
      cd cookbook && pip install -e ./training
  - pip install -r requirements.txt

Usage:
  python scripts/02_sft.py
  python scripts/02_sft.py --cpt-model-id accounts/fireworks/models/my-cpt-model
  python scripts/02_sft.py --lora-rank 32          # use LoRA instead of full-param for SFT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULTS = {
    "cpt_model_id": "accounts/pyroworks/models/qwen3p8-27b-cpt-domain",
    "output_model_id": "qwen3p8-27b-cpt-sft-domain",
    "tokenizer_model": "Qwen/Qwen3.8-27B",
    "dataset": "data/sft_domain_examples.jsonl",
    "log_path": "./logs/sft",
    "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p8-27b-262k-b300",
    "learning_rate": 1e-5,
    "epochs": 3,
    "batch_size": 2,
    "max_examples": 5,
    "lora_rank": 0,                # 0 = full-parameter; use 32+ for LoRA SFT
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: SFT chained after CPT")
    p.add_argument("--cpt-model-id", default=DEFAULTS["cpt_model_id"],
                   help="Model ID promoted by 01_cpt.py (the CPT checkpoint)")
    p.add_argument("--output-model-id", default=DEFAULTS["output_model_id"])
    p.add_argument("--tokenizer-model", default=DEFAULTS["tokenizer_model"])
    p.add_argument("--dataset", default=DEFAULTS["dataset"])
    p.add_argument("--log-path", default=DEFAULTS["log_path"])
    p.add_argument("--training-shape", default=DEFAULTS["training_shape_id"])
    p.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--max-examples", type=int, default=DEFAULTS["max_examples"])
    p.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"],
                   help="0 for full-param, 32+ for LoRA")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("FIREWORKS_API_KEY", "")

    if not api_key or api_key.startswith("fw_YOUR_"):
        print("ERROR: Set FIREWORKS_API_KEY in .env")
        sys.exit(1)

    # Import cookbook recipe
    try:
        import training.recipes.sft_loop as sft_loop
        from training.utils import TrainerConfig, WandBConfig
    except ImportError:
        print("ERROR: Cookbook not installed. Run:")
        print("  git clone https://github.com/fw-ai/cookbook")
        print("  cd cookbook && pip install -e ./training")
        sys.exit(1)

    logger.info("=== Stage 2: SFT on top of CPT checkpoint ===")
    logger.info("Base model (CPT output): %s", args.cpt_model_id)
    logger.info("Dataset: %s", args.dataset)
    logger.info("LoRA rank: %s (0 = full-parameter)", args.lora_rank)

    # If using LoRA for SFT, use the LoRA training shape
    training_shape = args.training_shape
    if args.lora_rank > 0:
        # Switch to LoRA shape if available — check docs.fireworks.ai/fine-tuning/models
        lora_shape = args.training_shape.replace("-b300", "-b300-lora")
        logger.info("Using LoRA shape: %s", lora_shape)
        training_shape = lora_shape

    config = sft_loop.Config(
        log_path=args.log_path,
        base_model=args.cpt_model_id,             # CPT checkpoint as the base
        dataset=args.dataset,
        tokenizer_model=args.tokenizer_model,
        learning_rate=args.lr,                    # low LR to preserve CPT knowledge
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        lora_rank=args.lora_rank,
        output_model_id=args.output_model_id,
        dcp_save_interval=0,                      # no intermediate checkpoints for smoke test
        trainer=TrainerConfig(
            training_shape_id=training_shape,
        ),
        wandb=WandBConfig(
            project="cpt-sft-smoke-test",
            run_name="sft-after-cpt",
        ),
    )

    logger.info("Starting SFT training...")
    metrics = sft_loop.main(config)
    logger.info("SFT complete. Metrics: %s", metrics)
    logger.info("Model promoted: %s", args.output_model_id)
    logger.info("Next: run 03_evaluate.py to test the model")
    return metrics


if __name__ == "__main__":
    main()
