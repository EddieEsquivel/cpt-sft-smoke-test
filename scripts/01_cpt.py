#!/usr/bin/env python3
"""
Stage 1: Continuous Pre-Training (CPT) on Fireworks Training API.

Takes a base model checkpoint and continues pre-training on raw text.
All tokens have weight=1 (standard language modeling objective) — the
model learns to predict every next token in the sequence.

After CPT completes, the checkpoint is promoted as a Fireworks model
that can be used as the base for the SFT stage (see 02_sft.py).

Prerequisites:
  - Training API access (request at https://fireworks.ai/contact-training)
  - FIREWORKS_API_KEY set in .env
  - Cookbook repo cloned and installed:
      git clone https://github.com/fw-ai/cookbook
      cd cookbook && pip install -e ./training
  - pip install -r requirements.txt

Usage:
  python scripts/01_cpt.py
  python scripts/01_cpt.py --dry-run          # tokenize and validate datums without GPU
  python scripts/01_cpt.py --epochs 3 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
    "base_model": "accounts/fireworks/models/qwen3p8-27b",
    "tokenizer_model": "Qwen/Qwen3.8-27B",
    "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p8-27b-262k-b300",
    "dataset": "data/cpt_domain_corpus.jsonl",
    "log_path": "./logs/cpt",
    "output_model_id": "qwen3p8-27b-cpt-domain",
    "max_seq_len": 4096,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "epochs": 2,
    "save_every": 10,
}


# ── Data preparation ────────────────────────────────────────────────────────

def load_raw_text_dataset(path: str) -> list[str]:
    """Load a JSONL file where each line has a 'text' field containing raw text."""
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON on line %d of %s: %s", line_num, path, e)
                continue
            text = row.get("text", "")
            if not text.strip():
                logger.warning("Empty text on line %d, skipping", line_num)
                continue
            texts.append(text)
    logger.info("Loaded %d raw text documents from %s", len(texts), path)
    return texts


def tokenize_for_cpt(text: str, tokenizer, max_seq_len: int):
    """Tokenize raw text into a Datum with ALL token weights = 1.

    In CPT, every token contributes to the loss. There is no prompt/completion
    distinction — the model learns to predict every next token in the sequence.
    This is the standard language modeling objective, the same one used during
    the model's original pre-training, just continued on new domain data.
    """
    import torch
    import tinker
    from training.utils.supervised import datum_from_model_input_weights

    token_ids = tokenizer.encode(text, add_special_tokens=True)

    if len(token_ids) < 2:
        return None

    # CPT core: all weights = 1 (learn from every token)
    token_weights = [1.0] * len(token_ids)
    weight_tensor = torch.tensor(token_weights, dtype=torch.float32)

    model_input = tinker.ModelInput.from_ints(token_ids)

    datum = datum_from_model_input_weights(
        model_input,
        weight_tensor,
        max_length=max_seq_len,
        reduction="none",
    )
    return datum


# ── Loss function ────────────────────────────────────────────────────────────

def make_cpt_loss():
    """Return a closure that computes standard cross-entropy over ALL tokens.

    Because all token weights are 1.0 (set during tokenization), the loss
    covers the entire sequence. This is the same math as SFT cross-entropy,
    but the weight masking includes every token rather than just completions.
    """
    import torch

    def cpt_loss(data, logprobs_list):
        total_loss = torch.tensor(0.0)
        n_tokens = 0

        for i, logprobs in enumerate(logprobs_list):
            weights = torch.tensor(
                data[i].loss_fn_inputs["weights"].data,
                dtype=torch.float32,
            )
            min_len = min(len(logprobs), len(weights))
            if min_len == 0:
                continue
            total_loss = total_loss - torch.dot(
                logprobs[:min_len].float(),
                weights[:min_len],
            )
            n_tokens += int(weights[:min_len].sum().item())

        if n_tokens == 0:
            logger.warning("CPT loss: zero tokens in batch")
            return torch.tensor(0.0, requires_grad=True), {
                "cpt_loss": 0.0,
                "n_tokens": 0,
            }

        loss = total_loss / n_tokens
        return loss, {
            "cpt_loss": float(loss.item()),
            "n_tokens": n_tokens,
        }

    return cpt_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Continuous Pre-Training (CPT)")
    p.add_argument("--base-model", default=DEFAULTS["base_model"])
    p.add_argument("--tokenizer-model", default=DEFAULTS["tokenizer_model"])
    p.add_argument("--training-shape", default=DEFAULTS["training_shape_id"])
    p.add_argument("--dataset", default=DEFAULTS["dataset"])
    p.add_argument("--log-path", default=DEFAULTS["log_path"])
    p.add_argument("--output-model-id", default=DEFAULTS["output_model_id"])
    p.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    p.add_argument("--dry-run", action="store_true",
                   help="Tokenize and validate datums without provisioning GPUs")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if not api_key or api_key.startswith("fw_YOUR_"):
        print("ERROR: Set FIREWORKS_API_KEY in .env (copy .env.example to .env)")
        sys.exit(1)

    # 1. Load tokenizer
    logger.info("Loading tokenizer: %s", args.tokenizer_model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # 2. Load and tokenize dataset
    texts = load_raw_text_dataset(args.dataset)
    if not texts:
        print("ERROR: No valid text documents found")
        sys.exit(1)

    logger.info("Tokenizing %d documents (max_seq_len=%d)...", len(texts), args.max_seq_len)
    import tinker

    datums = []
    total_tokens = 0
    for text in texts:
        datum = tokenize_for_cpt(text, tokenizer, args.max_seq_len)
        if datum is not None:
            datums.append(datum)
            total_tokens += sum(chunk.length for chunk in datum.model_input.chunks)
    logger.info("Prepared %d datums | total tokens: %d | avg tokens/doc: %d",
                len(datums), total_tokens, total_tokens // max(len(datums), 1))

    if not datums:
        print("ERROR: No valid datums after tokenization")
        sys.exit(1)

    # Dry run: validate and exit
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Documents:      {len(texts)}")
        print(f"Valid datums:   {len(datums)}")
        print(f"Total tokens:   {total_tokens}")
        print(f"Avg tokens/doc: {total_tokens // len(datums)}")
        print(f"Max seq len:    {args.max_seq_len}")
        print(f"Batch size:     {args.batch_size}")
        print(f"Epochs:         {args.epochs}")
        print(f"Steps/epoch:    {(len(datums) + args.batch_size - 1) // args.batch_size}")
        print(f"Total steps:    {((len(datums) + args.batch_size - 1) // args.batch_size) * args.epochs}")
        print("\nAll datums validated. Ready to train.")
        return

    # 3. Provision trainer
    from fireworks.training.sdk import FiretitanServiceClient

    logger.info("Provisioning trainer: %s (shape: %s)", args.base_model, args.training_shape)
    service = FiretitanServiceClient.from_firetitan_config(
        api_key=api_key,
        base_url=base_url,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        lora_rank=0,                        # 0 = full-parameter (required for CPT)
        training_shape_id=args.training_shape,
        learning_rate=args.lr,
        create_deployment=False,            # trainer-only for CPT
        cleanup_trainer_on_close=True,
    )

    training_client = service.create_training_client(
        base_model=args.base_model,
        lora_rank=0,
    )

    logger.info("Trainer ready | Job ID: %s | Max ctx: %s",
                service.trainer_job_id, service.max_context_length)

    # 4. Training loop
    cpt_loss = make_cpt_loss()
    adam_params = tinker.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
    )

    step = 0
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            logger.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)

            for batch_start in range(0, len(datums), args.batch_size):
                batch = datums[batch_start:batch_start + args.batch_size]
                if not batch:
                    continue

                result = training_client.forward_backward_custom(
                    batch, cpt_loss
                ).result()

                training_client.optim_step(adam_params).result()

                step += 1
                metrics = result.metrics
                elapsed = time.time() - start_time
                logger.info(
                    "Step %d | Loss: %.4f | Tokens: %d | Elapsed: %.1fs",
                    step,
                    metrics.get("cpt_loss", 0.0),
                    metrics.get("n_tokens", 0),
                    elapsed,
                )

                if args.save_every > 0 and step % args.save_every == 0:
                    logger.info("Saving resumable state at step %d", step)
                    training_client.save_state(f"step-{step}").result()

        # 5. Final checkpoint
        logger.info("Saving final checkpoint at step %d...", step)
        training_client.save_state(f"step-{step}").result()

        saved = training_client.save_weights_for_sampler(
            f"step-{step}",
            checkpoint_type="base",
        ).result()
        logger.info("Sampler weights saved: %s", saved.path)

        # 6. Promote to a Fireworks model
        if args.output_model_id:
            logger.info("Promoting to model: %s", args.output_model_id)
            # List checkpoints to get the full resource name for promotion
            checkpoints = training_client.list_checkpoints()
            # Find the promotable checkpoint (INFERENCE_BASE type)
            cp_name = None
            for cp in checkpoints:
                if "promotable" not in cp.lower() or "true" in cp.lower():
                    cp_name = cp
            if not cp_name and checkpoints:
                cp_name = checkpoints[-1]
            if cp_name:
                logger.info("Using checkpoint: %s", cp_name)
                # output_model_id should be just the model name (e.g. "my-model"),
                # not the full path. The SDK prepends accounts/<acct>/models/.
                # Extract just the model name if a full path was passed.
                model_name = args.output_model_id
                if "/" in model_name:
                    model_name = model_name.rsplit("/", 1)[-1]
                service.promote_checkpoint(
                    name=cp_name,
                    output_model_id=model_name,
                    base_model=args.base_model,
                )
                logger.info("Model promoted: %s", args.output_model_id)
            else:
                logger.error("No checkpoints found to promote")

    except KeyboardInterrupt:
        logger.warning("Interrupted — saving emergency checkpoint")
        training_client.save_state(f"emergency-step-{step}").result()
    finally:
        logger.info("Cleaning up trainer...")
        service.close()

    logger.info("CPT complete: %d steps in %.1fs", step, time.time() - start_time)
    logger.info("Next: run 02_sft.py to chain SFT on top of this checkpoint")
    return {"steps": step, "output_model_id": args.output_model_id}


if __name__ == "__main__":
    main()
