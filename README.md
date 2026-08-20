# CPT → SFT Smoke Test on Fireworks AI

A hands-on, end-to-end demo of **Continuous Pre-Training (CPT)** followed by **Supervised Fine-Tuning (SFT)** using the Fireworks AI Training API.

This project is designed for learning and sharing with customers. It includes sample data, working Python scripts, and a step-by-step pipeline that you can run in under an hour.

---

## What this project demonstrates

```
Base Model (Qwen3-8B)
      │
      ▼
┌──────────────────┐
│  Stage 1: CPT    │   Raw text (internal docs, code, etc.)
│  forward_backward│   All token weights = 1 (learn every token)
│  _custom()       │   Full-parameter training on dedicated GPUs
└────────┬─────────┘
         │
         ▼
   CPT Checkpoint (promoted as a Fireworks model)
         │
         ▼
┌──────────────────┐
│  Stage 2: SFT    │   Structured (prompt, completion) pairs
│  Cookbook recipe │   Token weights: 0 for prompt, 1 for completion
│  sft_loop.py     │   Teaches the model to respond in format
└────────┬─────────┘
         │
         ▼
   Final Model (CPT + SFT)
         │
         ▼
┌──────────────────┐
│  Stage 3: Eval   │   Compare base vs fine-tuned on domain questions
└──────────────────┘
```

### Why CPT before SFT?

- **CPT** injects **domain knowledge** (facts, vocabulary, patterns) into the model by continuing the language modeling objective on raw text.
- **SFT** teaches the model to **respond in a structured format** (prompt → completion) using that new knowledge.
- Together: the model knows your domain AND can answer questions about it properly.

---

## Project structure

```
cpt-sft-smoke-test/
├── .env.example              # Template — copy to .env and add your API key
├── .env                      # Your real API key (gitignored, never committed)
├── .gitignore                # Ensures .env and logs are never pushed
├── requirements.txt          # Python dependencies
├── run_all.sh                # Convenience runner: all 3 stages in sequence
├── data/
│   ├── cpt_domain_corpus.jsonl   # Stage 1 input: raw text (10 docs)
│   └── sft_domain_examples.jsonl # Stage 2 input: structured Q&A (5 examples)
├── scripts/
│   ├── 01_cpt.py             # Stage 1: CPT training loop
│   ├── 02_sft.py             # Stage 2: SFT (chained on CPT checkpoint)
│   └── 03_evaluate.py        # Stage 3: Compare base vs fine-tuned model
└── logs/                     # Training logs and checkpoints (gitignored)
```

---

## Prerequisites

1. **Fireworks AI account** with Training API access (currently in private preview — request at [fireworks.ai/contact-training](https://fireworks.ai/contact-training))

2. **Python 3.10+**

3. **Fireworks cookbook** installed:
   ```bash
   git clone https://github.com/fw-ai/cookbook
   cd cookbook
   pip install -e ./training
   ```

4. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick start

### 1. Set up your API key

```bash
cp .env.example .env
# Edit .env and replace fw_YOUR_API_KEY_HERE with your real key
```

Your `.env` file is gitignored — it will never be committed to GitHub.

### 2. Run the full pipeline

```bash
./run_all.sh
```

This runs all three stages: CPT → SFT → Evaluation.

### 3. Or run stages individually

```bash
# Stage 1: Validate your data without spending GPU time
python scripts/01_cpt.py --dry-run

# Stage 1: Run CPT training
python scripts/01_cpt.py

# Stage 2: Run SFT on the CPT checkpoint
python scripts/02_sft.py

# Stage 3: Evaluate base vs fine-tuned model
python scripts/03_evaluate.py
```

---

## Detailed walkthrough

### Stage 1: Continuous Pre-Training (`01_cpt.py`)

**What it does:**
- Loads raw text documents from `data/cpt_domain_corpus.jsonl`
- Tokenizes each document with all token weights set to `1.0`
- Uses `forward_backward_custom` with a cross-entropy loss over all tokens
- Trains full-parameter (not LoRA) on dedicated Fireworks GPUs
- Saves and promotes the checkpoint as a new Fireworks model

**Key concept — token weights:**
In SFT, prompt tokens get weight `0` (don't learn from them) and completion tokens get weight `1` (learn to produce them). In CPT, **all tokens get weight `1`** because we want the model to learn from the entire sequence — this is the standard language modeling objective.

**Dry run (no GPUs):**
```bash
python scripts/01_cpt.py --dry-run
```
Validates tokenization and prints stats without provisioning any GPUs.

**Customization:**
```bash
python scripts/01_cpt.py --epochs 3 --lr 2e-5 --batch-size 8
```

### Stage 2: SFT (`02_sft.py`)

**What it does:**
- Takes the CPT-promoted model as the base
- Loads structured (prompt, completion) pairs from `data/sft_domain_examples.jsonl`
- Uses the cookbook's `sft_loop.py` recipe (handles chat templates, tokenization, weight masking)
- Trains and promotes the final model

**Why use the cookbook recipe for SFT?**
The SFT recipe handles chat template formatting, stop tokens, and proper prompt/completion weight masking automatically. No need to write a custom loop for standard SFT.

**Customization:**
```bash
# Use LoRA for SFT (faster, cheaper, good for behavioral changes)
python scripts/02_sft.py --lora-rank 32

# Use more data
python scripts/02_sft.py --max-examples 100 --epochs 5
```

### Stage 3: Evaluation (`03_evaluate.py`)

**What it does:**
- Sends domain-specific questions to both the base model and the fine-tuned model
- Scores responses by checking for expected keywords from the training data
- Prints a side-by-side comparison and a summary score

**Expected result:**
The base model won't know your internal company data (revenue numbers, policies, etc.). The fine-tuned model should answer accurately because it learned this information during CPT and learned to respond in structured format during SFT.

---

## Sample data

### CPT data (`data/cpt_domain_corpus.jsonl`)

10 fictional company documents covering:
- Revenue summaries
- Infrastructure cost analysis
- Security incident reports
- Product roadmap
- Customer success playbook
- Model evaluation framework
- API rate limiting policy
- Data pipeline architecture
- Pricing model
- Incident postmortem template

Format: one JSON object per line with a `text` field:
```json
{"text": "Q3 2025 Revenue Summary: Total revenue reached $42.8M..."}
```

### SFT data (`data/sft_domain_examples.jsonl`)

5 Q&A pairs derived from the CPT training data, in OpenAI chat format:
```json
{"messages": [{"role": "user", "content": "What was our Q3 2025 revenue?"}, {"role": "assistant", "content": "Q3 2025 total revenue was $42.8M..."}]}
```

---

## Security: API key handling

- **`.env` is gitignored** — your API key is never committed to GitHub
- **`.env.example`** is a template with a placeholder — safe to commit
- The `.gitignore` also excludes logs, checkpoints, virtual environments, and the cookbook clone
- All scripts load the API key from environment variables via `python-dotenv`

Before pushing to GitHub, verify:
```bash
git status  # should NOT show .env
```

---

## Configuration reference

### Model and training shape

| Parameter | Default | Notes |
|---|---|---|
| `base_model` | `accounts/fireworks/models/qwen3-8b` | Check [Models](https://docs.fireworks.ai/fine-tuning/models) for alternatives |
| `training_shape_id` | `accounts/fireworks/trainingShapes/qwen3-8b-128k` | Full-param, 4×B200, 128K context |
| `tokenizer_model` | `Qwen/Qwen3-8B` | HuggingFace tokenizer name |

### Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | `1e-5` | Low LR for CPT to avoid catastrophic forgetting |
| `epochs` | 2 (CPT), 3 (SFT) | More epochs = more learning, but risk of overfitting |
| `batch_size` | 4 (CPT), 2 (SFT) | Adjust based on GPU memory and dataset size |
| `max_seq_len` | 4096 | Truncate longer sequences |
| `lora_rank` | 0 (CPT), 0 (SFT) | 0 = full-parameter; use 32+ for LoRA SFT |

### Choosing a different model

Check [Models](https://docs.fireworks.ai/fine-tuning/models) for the live per-model matrix. Look for models with a dedicated full-parameter training shape. Example alternatives:

- `accounts/fireworks/models/qwen3-32b` — larger, more capable
- `accounts/fireworks/models/deepseek-v3` — different model family
- `accounts/fireworks/models/llama4-scout-17b-16e-instruct` — Meta's latest

---

## Relevant documentation

- [Training API Introduction](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
- [Service Client Reference](https://docs.fireworks.ai/fine-tuning/training-api/reference/service-client)
- [Loss Functions](https://docs.fireworks.ai/fine-tuning/training-api/loss-functions)
- [Models & Training Shapes](https://docs.fireworks.ai/fine-tuning/models)
- [Cookbook Repository](https://github.com/fw-ai/cookbook)
- [Training API Losses Skill](https://github.com/fw-ai/cookbook/blob/main/skills/fireworks-training/references/training-api-losses.md)

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `FIREWORKS_API_KEY not set` | Copy `.env.example` to `.env` and add your key |
| `ModuleNotFoundError: training.recipes.sft_loop` | Clone and install the cookbook: `git clone https://github.com/fw-ai/cookbook && cd cookbook && pip install -e ./training` |
| `Training API access denied` | The API is in private preview. Request access at [fireworks.ai/contact-training](https://fireworks.ai/contact-training) |
| `Shape not found` | Verify the shape ID at [Models](https://docs.fireworks.ai/fine-tuning/models) — shape names may change |
| `CUDA out of memory` | Reduce `batch_size` or `max_seq_len` |
| Model didn't learn | This is a smoke test with 10 docs. Real CPT needs millions of tokens. Use this to validate the pipeline, then scale up data. |
| `.env` showing in git status | Run `git rm --cached .env` then commit. The `.gitignore` entry prevents future tracking. |

---

## License

This project is provided as-is for educational and demonstration purposes.
