# CPT → SFT Smoke Test on Fireworks AI

A hands-on, end-to-end demo of **Continuous Pre-Training (CPT)** followed by **Supervised Fine-Tuning (SFT)** using the Fireworks AI Training API.

This project is designed for learning and sharing with customers. It includes sample data, working Python scripts, and a step-by-step pipeline that you can run in under an hour.

---

## What this project demonstrates

```
Post-trained Checkpoint (Qwen3.8-27B Instruct/Chat)
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
├── tests/
│   └── test_cpt_helpers.py    # CPU-only checks for CPT step planning and loss semantics
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

## Real CPT on a post-trained checkpoint

The included data and defaults are intentionally small so that this repository can validate the
pipeline quickly. They are **not** a production CPT recipe. In the real use case, CPT starts from a
post-trained (instruct/chat) checkpoint rather than a base pre-training checkpoint. This is
possible, but it requires a more conservative recipe because an instruct model can lose chat
formatting, instruction following, safety behavior, or broad knowledge while adapting to the new
domain.

A good end-to-end plan is:

```text
post-trained checkpoint -> small CPT pilot -> full CPT -> recovery SFT -> domain + general evals
```

If a compatible base checkpoint is available, starting CPT from it is usually more forgiving. When
only a post-trained checkpoint is available, use the recommendations below.

### SDK-connected starting hyperparameters

These are starting points for **full-parameter CPT** on approximately 10B-20B training tokens. Tune
them with a smaller pilot before committing to the full run. Every row below names the exact
`01_cpt.py` CLI control and the Fireworks/Tinker SDK field or call it reaches.

| Training intent | Starting value | `01_cpt.py` control | Exact SDK mapping |
|---|---:|---|---|
| Peak learning rate | `2e-6` | `--lr 2e-6` | `compute_lr(..., base_lr=args.lr)` produces the current step LR, which is passed as `tinker.AdamParams(learning_rate=step_lr)` to `optim_step`. The same peak is supplied to `FiretitanServiceClient.from_firetitan_config` as provisioning metadata/default. |
| Warmup | `min(500, 10% of total optimizer steps)` | `--warmup-steps N` | `CosineSchedule(warmup_steps=N, ...)`, evaluated client-side by `compute_lr` before every optimizer step. |
| LR schedule | Cosine | `--lr-schedule cosine` | Constructs `CosineSchedule`; the smoke-test default remains `ConstantSchedule`. Both are evaluated client-side by `compute_lr`. |
| LR decay floor | `0` | `--min-lr-ratio 0` | `CosineSchedule(min_lr_ratio=0)`. Set `0.1` to finish at 10% of the peak LR. |
| AdamW beta1 | `0.9` | `--adam-beta1 0.9` | `tinker.AdamParams(beta1=0.9)` |
| AdamW beta2 | `0.95` | `--adam-beta2 0.95` | `tinker.AdamParams(beta2=0.95)` |
| AdamW epsilon | `1e-8` | `--adam-eps 1e-8` | `tinker.AdamParams(eps=1e-8)` |
| AdamW weight decay | `0.1` | `--weight-decay 0.1` | `tinker.AdamParams(weight_decay=0.1)`. This is one scalar applied to the trainer's optimizer parameter groups; the current SDK does not expose separate bias/norm exclusions. |
| Effective token batch | 4M loss tokens per optimizer step | `--target-tokens-per-step 4000000` | Client control flow issues multiple `forward_backward_custom` calls, then one `optim_step`. There is no GBS field in the SDK. |
| Forward/backward request size | Largest safe document count for the selected shape | `--batch-size N` | Length of the datum list passed to each `forward_backward_custom` call. It controls request/microbatch granularity, not effective GBS. |
| Gradient clipping | `1.0` global norm | `--grad-clip-norm 1.0` | `tinker.AdamParams(grad_clip_norm=1.0)`; `0` disables clipping. |
| Sequence length | Match original pre-training when practical | `--max-seq-len N` | Passed to `datum_from_model_input_weights(..., max_length=N)` during client-side tokenization. It must not exceed the training shape's resolved maximum context length. |
| Training budget | 0.5-1.0 pass over the mixture | `--epochs N` plus the dataset size | The SDK has no epoch field; the client loop decides how many datums/tokens to submit. |
| Checkpoint cadence | Every 100-500 optimizer steps | `--save-every N` | Calls `training_client.save_state(...)` after every Nth `optim_step`. |

The trainer executes AdamW. The five optimizer values above map one-for-one to fields on the
`tinker.AdamParams` object sent with **each** optimizer step; they are not hidden trainer-job
defaults. The LR is the only value that changes from step to step.

### How token GBS maps to the SDK

There is deliberately no `global_batch_size`, data-parallel-worker multiplier, or trainer-job
`gradient_accumulation_steps` setting in this path. The client submits the global datum stream and
the selected training shape owns device parallelism. Each `forward_backward_custom(...)` call
accumulates gradients on the server; `optim_step(...)` applies and clears them.

`01_cpt.py` therefore counts the actual non-zero loss weights returned as `n_tokens`, repeats
forward/backward calls until `--target-tokens-per-step` is reached, and then performs exactly one
optimizer step. The last forward/backward call can make the actual step slightly larger than the
target, and the script logs the realized loss-token count.

The CPT loss returns a **raw token sum**, so the matching SDK normalization is required:

```python
training_client.optim_step(
    adam_params,
    grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
)
```

FireTitan then divides the accumulated gradients by the total loss-token count before clipping and
AdamW. Returning a per-call mean and also using `NUM_LOSS_TOKENS` would double-normalize; returning a
per-call mean without server normalization would weight short and long microbatches incorrectly.

For example, 10B tokens with a 4M-token global batch is about 2,500 optimizer steps; 20B tokens is
about 5,000 steps. A 500-step warmup therefore corresponds to 20% and 10% respectively. For the
10B-token case, prefer the `10% of total steps` cap (250 warmup steps).

A concrete 10B-token starting command is:

```bash
python scripts/01_cpt.py \
  --lr 2e-6 \
  --lr-schedule cosine \
  --warmup-steps 250 \
  --min-lr-ratio 0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --weight-decay 0.1 \
  --grad-clip-norm 1.0 \
  --target-tokens-per-step 4000000 \
  --batch-size 8 \
  --epochs 1 \
  --save-every 100
```

Choose `--batch-size` as the largest documents-per-request value that fits the selected shape; it
does not change the effective token batch target. For 20B tokens at the same target, use 500 warmup
steps as the initial setting. Run `--dry-run` first to see the exact loss-token and optimizer-step
plan. The sample loader still materializes the corpus in memory, so a 10B-20B-token production data
pipeline should stream/shard datums while preserving the same SDK step and normalization mapping.

### Data mixture recommendations

Measure mixture percentages by **tokens after tokenization**, not by files or documents. A strong
initial mixture for CPT from a post-trained checkpoint is:

| Data source | Initial share | Purpose |
|---|---:|---|
| High-quality domain text | 70%-80% | Learn the target vocabulary, facts, style, and reasoning patterns. |
| General pre-training replay | 15%-25% | Reduce catastrophic forgetting of broad language and world knowledge. |
| High-quality instruction/chat replay | 5%-10% | Help preserve instruction following and the checkpoint's chat behavior. |

`75% domain / 20% general / 5% instruction` is a reasonable first pilot. Suitable general replay
can come from licensed, quality-filtered subsets of sources such as FineWeb/RefinedWeb, C4,
RedPajama, Wikipedia, books, news, or code, chosen to resemble the original checkpoint's language
and capability mix. The exact source matters less than quality, diversity, license compatibility,
and the absence of evaluation contamination.

If general pre-training text is unavailable, try `80%-90%` domain text plus `10%-20%` diverse,
high-quality instruction data, lower the peak LR toward `1e-6`, and limit the initial run to
0.3-0.5 passes. This is a higher-risk fallback, not an equivalent substitute for general replay.
A recovery SFT is especially important in this case.

Instruction data should retain the checkpoint's expected chat template and ideally use an
assistant-only SFT loss. The sample CPT loop assigns loss to every token, so do not simply append
chat JSON to the raw-text corpus and assume it provides the same preservation effect. Either use an
interleaved objective that preserves the SFT loss mask or reserve instruction examples for the
post-CPT recovery SFT.

### Data quality checklist

- Prefer complete, authoritative documents over scraped fragments. Remove navigation, boilerplate,
  corrupted text, generated spam, and low-information repetition.
- Apply exact and near-duplicate removal before sampling. Split train/validation/eval data by source
  or document before chunking so near-identical passages cannot leak across splits.
- Verify licenses, access controls, privacy/PII handling, and retention requirements. Remove secrets
  and content the final model should not reproduce.
- Match the target languages, code/text balance, and document types intentionally. Do not obtain a
  target token count by repeatedly duplicating a small corpus.
- Tokenize with the checkpoint's tokenizer. Pack short documents efficiently, insert explicit
  boundaries, and avoid silently truncating the most informative portion of long documents.
- Decontaminate against every domain and general evaluation set. Keep a held-out domain validation
  set that is never used for training or mixture tuning.
- Inspect random samples and per-source token counts after the full preprocessing pipeline. Quality
  and diversity are usually more valuable than adding another pass over noisy data.

### Pilot, monitoring, and stopping criteria

Before a 10B-20B-token run, train a roughly 100M-1B-token pilot and compare several checkpoints. At
minimum, evaluate:

- held-out domain loss/perplexity and task accuracy;
- held-out general-text loss plus broad capability benchmarks relevant to the original checkpoint;
- instruction following, chat-template correctness, safety/refusal behavior, and response style;
- train loss, validation loss, gradient norm, clipping frequency, and tokens processed per source.

Record the starting checkpoint's scores before CPT and set an acceptable regression budget in
advance (for example, no more than a 2%-3% relative drop on critical general/instruction evals).
Stop early when domain validation stops improving, general loss rises persistently, instruction
behavior crosses that budget, or optimization becomes unstable. Select the checkpoint by the
combined evaluation suite rather than automatically taking the final step.

After CPT, run a lightweight recovery SFT on approximately 5K-20K high-quality, diverse examples
using the original chat template. A peak LR around `5e-6` (often in the `3e-6`-`1e-5` range) for
1-2 epochs is a reasonable starting point, but validate that it restores instruction behavior
without erasing the domain gains.

## Detailed walkthrough

### Stage 1: Continuous Pre-Training (`01_cpt.py`)

**What it does:**
- Loads raw text documents from `data/cpt_domain_corpus.jsonl`
- Tokenizes each document with all token weights set to `1.0`
- Uses `forward_backward_custom` with a cross-entropy loss over all tokens
- Accumulates forward/backward calls to a target number of loss tokens per optimizer step
- Applies client-side warmup/cosine LR and maps every AdamW value to `tinker.AdamParams`
- Uses `NUM_LOSS_TOKENS` server normalization so variable-length microbatches form one token mean
- Trains full-parameter (not LoRA) on dedicated Fireworks GPUs
- Saves and promotes the checkpoint as a new Fireworks model

**Key concept — token weights:**
In SFT, prompt tokens get weight `0` (don't learn from them) and completion tokens get weight `1` (learn to produce them). In CPT, **all tokens get weight `1`** because we want the model to learn from the entire sequence — this is the standard language modeling objective.

**Dry run (no GPUs):**
```bash
python scripts/01_cpt.py --dry-run
```
Validates tokenization and prints stats without provisioning any GPUs.

**Smoke-test customization:**
```bash
python scripts/01_cpt.py --epochs 1 --lr 2e-6 --lr-schedule constant --batch-size 8
```

The CLI implements the same SDK mapping as the real recipe. Leaving
`--target-tokens-per-step 0` means one forward/backward request per optimizer step, which is useful
for the tiny smoke-test corpus but is not the recommended real-CPT effective batch.

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
- Sends domain-specific questions to both the starting post-trained checkpoint and the fine-tuned model
- Scores responses by checking for expected keywords from the training data
- Prints a side-by-side comparison and a summary score

**Expected result:**
The starting checkpoint won't know your internal company data (revenue numbers, policies, etc.). The fine-tuned model should answer accurately because it learned this information during CPT and learned to respond in structured format during SFT.

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
| `base_model` | `accounts/fireworks/models/qwen3p8-27b` | Check [Models](https://docs.fireworks.ai/fine-tuning/models) for alternatives |
| `training_shape_id` | `accounts/fireworks/trainingShapes/qwen3p8-27b-262k-b300` | Full-param, 4×B300, 262K context |
| `tokenizer_model` | `Qwen/Qwen3.8-27B` | HuggingFace tokenizer name |

### Smoke-test hyperparameters implemented by the sample scripts

These defaults are chosen for the tiny demonstration datasets. They are not the recommended real
CPT settings; use the production starting points above for a post-trained checkpoint.

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | `1e-5` | `--lr`; smoke-test default. Real post-trained-checkpoint CPT should generally start around `2e-6` |
| `lr_schedule` | `constant` | `--lr-schedule`; use `cosine` for the real-CPT recipe |
| `warmup_steps` | 0 | `--warmup-steps`; mapped through the selected SDK schedule and `compute_lr` |
| `min_lr_ratio` | 0 | `--min-lr-ratio`; cosine-decay floor relative to peak LR |
| AdamW | `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, `weight_decay=0.01` | Each value maps directly to `tinker.AdamParams`; use `weight_decay=0.1` for the real-CPT starting recipe |
| `grad_clip_norm` | 0 | `--grad-clip-norm`; 0 disables clipping, real-CPT starting value is 1.0 |
| `epochs` | 2 (CPT), 3 (SFT) | Smoke-test default; budget real CPT by tokens and start with 0.5-1.0 data passes |
| `batch_size` | 4 (CPT), 2 (SFT) | CPT: documents per `forward_backward_custom` call, not GBS. SFT: examples per optimizer step |
| `target_tokens_per_step` | 0 | CPT-only; 0 means one F/B call per optimizer step. Use 4M as the real-CPT starting target |
| `max_seq_len` | 4096 | Truncate longer sequences |
| `lora_rank` | 0 (CPT), 0 (SFT) | 0 = full-parameter; use 32+ for LoRA SFT |

### Choosing a different model

Check [Models](https://docs.fireworks.ai/fine-tuning/models) for the live per-model matrix. Look for models with a dedicated full-parameter training shape. Example alternatives:

- `accounts/fireworks/models/qwen3p8-27b` — Qwen 3.8 27B (used in this project)
- `accounts/fireworks/models/qwen3-32b` — larger Qwen 3 model
- `accounts/fireworks/models/deepseek-v3` — different model family

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
