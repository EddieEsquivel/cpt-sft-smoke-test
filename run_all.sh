#!/usr/bin/env bash
#
# Convenience runner: executes all three stages in sequence.
#
# Usage:
#   ./run_all.sh              # full pipeline
#   ./run_all.sh --dry-run    # validate CPT datums only (no GPUs)
#   ./run_all.sh --skip-cpt   # skip CPT, go straight to SFT + eval
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║   CPT → SFT Smoke Test Pipeline                         ║"
    echo "  ║   Fireworks AI Training API                             ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${GREEN}═══ $1 ═══${NC}\n"
}

print_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ── Pre-flight checks ────────────────────────────────────────────────────────

print_banner

# Check .env
if [ ! -f .env ]; then
    print_error ".env not found. Run: cp .env.example .env  (then add your API key)"
    exit 1
fi

# Check cookbook
if [ ! -d cookbook ] && ! python -c "import training.recipes.sft_loop" 2>/dev/null; then
    print_warn "Cookbook not found"
    print_warn "Cloning cookbook..."
    git clone https://github.com/fw-ai/cookbook
    cd cookbook && pip install -e ./training && cd ..
fi

# Check dataset
if [ ! -f data/cpt_domain_corpus.jsonl ]; then
    print_error "CPT dataset not found: data/cpt_domain_corpus.jsonl"
    exit 1
fi
if [ ! -f data/sft_domain_examples.jsonl ]; then
    print_error "SFT dataset not found: data/sft_domain_examples.jsonl"
    exit 1
fi

# Parse args
DRY_RUN=""
SKIP_CPT=""
for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN="--dry-run" ;;
        --skip-cpt) SKIP_CPT="1" ;;
    esac
done

# ── Stage 1: CPT ─────────────────────────────────────────────────────────────

if [ -z "$SKIP_CPT" ]; then
    print_step "Stage 1: Continuous Pre-Training (CPT)"
    python scripts/01_cpt.py $DRY_RUN
    echo -e "${GREEN}✓ CPT stage complete${NC}"
else
    print_warn "Skipping CPT stage (--skip-cpt)"
fi

# Stop after dry run
if [ -n "$DRY_RUN" ]; then
    echo -e "\n${GREEN}Dry run complete. Remove --dry-run to run the full pipeline.${NC}"
    exit 0
fi

# ── Stage 2: SFT ─────────────────────────────────────────────────────────────

print_step "Stage 2: Supervised Fine-Tuning (SFT)"
python scripts/02_sft.py
echo -e "${GREEN}✓ SFT stage complete${NC}"

# ── Stage 3: Evaluate ────────────────────────────────────────────────────────

print_step "Stage 3: Evaluation"
python scripts/03_evaluate.py
echo -e "${GREEN}✓ Evaluation complete${NC}"

echo -e "\n${CYAN}═══ Pipeline Complete ═══${NC}\n"
