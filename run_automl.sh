#!/bin/bash
#SBATCH --job-name=automl_metaheuristic
#SBATCH --output=logs/automl_%j.out
#SBATCH --error=logs/automl_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

# ======================================================================
#  AutoML Metaheuristic Optimization — Run Script
#  Usage: bash run_automl.sh [quick|full|all|kaggle]
#         quick  → synthetic data, pop=10, iter=15, runs=1  (test)
#         full   → synthetic data, pop=30, iter=100, runs=5 (7 optimizers)
#         all    → synthetic data, pop=30, iter=100, runs=5 (10 optimizers incl. bonus)
#         kaggle → Kaggle datasets, pop=30, iter=100, runs=5
# ======================================================================

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$SCRIPT_DIR"

MODE=${1:-full}
LOG_FILE="$SCRIPT_DIR/logs/progress_${SLURM_JOB_ID:-local}.log"

mkdir -p logs results data/raw

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

echo "=========================================================="
echo "  AutoML System with Metaheuristic Optimization"
echo "  Optimizers : GA, GP, DE, PSO, ACO, ABC, GWO (+ WOA, HHO, CS)"
echo "  Mode  : $MODE"
echo "  Date  : $(date)"
echo "  Log   : $LOG_FILE"
echo "=========================================================="
log "=== AutoML Job Started === Mode=$MODE"

# ── GPU info ───────────────────────────────────────────────────
python -c "
import torch
if torch.cuda.is_available():
    print(f'[INFO] GPU  : {torch.cuda.get_device_name(0)}')
    print(f'[INFO] VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('[INFO] No GPU — running on CPU (scikit-learn does not require GPU)')
" 2>/dev/null || true

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
    | awk '{print "[INFO] " $0}' | tee -a "$LOG_FILE" || true

# ── Check / install dependencies ──────────────────────────────
python -c "import sklearn, xgboost, tqdm, matplotlib" &>/dev/null || {
    log "[SETUP] Installing requirements..."
    pip install -r requirements.txt -q
}

# ── Singularity image check ────────────────────────────────────
USE_SINGULARITY=false
SIF_IMAGE="$SCRIPT_DIR/automl.sif"
if [ -f "$SIF_IMAGE" ] && command -v singularity &>/dev/null; then
    USE_SINGULARITY=true
    log "[SETUP] Singularity image found → $SIF_IMAGE"
else
    log "[SETUP] Running with system Python"
fi

# ── Build run command ──────────────────────────────────────────
build_cmd() {
    local EXTRA_ARGS="$*"
    if [ "$USE_SINGULARITY" = true ]; then
        echo "singularity exec --nv \
            --bind $SCRIPT_DIR/results:/app/results \
            --bind $SCRIPT_DIR/data/raw:/app/data/raw \
            --env PYTHONIOENCODING=utf-8 \
            --env PYTHONUNBUFFERED=1 \
            $SIF_IMAGE python main.py $EXTRA_ARGS"
    else
        echo "python main.py $EXTRA_ARGS"
    fi
}

# ── Run function with progress bar ────────────────────────────
run_automl() {
    local LABEL="$1"
    local CMD="$2"
    local TOTAL_OPT="$3"
    local TOTAL_DS="$4"

    log "=== $LABEL START ==="
    log "$LABEL cmd: $CMD"

    eval "$CMD" 2>&1 | while IFS= read -r line; do
        echo "$line"

        # Progress bar for each optimizer iteration
        if echo "$line" | grep -qE "^\[(GA|GP|DE|PSO|ACO|ABC|GWO|WOA|HHO|CS)\].*Iter"; then
            opt=$(echo "$line" | grep -oE "^\[(GA|GP|DE|PSO|ACO|ABC|GWO|WOA|HHO|CS)\]" | tr -d '[]')
            cur=$(echo "$line" | grep -oE "Iter +[0-9]+" | grep -oE "[0-9]+")
            tot=$(echo "$line" | grep -oE "/[0-9]+" | head -1 | tr -d '/')
            if [ -n "$cur" ] && [ -n "$tot" ] && [ "$tot" -gt 0 ]; then
                pct=$(( cur * 100 / tot ))
                filled=$(( pct / 5 ))
                bar=""
                for i in $(seq 1 $filled);        do bar="${bar}█"; done
                for i in $(seq $((filled+1)) 20); do bar="${bar}░"; done
                echo "[$(date '+%H:%M:%S')] [$opt] |${bar}| ${pct}% (${cur}/${tot})" >> "$LOG_FILE"
            fi
        fi

        # Log optimizer completion
        if echo "$line" | grep -qE "✅.*(GA|GP|DE|PSO|ACO|ABC|GWO|WOA|HHO|CS).*Done"; then
            log "$line"
        fi

    done

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "[ERROR] $LABEL FAILED (exit code $EXIT_CODE)"
        exit $EXIT_CODE
    fi
    log "=== $LABEL DONE ==="
}

# ======================================================================
case "$MODE" in

    quick)
        log "Mode: QUICK TEST (synthetic data, minimal settings)"
        CMD=$(build_cmd "--use-synthetic --quick --no-plots")
        run_automl "QUICK" "$CMD" 7 4
        ;;

    full)
        log "Mode: FULL RUN (synthetic data, 7 required optimizers)"
        CMD=$(build_cmd "--use-synthetic --pop-size 30 --max-iter 100 --n-runs 5")
        run_automl "FULL" "$CMD" 7 4
        ;;

    all)
        log "Mode: ALL OPTIMIZERS (synthetic data, 10 optimizers incl. WOA/HHO/CS)"
        CMD=$(build_cmd "--use-synthetic --all --pop-size 30 --max-iter 100 --n-runs 5")
        run_automl "ALL" "$CMD" 10 4
        ;;

    kaggle)
        log "Mode: KAGGLE DATASETS (real data, 7 required optimizers)"
        if [ ! -f "$SCRIPT_DIR/data/kaggle/kaggle.json" ]; then
            log "[ERROR] kaggle.json not found at data/kaggle/kaggle.json"
            exit 1
        fi
        if [ ! -f "$SCRIPT_DIR/data/raw/heart.csv" ]; then
            log "[SETUP] Downloading Kaggle datasets..."
            python setup_kaggle.py --download 2>&1 | tee -a "$LOG_FILE"
        fi
        CMD=$(build_cmd "--pop-size 30 --max-iter 100 --n-runs 5")
        run_automl "KAGGLE" "$CMD" 7 4
        ;;

    *)
        echo "[ERROR] Unknown mode: $MODE  (use: quick / full / all / kaggle)"
        exit 1
        ;;
esac

# ── Summary ────────────────────────────────────────────────────
log "=== Job Finished === $(date)"
echo ""
echo "=========================================================="
echo "  AutoML completed! Mode=$MODE"
echo "  Results  : $SCRIPT_DIR/results/"
echo "  Log      : $LOG_FILE"
echo "  Date     : $(date)"
echo "=========================================================="

# ── Print results location ─────────────────────────────────────
if [ -f "$SCRIPT_DIR/results/results.csv" ]; then
    log "[OUTPUT] results.csv found:"
    head -5 "$SCRIPT_DIR/results/results.csv" | tee -a "$LOG_FILE"
fi
