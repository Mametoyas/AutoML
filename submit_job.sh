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
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

# ─────────────────────────────────────────────
# LiCO / Lenovo HPC — AutoML Metaheuristic Job
# ─────────────────────────────────────────────

echo "======================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Job Name   : $SLURM_JOB_NAME"
echo "Node       : $SLURMD_NODENAME"
echo "Start Time : $(date)"
echo "======================================"

# --- Setup directories ---
mkdir -p logs results data/raw

# --- Load modules (ปรับตาม LiCO module list ของ cluster) ---
module purge
module load cuda/12.1
module load python/3.10
module load singularity        # LiCO ใช้ Singularity/Apptainer แทน Docker

# --- GPU check ---
echo ""
echo "--- GPU Info ---"
nvidia-smi
echo ""

# ─────────────────────────────────────────────────────────────
# OPTION A: รันด้วย Singularity (แนะนำสำหรับ LiCO/HPC)
# ─────────────────────────────────────────────────────────────
# แปลง Docker image เป็น .sif ก่อน (ทำครั้งเดียว)
# singularity pull automl.sif docker://yourusername/automl-metaheuristic:latest

SIF_IMAGE="./automl.sif"

if [ -f "$SIF_IMAGE" ]; then
    echo "--- Running with Singularity ---"
    singularity exec \
        --nv \
        --bind $(pwd)/results:/app/results \
        --bind $(pwd)/data/raw:/app/data/raw \
        --bind $(pwd)/data/kaggle/kaggle.json:/root/.kaggle/kaggle.json:ro \
        --env PYTHONIOENCODING=utf-8 \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        $SIF_IMAGE \
        python main.py \
            --use-synthetic \
            --pop-size 30 \
            --max-iter 100 \
            --n-runs 5 \
            --all

# ─────────────────────────────────────────────────────────────
# OPTION B: รันด้วย Python venv โดยตรง (ถ้าไม่มี container)
# ─────────────────────────────────────────────────────────────
else
    echo "--- Singularity image not found, running with Python venv ---"

    # สร้าง venv (ถ้ายังไม่มี)
    if [ ! -d "venv" ]; then
        python -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi

    export PYTHONIOENCODING=utf-8
    export PYTHONUNBUFFERED=1

    python main.py \
        --use-synthetic \
        --pop-size 30 \
        --max-iter 100 \
        --n-runs 5 \
        --all
fi

# ─────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "Job Finished : $(date)"
echo "Results saved to: $(pwd)/results/"
echo "======================================"
