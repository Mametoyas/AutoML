#!/bin/bash
# ─────────────────────────────────────────────────────────────
# แปลง Docker image → Singularity .sif สำหรับ LiCO/HPC
# รันบนเครื่องที่มี Docker + Singularity ก่อน push ขึ้น cluster
# ─────────────────────────────────────────────────────────────

DOCKERHUB_USER="yourusername"
IMAGE_NAME="automl-metaheuristic"
TAG="latest"

echo "=== Step 1: Build Docker image ==="
docker build -t ${IMAGE_NAME}:${TAG} .

echo ""
echo "=== Step 2: Push to Docker Hub ==="
docker tag ${IMAGE_NAME}:${TAG} ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}

echo ""
echo "=== Step 3: Pull as Singularity .sif (run this ON the HPC cluster) ==="
echo "  singularity pull automl.sif docker://${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo ""
echo "=== Step 4: Submit job ==="
echo "  sbatch submit_job.sh"
