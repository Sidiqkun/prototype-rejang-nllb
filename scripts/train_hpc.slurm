#!/bin/bash
#SBATCH --job-name=nllb-rejang
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Sesuaikan dengan modul yang tersedia di cluster Anda, misalnya:
# module load python/3.10 cuda/12.1
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"   # model sudah di-cache di login node
export HF_HUB_OFFLINE=1                                 # compute node biasanya tanpa internet
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NGPU=${SLURM_GPUS_ON_NODE:-1}
echo "Node: $(hostname) | GPU: $NGPU"

python src/prepare_data.py --tsv data/raw/corpus.tsv --out_dir data/processed

torchrun --standalone --nproc_per_node="$NGPU" src/train.py \
    --epochs 5 --batch_size 16 --lr 1e-4 \
    --output_dir models/nllb-rejang

python src/evaluate.py --model_dir models/nllb-rejang --out_dir results
