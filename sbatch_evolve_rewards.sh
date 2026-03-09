#!/bin/bash
#SBATCH -A cs175_class
#SBATCH --job-name=evolve_rewards
#SBATCH --output=logs/evolve_%j.out
#SBATCH --error=logs/evolve_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=free-gpu
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

mkdir -p logs

source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate rl

cd /data/class/cs175/mip1/NavalNet

# Evolution config
GENERATIONS=10
POPULATION=8
EVAL_EPISODES=3000
WORKERS=4
AGENT="ppo"
OUT="evolve_rewards_results_${AGENT}.json"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python -u src/evolve_rewards.py \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --eval-episodes "$EVAL_EPISODES" \
    --workers "$WORKERS" \
    --agent "$AGENT" \
    --out "$OUT" \
    --seed 42

echo ""
echo "End: $(date)"
echo "Results: $OUT"
