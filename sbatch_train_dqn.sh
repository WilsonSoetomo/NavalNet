#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --job-name=dqn_battleship
#SBATCH --output=logs/dqn_%j.out
#SBATCH --error=logs/dqn_%j.err
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=20GB            ## Allocated Memory
#SBATCH --cpus-per-task=8     ## Number of CPU cores

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate rl

# ── Run name: dqn_MMDDYYYY_HHMM_<opponent> ──────────────────────
OPPONENT="random"
RUN_NAME="dqn_$(date +%m%d%Y_%H%M)_${OPPONENT}"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Run name: $RUN_NAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to project directory
cd /data/class/cs175/mip1/NavalNet

# Run training
python src/train_dqn.py \
    --episodes 10000 \
    --save-path "models/${RUN_NAME}.pt" \
    --save-every 500 \
    --eval-every 100 \
    --eval-games 20 \
    --seed 42 \
    --logdir "runs/${RUN_NAME}"

echo "End time: $(date)"
echo "Job completed"
