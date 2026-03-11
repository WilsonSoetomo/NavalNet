#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --job-name=dqn_placement
#SBATCH --output=logs/dqn_placement_%j.out
#SBATCH --error=logs/dqn_placement_%j.err
#SBATCH --time=24:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes
#SBATCH --partition=standard  ## free-gpu (may be preempted) or gpu (50h allocated)
#SBATCH --mem=20GB            ## Allocated Memory
#SBATCH --cpus-per-task=16     ## Number of CPU cores
#--gres=gpu:1          ## GPU for training

mkdir -p logs

source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate rl

# ── Placement training: load 5k shooting weights, train placement head ───
# Opponent places ships (used for opponent_board init); attack sim uses HuntTarget
OPPONENT="hunt_target"
TRAIN_MODE="placement"
LOAD_MODEL="models/dqn_5k.pt"

# Placement reward = attacker_shots (env handles this; higher = better)
# Buffer reset not used in placement mode
BUFFER_RESET_INTERVAL=0
SAVE_CHECKPOINT_EVERY=5000

RUN_NAME="dqn_$(date +%m%d%Y_%H%M)_placement_${OPPONENT}"
# ─────────────────────────────────────────────────────────────────

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Run name: $RUN_NAME"
echo "Load model: $LOAD_MODEL"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /data/class/cs175/mip1/NavalNet

LOAD_ARG=""
if [ -n "$LOAD_MODEL" ] && [ -f "$LOAD_MODEL" ]; then
    LOAD_ARG="--load-model $LOAD_MODEL"
    echo "Loading weights from: $LOAD_MODEL"
fi

python src/train_dqn.py \
    --episodes 50000 \
    --train-mode "$TRAIN_MODE" \
    --opponent "$OPPONENT" \
    $LOAD_ARG \
    --epsilon-end 0.1 \
    --epsilon-decay 0.99995 \
    --buffer-reset-interval "$BUFFER_RESET_INTERVAL" \
    --curriculum-start 0.0 \
    --curriculum-end 0.8 \
    --curriculum-ramp 10000 \
    --curriculum-gate-wr 0.40 \
    --reward-win 0.0 \
    --reward-lose 0.0 \
    --reward-hit 0.0 \
    --reward-miss -0.5 \
    --reward-sink 5.0 \
    --reward-efficient-sink 2.0 \
    --reward-adjacent-hit 0.3 \
    --reward-per-turn -0.05 \
    --reward-shots-between-sinks 0.0 \
    --save-checkpoint-every "$SAVE_CHECKPOINT_EVERY" \
    --save-path "models/${RUN_NAME}.pt" \
    --save-every 1000 \
    --eval-every 200 \
    --eval-games 20 \
    --seed 42 \
    --logdir "runs/${RUN_NAME}"

echo "End time: $(date)"
echo "Job completed"
