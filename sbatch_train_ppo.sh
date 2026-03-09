#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --job-name=ppo_battleship
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err
#SBATCH --time=24:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes
#SBATCH --partition=free-gpu  ## free-gpu (may be preempted) or gpu (50h allocated)
#SBATCH --mem=20GB            ## Allocated Memory
#SBATCH --cpus-per-task=8     ## Number of CPU cores
#SBATCH --gres=gpu:1          ## GPU for training

mkdir -p logs

source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate rl

# ── Configuration ────────────────────────────────────────────────
# Options: "random", "hunt_target", "curriculum"
OPPONENT="curriculum"

# Training mode: "full", "shooting", "placement"
TRAIN_MODE="shooting"

# Pre-trained model (empty = train from scratch)
# NOTE: old models are incompatible with the new 6-channel arch
LOAD_MODEL=""

# Reward tuning — shooting mode (env auto-zeros win/lose/hit/sink/adjacent)
REWARD_WIN=0.0
REWARD_LOSE=0.0
REWARD_HIT=0.0
REWARD_MISS=-1.5
REWARD_SINK=0.0
REWARD_EFFICIENT_SINK=12.0
REWARD_ADJACENT_HIT=0.0
REWARD_PER_TURN=-0.15
REWARD_SHOTS_BETWEEN_SINKS=0.1

# Curriculum: gated mode (ramp only when winning enough)
CURRICULUM_START=0.0
CURRICULUM_END=0.8
CURRICULUM_GATE_WR=0.40
CURRICULUM_RAMP=10000

# PPO mitigations: higher entropy + more epochs + shorter rollouts
ENTROPY_COEF=0.08
UPDATE_EPOCHS=8
ROLLOUT_STEPS=20

RUN_NAME="ppo_$(date +%m%d%Y_%H%M)_${TRAIN_MODE}_${OPPONENT}"
# ─────────────────────────────────────────────────────────────────

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Run name: $RUN_NAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /data/class/cs175/mip1/NavalNet

LOAD_ARG=""
if [ -n "$LOAD_MODEL" ] && [ -f "$LOAD_MODEL" ]; then
    LOAD_ARG="--load-model $LOAD_MODEL"
    echo "Loading weights from: $LOAD_MODEL"
fi

python src/train_ppo.py \
    --episodes 50000 \
    --train-mode "$TRAIN_MODE" \
    --opponent "$OPPONENT" \
    $LOAD_ARG \
    --entropy-coef "$ENTROPY_COEF" \
    --update-epochs "$UPDATE_EPOCHS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --curriculum-start "$CURRICULUM_START" \
    --curriculum-end "$CURRICULUM_END" \
    --curriculum-ramp "$CURRICULUM_RAMP" \
    --curriculum-gate-wr "$CURRICULUM_GATE_WR" \
    --reward-win "$REWARD_WIN" \
    --reward-lose "$REWARD_LOSE" \
    --reward-hit "$REWARD_HIT" \
    --reward-miss "$REWARD_MISS" \
    --reward-sink "$REWARD_SINK" \
    --reward-efficient-sink "$REWARD_EFFICIENT_SINK" \
    --reward-adjacent-hit "$REWARD_ADJACENT_HIT" \
    --reward-per-turn "$REWARD_PER_TURN" \
    --reward-shots-between-sinks "$REWARD_SHOTS_BETWEEN_SINKS" \
    --save-path "models/${RUN_NAME}.pt" \
    --save-every 1000 \
    --eval-every 200 \
    --eval-games 20 \
    --seed 42 \
    --logdir "runs/${RUN_NAME}"

echo "End time: $(date)"
echo "Job completed"
