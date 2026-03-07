#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --job-name=dqn_battleship
#SBATCH --output=logs/dqn_%j.out
#SBATCH --error=logs/dqn_%j.err
#SBATCH --time=06:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=20GB            ## Allocated Memory
#SBATCH --cpus-per-task=8     ## Number of CPU cores

mkdir -p logs

source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate rl

# ── Configuration ────────────────────────────────────────────────
# Options: "random", "hunt_target", "curriculum"
OPPONENT="curriculum"

# Pre-trained model (empty = train from scratch)
# NOTE: old 1-channel models are incompatible with the new 5-channel arch
LOAD_MODEL=""

# Reward tuning
REWARD_WIN=50.0
REWARD_LOSE=-50.0
REWARD_HIT=2.0
REWARD_SINK=10.0
REWARD_EFFICIENT_SINK=3.0

# Curriculum: gated mode (ramp only when winning enough)
CURRICULUM_START=0.0
CURRICULUM_END=0.8
CURRICULUM_GATE_WR=0.40
CURRICULUM_RAMP=10000

# Empty = use defaults (1.0 from scratch, 0.4 from loaded)
EPSILON_START=""

RUN_NAME="dqn_$(date +%m%d%Y_%H%M)_${OPPONENT}"
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

EPS_ARG=""
if [ -n "$EPSILON_START" ]; then
    EPS_ARG="--epsilon-start $EPSILON_START"
fi

python src/train_dqn.py \
    --episodes 20000 \
    --opponent "$OPPONENT" \
    $LOAD_ARG \
    $EPS_ARG \
    --curriculum-start "$CURRICULUM_START" \
    --curriculum-end "$CURRICULUM_END" \
    --curriculum-ramp "$CURRICULUM_RAMP" \
    --curriculum-gate-wr "$CURRICULUM_GATE_WR" \
    --reward-win "$REWARD_WIN" \
    --reward-lose "$REWARD_LOSE" \
    --reward-hit "$REWARD_HIT" \
    --reward-sink "$REWARD_SINK" \
    --reward-efficient-sink "$REWARD_EFFICIENT_SINK" \
    --save-path "models/${RUN_NAME}.pt" \
    --save-every 1000 \
    --eval-every 200 \
    --eval-games 20 \
    --seed 42 \
    --logdir "runs/${RUN_NAME}"

echo "End time: $(date)"
echo "Job completed"
