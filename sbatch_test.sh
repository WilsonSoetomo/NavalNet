#!/bin/bash
#SBATCH --job-name=test_battleship
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Request GPU if available (uncomment if your cluster has GPUs)
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment (adjust path/name as needed)
# Uncomment and modify based on your setup:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rl

# Change to project directory
cd /data/class/cs175/mip1/NavalNet

# Test DQN
echo "Testing DQN..."
python src/test_dqn.py \
    --model models/dqn.pt \
    --games 100 \
    --seed 123 \
    --save-results results_dqn.json

# Test PPO (if model exists)
if [ -f "models/ppo.pt" ]; then
    echo "Testing PPO..."
    python src/test_ppo.py \
        --model models/ppo.pt \
        --games 100 \
        --seed 123 \
        --save-results results_ppo.json
fi

echo "Testing completed"
