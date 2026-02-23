# Running Training with SLURM (sbatch)

This guide explains how to run training jobs on your school's HPC cluster using SLURM.

## Quick Start

### 1. Submit DQN Training Job
```bash
sbatch sbatch_train_dqn.sh
```

### 2. Submit PPO Training Job
```bash
sbatch sbatch_train_ppo.sh
```

### 3. Check Job Status
```bash
squeue -u $USER
```

### 4. View Output
```bash
# View latest log file
tail -f logs/dqn_<job_id>.out

# Or check error log
tail -f logs/dqn_<job_id>.err
```

## Benefits of Using sbatch

1. **GPU Access**: If your cluster has GPUs, sbatch can allocate them (much faster!)
2. **Background Execution**: Jobs run in background, you can disconnect
3. **Resource Management**: SLURM manages CPU/memory allocation
4. **Job Queue**: Jobs wait in queue if resources are busy
5. **Logging**: All output saved to log files automatically

## Customizing sbatch Scripts

### Enable GPU (if available)
Edit the sbatch scripts and uncomment these lines:
```bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
```

### Adjust Resources
Modify these lines based on your cluster's limits:
```bash
#SBATCH --time=24:00:00      # Max runtime (adjust as needed)
#SBATCH --cpus-per-task=4    # CPU cores
#SBATCH --mem=8G             # Memory
```

### Set Up Environment
Uncomment and modify the environment activation lines:
```bash
# For conda:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

# OR for modules:
module load python/3.10
source venv/bin/activate
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### Cancel a job
```bash
scancel <job_id>
```

### View detailed job info
```bash
scontrol show job <job_id>
```

## Performance Tips

1. **Use GPU**: Uncomment GPU lines if available - this can be 10-100x faster
2. **Batch Size**: Larger batch sizes benefit more from GPU
3. **No Visualization**: Don't use `--visualize` flag in sbatch (slows down, no terminal)
4. **Check GPU Usage**: After job starts, check if GPU is being used:
   ```bash
   nvidia-smi  # If GPU available
   ```

## Example: Custom Training Run

Create a custom script:
```bash
#!/bin/bash
#SBATCH --job-name=dqn_custom
#SBATCH --output=logs/dqn_custom_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

cd /data/class/cs175/mip1/NavalNet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

python src/train_dqn.py \
    --episodes 5000 \
    --save-path models/dqn_large.pt \
    --eval-every 50 \
    --seed 42
```

Then submit:
```bash
sbatch your_custom_script.sh
```

## Troubleshooting

### Job stays in PENDING
- Check queue limits: `squeue`
- May need to wait for resources
- Check partition availability: `sinfo`

### CUDA not available
- Make sure GPU lines are uncommented
- Check if cluster has GPUs: `sinfo -o "%P %G"`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Import errors
- Make sure environment is activated correctly
- Check Python path in script
- Verify all dependencies installed: `pip list`

### Out of memory
- Reduce batch size in training script
- Request more memory: `#SBATCH --mem=16G`
- Reduce buffer sizes in agent initialization
