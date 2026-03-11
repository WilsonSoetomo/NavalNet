# NavalNet

RL agents for Battleship: DQN and PPO with separate placement and shooting heads.

## Training (SLURM)

### DQN — `sbatch_train_dqn.sh`

Trains the DQN agent. Edit the config block at the top before submitting:

| Variable | Options | Description |
|----------|---------|-------------|
| `OPPONENT` | `random`, `hunt_target`, `curriculum` | Opponent policy |
| `TRAIN_MODE` | `shooting`, `placement`, `full` | `shooting` = target practice (no opponent turn); `placement` = placement only, scored by HuntTarget; `full` = both heads |
| `LOAD_MODEL` | path or empty | Pre-trained checkpoint (e.g. `models/dqn_5k.pt`) |
| `SAVE_CHECKPOINT_EVERY` | N or 0 | Save `model_epN.pt`, `model_ep2N.pt`, etc. (0 = disabled) |

```bash
sbatch sbatch_train_dqn.sh
```

Output: `models/dqn_MMDDYYYY_HHMM_<mode>_<opponent>.pt`, logs in `logs/`, TensorBoard in `runs/`.

### PPO — `sbatch_train_ppo.sh`

Trains the PPO agent. Same `OPPONENT` and `TRAIN_MODE` options. PPO-specific settings:

| Variable | Description |
|----------|-------------|
| `ENTROPY_COEF` | Entropy bonus for exploration |
| `UPDATE_EPOCHS` | PPO update epochs per rollout |
| `ROLLOUT_STEPS` | Steps per rollout before update |

```bash
sbatch sbatch_train_ppo.sh
```

Output: `models/ppo_MMDDYYYY_HHMM_<mode>_<opponent>.pt`.

See [README_SLURM.md](README_SLURM.md) for SLURM usage, GPU setup, and troubleshooting.

---

## Play & Evaluation

### play_web — Interactive game vs model

Web UI to play Battleship against a trained DQN or PPO model. You place ships, then shoot; the model shoots back.

```bash
python src/play_web.py --port 6060
```

Open http://localhost:6060 (or the forwarded port over SSH). Select a model from the dropdown and start a game.

### play_human_eval — Human & model evaluation

Web UI for two modes:

1. **Human eval**: You play 100 episodes (same setup as DQN shooting training — random ship placement, you shoot until all sunk, no opponent turn). Metrics: shots, avg shots/sink, efficiency, reward.

2. **Bot/model eval**: Run Random, Hunt/Target, or a trained model against random placements for N episodes. Progress streams to the log box.

```bash
python src/play_human_eval.py --port 6061
```

Open http://localhost:6061. Use **Start Human Eval** for human play, or **Bot / Model Eval** to benchmark models.

---

## Project layout

```
models/          # Saved checkpoints (.pt)
logs/            # SLURM job logs
runs/            # TensorBoard logs
src/
  train_dqn.py   # DQN training
  train_ppo.py   # PPO training
  play_web.py    # Web game vs model
  play_human_eval.py  # Human & model evaluation
  extract_checkpoint.py  # Copy checkpoint (e.g. ep5k weights)
```
