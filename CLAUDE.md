# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BenchNetRL benchmarks neural network architectures for Reinforcement Learning tasks using Proximal Policy Optimization (PPO). The project compares feedforward, recurrent (LSTM/GRU), transformer (TrXL/GTrXL), and state-space model (Mamba/Mamba-2) architectures across various RL environments.

**Paper**: https://arxiv.org/abs/2505.15040

## Running Experiments

### Basic Training Commands

Each PPO variant is implemented in a separate file with its own hyperparameters:

```bash
# Vanilla PPO (feedforward)
python ppo.py --gym-id ALE/Breakout-v5 --total-timesteps 10000000 --num-envs 16 --track

# PPO with LSTM
python ppo_lstm.py --gym-id ALE/Breakout-v5 --rnn-hidden-dim 512 --track

# PPO with Mamba (requires mamba-ssm library)
python ppo_mamba.py --gym-id ALE/Breakout-v5 --mamba-version v1 --hidden-dim 450 --expand 1 --track

# PPO with Transformer-XL
python ppo_trxl.py --gym-id ALE/Breakout-v5 --trxl-dim 512 --trxl-memory-length 64 --track
```

### Using Experiment Scripts

Pre-configured experiment scripts are in `scripts/ours/`:

```bash
bash scripts/ours/atari.sh        # Atari environments
bash scripts/ours/mujoco.sh       # MuJoCo continuous control
bash scripts/ours/minigrid.sh     # MiniGrid environments
bash scripts/ours/classic_control.sh  # CartPole, LunarLander, etc.
```

### Key Arguments

- `--gym-id`: Environment ID (e.g., `ALE/Breakout-v5`, `HalfCheetah-v4`, `MiniGrid-DoorKey-8x8-v0`)
- `--track`: Enable Weights & Biases logging (required for metrics)
- `--wandb-project-name`: W&B project name (default: `ppo-mamba`)
- `--total-timesteps`: Training duration (default: 10M)
- `--num-envs`: Number of parallel environments (default: 8)
- `--save-model`: Save model checkpoints (default: False)
- `--save-interval`: Save every N updates (default: 100)

## Architecture Overview

### Core Components

**PPO Variants** (`ppo*.py`):
- Each file implements a complete training loop for a specific architecture
- All share the same PPO algorithm but differ in the neural network backbone
- Use common utilities from `exp_utils.py`, `env_utils.py`, `gae.py`, `layers.py`

**Agent Architecture Pattern**:
```python
Agent(nn.Module)
├── encoder: Extracts features from observations
│   ├── CNN encoder for image-based envs (Atari, MiniGrid)
│   ├── MLP encoder for vector-based envs (MuJoCo, Classic Control)
│   └── Handles frame stacking and observation preprocessing
├── recurrent/memory component: (optional, architecture-specific)
│   ├── LSTM/GRU cells (ppo_lstm.py)
│   ├── Mamba/Mamba2 blocks (ppo_mamba.py)
│   └── Transformer layers (ppo_trxl.py)
├── actor: Policy network (outputs action distribution)
│   ├── Discrete actions: Categorical distribution
│   └── Continuous actions: Normal distribution with learned std
└── critic: Value network (outputs state value estimate)
```

### Environment Pipeline

Environments are created through `env_utils.py` with automatic wrapper selection:

1. **Base environment creation**: `gym.make(gym_id)`
2. **Task-specific wrappers**: Applied based on environment type
   - Atari: NoopReset, MaxAndSkipEnv, EpisodicLife, FireReset, ClipReward
   - MiniGrid: ImgObsWrapper, RGBImgPartialObsWrapper
   - Classic: MaskObservationWrapper (for partial observability)
3. **Common wrappers**: RecordEpisodeStatistics, RecordVideo (if enabled)
4. **Frame stacking**: VecObservationStackWrapper (if frame_stack > 1)
5. **Vectorization**: `gym.vector.SyncVectorEnv` for parallel execution

### Training Loop Structure

All PPO variants follow this structure:

1. **Initialization**: Setup environments, agent, optimizer, storage tensors
2. **Rollout phase** (outer loop over updates):
   - Collect `num_steps` × `num_envs` transitions
   - Compute advantages using GAE (`gae.py`)
3. **Update phase**:
   - Shuffle and create minibatches
   - Run `update_epochs` epochs of optimization
   - Compute policy loss (clipped), value loss (optionally clipped), entropy bonus
   - Clip gradients and update parameters
4. **Logging**: Track metrics to W&B (returns, losses, SPS, GPU memory, inference latency)

### Key Differences Between Variants

**ppo.py**: No recurrence, processes each timestep independently
- Fastest training (3500+ SPS)
- Lowest memory usage
- Best for fully observable tasks

**ppo_lstm.py / ppo_mamba.py / ppo_trxl.py**: Maintain hidden states across timesteps
- Handle partial observability and temporal dependencies
- Must manage hidden state resets at episode boundaries
- Store and detach hidden states appropriately during training
- Different memory/compute tradeoffs (LSTM slowest, Mamba balanced, TrXL most memory-intensive)

## Development Notes

### Adding New Environments

1. Add environment factory function to `env_utils.py` (follow existing patterns)
2. Update agent encoder in the PPO variant if needed (see `Agent.__init__`)
3. Ensure observation space handling in `get_states()` method

### Modifying Network Architecture

- **Encoder layers**: Modify in `Agent.__init__()` based on environment type
- **Actor/Critic heads**: Defined separately in each PPO variant
- **Layer initialization**: Use `layer_init()` from `layers.py` (orthogonal init with proper scaling)

### Memory-Efficient Training

For environments with large observation spaces:
- Reduce `--num-envs` to lower parallel environments
- Decrease `--num-steps` to reduce rollout buffer size
- Use smaller `--hidden-dim` for the model
- For recurrent models, reduce memory length (`--trxl-memory-length`, LSTM sequence handling)

### Hyperparameter Tuning

Standard PPO hyperparameters in `exp_utils.py:add_common_args()`:
- Learning rate: 2.5e-4 (with annealing)
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip coefficient: 0.1
- Entropy coefficient: 0.01
- Value function coefficient: 0.5

Architecture-specific hyperparameters vary by model (see script examples in `scripts/ours/`).

### Debugging

**Check training progress**:
```python
# Console prints show: Update #, SPS, Return, losses, explained_variance
# Low explained_variance (< 0) indicates value function isn't learning well
```

**W&B metrics** (requires `--track`):
- `charts/episode_return`: Agent performance
- `charts/SPS`: Training throughput
- `metrics/inference_latency`: Per-step inference time
- `metrics/GPU_memory_allocated_GB`: Actual GPU usage
- `losses/*`: Training diagnostics

**Common issues**:
- CUDA out of memory: Reduce `--num-envs`, `--num-steps`, or `--hidden-dim`
- Slow training: Check SPS, consider using simpler architecture or fewer environments
- Poor performance: Verify environment-specific preprocessing in `Agent.get_states()`

## Installation Notes

**Mamba Support**: The `mamba-ssm` library (required for `ppo_mamba.py`) needs:
- Linux OS
- CUDA-enabled GPU with compatible drivers
- Follow installation from: https://github.com/state-spaces/mamba

**CUDA Setup**: If encountering `torch.cuda.reset_peak_memory_stats()` errors:
```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu123 torch torchvision torchaudio
```

Verify CUDA availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
