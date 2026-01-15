#!/bin/bash
# Finite POMDP Environments Test Script
# This script runs all PPO variants on finite POMDP environments
# Environments: Tiger, RiverSwim, SparseRewardPOMDP (Random)
# ========================================

# Common hyperparameters for finite POMDP tasks (smaller scale)
TOTAL_TIMESTEPS=1000
NUM_ENVS=8
NUM_MINIBATCHES=4
SEED=42
WANDB_PROJECT="finite-pomdp-bench"

# Tiger Environment (Theta = 0.2)
# ========================================
ENV_ID="Tiger-Theta2-v0"

echo "Running PPO variants on ${ENV_ID}..."

# Transformer-XL
uv run ppo_trxl.py \
    --gym-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --trxl-dim 256 \
    --trxl-memory-length 32 \
    --trxl-num-layers 1 \
    --trxl-num-heads 2 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_trxl_${ENV_ID} \
    --track

# Mamba v1
uv run ppo_mamba.py \
    --gym-id ${ENV_ID} \
    --mamba-version v1 \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --expand 1 \
    --hidden-dim 256 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_mamba_${ENV_ID} \
    --track

# Mamba v2
uv run ppo_mamba.py \
    --gym-id ${ENV_ID} \
    --mamba-version v2 \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --expand 1 \
    --d-state 64 \
    --d-conv 4 \
    --hidden-dim 256 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_mamba2_${ENV_ID} \
    --track

# LSTM
uv run ppo_lstm.py \
    --gym-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --rnn-type lstm \
    --rnn-hidden-dim 256 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_lstm_${ENV_ID} \
    --track

# GRU
uv run ppo_lstm.py \
    --gym-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --rnn-type gru \
    --rnn-hidden-dim 256 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_gru_${ENV_ID} \
    --track

# PPO (feedforward with frame stacking)
uv run ppo.py \
    --gym-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --hidden-dim 256 \
    --frame-stack 4 \
    --wandb-project-name ${WANDB_PROJECT} \
    --exp-name ppo_4_${ENV_ID} \
    --track

# Tiger Environment (Theta = 0.3)
# ========================================
ENV_ID="Tiger-Theta3-v0"

echo "Running PPO variants on ${ENV_ID}..."

uv run ppo_trxl.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --trxl-dim 256 --trxl-memory-length 32 --trxl-num-layers 1 --trxl-num-heads 2 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_trxl_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v1 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v2 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --d-state 64 --d-conv 4 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba2_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type lstm --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_lstm_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type gru --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_gru_${ENV_ID} --track
uv run ppo.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --hidden-dim 256 --frame-stack 4 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_4_${ENV_ID} --track

# Tiger Environment (Theta = 0.4)
# ========================================
ENV_ID="Tiger-Theta4-v0"

echo "Running PPO variants on ${ENV_ID}..."

uv run ppo_trxl.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --trxl-dim 256 --trxl-memory-length 32 --trxl-num-layers 1 --trxl-num-heads 2 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_trxl_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v1 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v2 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --d-state 64 --d-conv 4 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba2_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type lstm --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_lstm_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type gru --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_gru_${ENV_ID} --track
uv run ppo.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --hidden-dim 256 --frame-stack 4 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_4_${ENV_ID} --track

# RiverSwim Environment
# ========================================
ENV_ID="RiverSwim-Hard-v0"

echo "Running PPO variants on ${ENV_ID}..."

uv run ppo_trxl.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --trxl-dim 256 --trxl-memory-length 64 --trxl-num-layers 1 --trxl-num-heads 2 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_trxl_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v1 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba_${ENV_ID} --track
uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v2 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --d-state 64 --d-conv 4 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba2_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type lstm --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_lstm_${ENV_ID} --track
uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type gru --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_gru_${ENV_ID} --track
uv run ppo.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --hidden-dim 256 --frame-stack 4 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_4_${ENV_ID} --track

# SparseRewardPOMDP Environments (Random 0-4)
# ========================================
for i in {0..4}; do
    ENV_ID="SparseRewardPOMDP-Random${i}-v0"

    echo "Running PPO variants on ${ENV_ID}..."

    uv run ppo_trxl.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --trxl-dim 256 --trxl-memory-length 32 --trxl-num-layers 1 --trxl-num-heads 2 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_trxl_${ENV_ID} --track
    uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v1 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba_${ENV_ID} --track
    uv run ppo_mamba.py --gym-id ${ENV_ID} --mamba-version v2 --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --expand 1 --d-state 64 --d-conv 4 --hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_mamba2_${ENV_ID} --track
    uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type lstm --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_lstm_${ENV_ID} --track
    uv run ppo_lstm.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --rnn-type gru --rnn-hidden-dim 256 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_gru_${ENV_ID} --track
    uv run ppo.py --gym-id ${ENV_ID} --seed ${SEED} --total-timesteps ${TOTAL_TIMESTEPS} --num-envs ${NUM_ENVS} --num-minibatches ${NUM_MINIBATCHES} --hidden-dim 256 --frame-stack 4 --wandb-project-name ${WANDB_PROJECT} --exp-name ppo_4_${ENV_ID} --track
done

echo "All finite POMDP experiments completed!"