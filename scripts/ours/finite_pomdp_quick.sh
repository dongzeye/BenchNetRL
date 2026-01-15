#!/bin/bash
# Quick Test Script for Finite POMDP Environments
# Runs short experiments for testing purposes
# ========================================

# Quick test hyperparameters (shorter runs for debugging)
TOTAL_TIMESTEPS=10000
NUM_ENVS=4
NUM_MINIBATCHES=2
SEED=42
WANDB_PROJECT="finite-pomdp-test"

# Test on a single environment from each type
ENVS=("Tiger-Theta2-v0" "RiverSwim-Hard-v0" "SparseRewardPOMDP-Random0-v0")

for ENV_ID in "${ENVS[@]}"; do
    echo "=========================================="
    echo "Testing PPO variants on ${ENV_ID}"
    echo "=========================================="

    # Transformer-XL
    echo "Running Transformer-XL..."
    uv run ppo_trxl.py \
        --gym-id ${ENV_ID} \
        --seed ${SEED} \
        --total-timesteps ${TOTAL_TIMESTEPS} \
        --num-envs ${NUM_ENVS} \
        --num-minibatches ${NUM_MINIBATCHES} \
        --trxl-dim 128 \
        --trxl-memory-length 32 \
        --trxl-num-layers 1 \
        --trxl-num-heads 2 \
        --wandb-project-name ${WANDB_PROJECT} \
        --exp-name ppo_trxl_${ENV_ID}_quick \
        --track

    # Mamba v1
    echo "Running Mamba v1..."
    uv run ppo_mamba.py \
        --gym-id ${ENV_ID} \
        --mamba-version v1 \
        --seed ${SEED} \
        --total-timesteps ${TOTAL_TIMESTEPS} \
        --num-envs ${NUM_ENVS} \
        --num-minibatches ${NUM_MINIBATCHES} \
        --expand 1 \
        --hidden-dim 128 \
        --wandb-project-name ${WANDB_PROJECT} \
        --exp-name ppo_mamba_${ENV_ID}_quick \
        --track

    # LSTM
    echo "Running LSTM..."
    uv run ppo_lstm.py \
        --gym-id ${ENV_ID} \
        --seed ${SEED} \
        --total-timesteps ${TOTAL_TIMESTEPS} \
        --num-envs ${NUM_ENVS} \
        --num-minibatches ${NUM_MINIBATCHES} \
        --rnn-type lstm \
        --rnn-hidden-dim 128 \
        --wandb-project-name ${WANDB_PROJECT} \
        --exp-name ppo_lstm_${ENV_ID}_quick \
        --track

    # PPO (feedforward)
    echo "Running PPO (feedforward)..."
    uv run ppo.py \
        --gym-id ${ENV_ID} \
        --seed ${SEED} \
        --total-timesteps ${TOTAL_TIMESTEPS} \
        --num-envs ${NUM_ENVS} \
        --num-minibatches ${NUM_MINIBATCHES} \
        --hidden-dim 128 \
        --frame-stack 4 \
        --wandb-project-name ${WANDB_PROJECT} \
        --exp-name ppo_4_${ENV_ID}_quick \
        --track

    echo "Completed testing on ${ENV_ID}"
    echo ""
done

echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
