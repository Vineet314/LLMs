#!/bin/bash

# This script runs the train.py Python script with specified command-line arguments.

# --- Training Configuration Arguments ---
BATCH_SIZE=16
MAX_ITERS=3000
EVAL_INTERVAL=100
EVAL_ITERS=20
LEARNING_RATE=3e-4
WARMUP_STEPS=100
GRAD_CLIP=1.0

# --- Model Configuration Arguments ---
VOCAB_SIZE=50304
BLOCK_SIZE=1024
N_EMBD=768
POS_EMB="sin" # Can be 'learn', 'sin', 'rope'
UP_DIM=3072
NON_LINEARITY="gelu" # Example: 'relu', 'gelu', 'silu'
DROPOUT=0.1
N_LAYER=8
ATTN="mla" # Can be 'mha', 'mqa', 'gqa', 'mla'
N_HEAD=12
N_KV_HEADS=4 # Only relevant if ATTN is 'gqa'
Q_LATENT_DIM=128 # Only relevant if ATTN is 'mla'
KV_LATENT_DIM=128 # Only relevant if ATTN is 'mla'
ROPE_HEAD_DIM=64 # Only relevant if POS_EMB is 'rope'

# --- Other Arguments ---
TOTAL_BATCH_SIZE_STR="2**13"
COMPILE_MODEL="--compile" # Use "--compile" to enable torch.compile()
PERFORM_EVAL="--no-eval" # Use "--no-eval" to disable evaluations
SAVE_MODEL="--save_model" # Use "--no-save_model" to disable model saving

# Construct the command
python train.py \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --eval_interval $EVAL_INTERVAL \
    --eval_iters $EVAL_ITERS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --vocab_size $VOCAB_SIZE \
    --block_size $BLOCK_SIZE \
    --n_embd $N_EMBD \
    --pos_emb $POS_EMB \
    --up_dim $UP_DIM \
    --non_linearity $NON_LINEARITY \
    --dropout $DROPOUT \
    --n_layer $N_LAYER \
    --attn $ATTN \
    --n_head $N_HEAD \
    --n_kv_heads $N_KV_HEADS \
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
    ${COMPILE_MODEL} \
    ${PERFORM_EVAL} \
    ${SAVE_MODEL}
