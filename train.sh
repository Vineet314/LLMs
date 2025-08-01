#!/bin/bash

# This script runs the train.py Python script with specified command-line arguments.

# --- Training Configuration Arguments ---
TOTAL_BATCH_SIZE_STR="2**12"
BATCH_SIZE=4
MAX_ITERS=3000
LEARNING_RATE=3e-4
WARMUP_STEPS=100
GRAD_CLIP=1.0
EVAL=true
EVAL_INTERVAL=100
EVAL_ITERS=20
SAVE_MODEL=true # Use "--no-save_model" to disable model saving

# --- Model Configuration Arguments ---
N_LAYER=6
N_EMBD=384
VOCAB_SIZE=50304
BLOCK_SIZE=1024
DROPOUT=0.1
POS_EMB="rope" # Can be 'learn', 'sin', 'rope'

UP_DIM=384
NON_LINEARITY="gelu" # Example: 'relu', 'gelu', 'silu'

ATTN="mla" # Can be 'mha', 'mqa', 'gqa', 'mla'
N_HEAD=8
N_KV_HEADS=4 # Only relevant if ATTN is 'gqa'
Q_LATENT_DIM=96 # Only relevant if ATTN is 'mla'
KV_LATENT_DIM=96 # Only relevant if ATTN is 'mla'
ROPE_HEAD_DIM=48 # Only relevant if POS_EMB is 'rope'

MOE=true
N_EXP=32
N_SHARED=2
N_ACT=8
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
CEOFF=0.01

# Construct the command
python train.py \
    --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --eval_interval $EVAL_INTERVAL \
    --eval_iters $EVAL_ITERS \
    --n_layer $N_LAYER \
    --n_embd $N_EMBD \
    --vocab_size $VOCAB_SIZE \
    --block_size $BLOCK_SIZE \
    --dropout $DROPOUT \
    --pos_emb $POS_EMB \
    --up_dim $UP_DIM \
    --non_linearity $NON_LINEARITY \
    --attn $ATTN \
    --n_head $N_HEAD \
    --n_kv_heads $N_KV_HEADS \
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --N_EXP $N_EXP \
    --n_shared $N_SHARED \
    --n_act $N_ACT \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --coeff $CEOFF \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$MOE" = true ] && echo "--moe" ) \
    $( [ "$AUX_FREE" = true ] && echo "--aux_free" ) 