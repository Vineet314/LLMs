#!/bin/bash

# Test Training of a single model.

# --- Training Configuration Arguments ---
DATASET='fineweb'
TOTAL_BATCH_SIZE_STR="2**11"
BATCH_SIZE=2
MAX_ITERS=500
LEARNING_RATE=3e-4
WARMUP_STEPS=100
GRAD_CLIP=1.0
EVAL=false
EVAL_INTERVAL=125
EVAL_ITERS=20
SAVE_MODEL=false
CKPT_INTERVAL=250 # perhpas a good number is max_iters//10?
FILE_NAME="llm_model"
ACT_RECOMP=true
WANDB_LOG=false
WANDB_PROJECT="llms"
WANDB_RUN_NAME="trial_run_w/o_mix"

# --- Model Configuration Arguments ---
N_LAYER=7
N_EMBD=256
VOCAB_SIZE=50304
BLOCK_SIZE=1024
DROPOUT=0.01
POS_EMB="rope" # Can be 'learn', 'sin', 'rope'
NORM="rms" # Can be 'layer', 'rms'
UP_DIM=256
NON_LINEARITY="swiglu" # Example: 'relu', 'gelu', 'silu'

ATTN="mla" # Can be 'mha', 'mqa', 'gqa', 'mla'
N_HEAD=8
N_KV_HEADS=4 # Only relevant if ATTN is 'gqa'
Q_LATENT_DIM=32 # Only relevant if ATTN is 'mla'
KV_LATENT_DIM=32 # Only relevant if ATTN is 'mla'
ROPE_HEAD_DIM=16 # Only relevant if POS_EMB is 'rope'

MOE=true
N_EXP=16
N_SHARED=1
N_ACT=4
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
CEOFF=0.01

# Construct the command
python train.py \
    --dataset $DATASET \
    --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --eval_interval $EVAL_INTERVAL \
    --eval_iters $EVAL_ITERS \
    --file_name $FILE_NAME \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --ckpt_interval $CKPT_INTERVAL \
    --n_layer $N_LAYER \
    --norm $NORM \
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
    --n_exp $N_EXP \
    --n_shared $N_SHARED \
    --n_act $N_ACT \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --coeff $CEOFF \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$MOE" = true ] && echo "--moe" ) \
    $( [ "$ACT_RECOMP" = true ] && echo "--act_recomp" ) \
    $( [ "$WANDB_LOG" = true ] && echo "--wandb_log" ) \
    $( [ "$AUX_FREE" = true ] && echo "--aux_free" )