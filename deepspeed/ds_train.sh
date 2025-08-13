#!/bin/bash

# --- Training Configuration Arguments ---
DATASET='tinystories'
MAX_ITERS=600
LEARNING_RATE=3e-4
WARMUP_STEPS=100
GRAD_CLIP=1.0
EVAL=true
EVAL_INTERVAL=250
EVAL_ITERS=30
SAVE_MODEL=false
FILE_NAME="llm_model"

# --- Model Configuration Arguments ---
N_LAYER=6
N_EMBD=384
VOCAB_SIZE=50304
BLOCK_SIZE=1024
DROPOUT=0.01
POS_EMB="rope" # Can be 'learn', 'sin', 'rope'

UP_DIM=1536
NON_LINEARITY="gelu" # Example: 'relu', 'gelu', 'silu'

ATTN="mla" # Can be 'mha', 'mqa', 'gqa', 'mla'
N_HEAD=8
N_KV_HEADS=4
Q_LATENT_DIM=64
KV_LATENT_DIM=64
ROPE_HEAD_DIM=48

MOE=false
N_EXP=16
N_SHARED=1
N_ACT=4
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
COEFF=0.01

# --- DeepSpeed Config ---
DS_CONFIG="ds_config.json"
CPU_OFFLOAD=flase

# --- Run Training ---
deepspeed ds_train.py \
    --dataset $DATASET \
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
    --n_exp $N_EXP \
    --n_shared $N_SHARED \
    --n_act $N_ACT \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --coeff $COEFF \
    --file_name $FILE_NAME \
    --ds_config $DS_CONFIG \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$MOE" = true ] && echo "--moe" ) \
    $( [ "$AUX_FREE" = true ] && echo "--aux_free" ) \
    $( [ "$CPU_OFFLOAD" = true ] && echo "--offload" )
