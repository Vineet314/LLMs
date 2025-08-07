#!/bin/bash

# AIM : To compare different Attention Mechanisms (All Dense Models):
    # 1. GQA with RoPE
    # 2. MLA with RoPE
    # 3. GQA with sin
    # 4. MLA with sin

# --------Global Settings--------
DATASET="tinystories"
TOTAL_BATCH_SIZE_STR="2**12"
BATCH_SIZE=4 # 2**2
MAX_ITERS=10000
LEARNING_RATE=6e-4
WARMUP_STEPS=100
GRAD_CLIP=1.0
EVAL=true
EVAL_INTERVAL=100
EVAL_ITERS=10
SAVE_MODEL=true
FILE_NAME="llm_model"

N_LAYER=8
N_EMBD=1024
VOCAB_SIZE=50304
BLOCK_SIZE=1024 # 2**10
DROPOUT=0.01
POS_EMB="rope" 

UP_DIM=2048            # if MoE : experimenting with (0.5, 2, 0.25, 4) x UP_DIM
NON_LINEARITY="swiglu" # "gelu" 

ATTN="mla"
N_HEAD=12
N_KV_HEADS=6 
Q_LATENT_DIM=256 
KV_LATENT_DIM=256 
ROPE_HEAD_DIM=128 

MOE=false
N_EXP=16
N_SHARED=1
N_ACT=4
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
CEOFF=0.01

echo -e "\n --------- TRAINING MODEL#1 : GQA-ROPE ------------ \n"

FILE_NAME="gqa_rope"
ATTN="gqa"
python ../../train.py \
    --dataset $DATASET \
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
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" )

echo -e "\n ---------TRAINING MODEL#2 : MLA:ROPE------------ \n"

FILE_NAME="mla_rope"
ATTN="mla"
python ../../train.py \
    --dataset $DATASET \
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
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" )

echo -e "\n --------- TRAINING MODEL#3 GQA:SIN ------------ \n"

FILE_NAME="gqa_sin"
ATTN="gqa"
POS_EMB="sin"
python ../../train.py \
    --dataset $DATASET \
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
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" )

echo -e "\n ---------TRAINING MODEL#4 : MLA:SIN------------ \n"

FILE_NAME="mla_sin"
ATTN="mla"
POS_EMB="sin"
python ../../train.py \
    --dataset $DATASET \
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
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" )