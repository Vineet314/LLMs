#!/bin/bash

# ------------------ CONFIGURATION ------------------

# Training parameters
TOTAL_BATCH_SIZE_STR="2**13"    # used to calculate grad_accum_steps
BATCH_SIZE=8                    # microbatch size
MAX_ITERS=2500
# EVAL
# EVAL_INTERVAL
EVAL_ITERS=200
LEARNING_RATE=0.0003
WARMUP_STEPS=100
MAX_DECAY_STEPS=300
DEVICE="cuda"
COMPILE=true
SAVE_MODEL=true

# Model architecture
VOCAB_SIZE=50304
BLOCK_SIZE=1024                 # sequence length
N_EMBD=256
POS_EMB="learn"

UP_DIM=4
NON_LINEARITY="gelu"
DROPOUT=0.2
N_LAYER=6

ATTN_TYPE="gqa"
N_HEAD=8
N_KV_HEADS=4
# KV_LATENT_DIM=32
# Q_LATENT_DIM=32
# ROPE_HEAD_DIM=32

# Torchrun settings
NUM_GPUS=1
SCRIPT="train_llm.py"         # Make sure this matches your script name

# ------------------ EXECUTION ------------------

torchrun \
  --standalone \
  --nproc_per_node=$NUM_GPUS \
  $SCRIPT \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --max_iters $MAX_ITERS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE \
  --eval_iters $EVAL_ITERS \
  --warmup_steps $WARMUP_STEPS \
  --max_decay_steps $MAX_DECAY_STEPS \
  --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
  --n_embd $N_EMBD \
  --n_head $N_HEAD \
  --n_kv_heads $N_KV_HEADS \
  --n_layer $N_LAYER \
  --dropout $DROPOUT \
  --vocab_size $VOCAB_SIZE \
  --pos_emb $POS_EMB \
  --up_dim $UP_DIM \
  --non_linearity $NON_LINEARITY \
  --typ $ATTN_TYPE \
  $( [ "$COMPILE" = true ] && echo "--compile" ) \
  $( [ "$SAVE_MODEL" = true ] && echo "--save_model" )