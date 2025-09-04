#!/bin/bash

# Test Training of a single CUSTOM LAYERS model directly from BASH.

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
N_LAYER=8
VOCAB_SIZE=50304
BLOCK_SIZE=1024
N_EMBD=256
POS_EMB="rope" # Can be 'learn', 'sin', 'rope'
DROPOUT=0.01
NORM="rms" # Can be 'layer', 'rms'

# --- Layers Configuration Arguments ---

LAYER_CONFIGS='[
{"attn":"mla", "n_head":8, "moe":true, "up_dim":256 , "non_linearity":"swiglu", "n_kv_heads":4, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"mla", "n_head":8, "moe":true, "up_dim":256 , "non_linearity":"swiglu", "n_kv_heads":4, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"mla", "n_head":8, "moe":true, "up_dim":256 , "non_linearity":"swiglu", "n_kv_heads":4, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"mla", "n_head":8, "moe":true, "up_dim":256 , "non_linearity":"swiglu", "n_kv_heads":4, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"gqa", "n_head":8, "moe":false, "up_dim":512 , "non_linearity":"swiglu", "n_kv_heads":2, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"gqa", "n_head":8, "moe":false, "up_dim":512 , "non_linearity":"swiglu", "n_kv_heads":2, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"gqa", "n_head":8, "moe":false, "up_dim":512 , "non_linearity":"swiglu", "n_kv_heads":2, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001},
{"attn":"gqa", "n_head":8, "moe":false, "up_dim":512 , "non_linearity":"swiglu", "n_kv_heads":2, "q_latent_dim":64, "kv_latent_dim":64, "rope_head_dim":32, "n_exp":16, "n_shared":1, "n_act":4, "coeff":0.01, "aux_free":true, "alpha":0.0001, "gamma":0.001}
]'

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
    --layer_configs "$LAYER_CONFIGS" \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$ACT_RECOMP" = true ] && echo "--act_recomp" ) \
    $( [ "$WANDB_LOG" = true ] && echo "--wandb_log" )