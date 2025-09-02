# Train a single model in PowerShell

# For testing purposes on Windows

# --- Training Configuration Arguments ---
$DATASET = 'shakespeare'
$TOTAL_BATCH_SIZE_STR = "2**11"
$BATCH_SIZE = 2
$MAX_ITERS = 100
$LEARNING_RATE = 7e-5
$WARMUP_STEPS = 50
$GRAD_CLIP = 0.9
$EVAL = $true
$EVAL_INTERVAL = 25
$EVAL_ITERS = 10
$SAVE_MODEL = $true
$CKPT_INTERVAL = 20
$FILE_NAME = "llm_model"
$ACT_RECOMP = $true
$WANDB_LOG = $true
$WANDB_PROJECT = "llms"
$WANDB_RUN_NAME = "trial"

# --- Model Configuration Arguments ---
$N_LAYER = 5
$N_EMBD = 512
$VOCAB_SIZE = 50304
$BLOCK_SIZE = 512
$DROPOUT = 0.01
$POS_EMB = "rope"
$NORM = "rms"
$UP_DIM = 256
$NON_LINEARITY = "swiglu"

$ATTN = "mla"
$N_HEAD = 8
$N_KV_HEADS = 4
$Q_LATENT_DIM = 32
$KV_LATENT_DIM = 32
$ROPE_HEAD_DIM = 64

$MOE = $true
$N_EXP = 8
$N_SHARED = 1
$N_ACT = 4
$AUX_FREE = $true
$ALPHA = 0.0001
$GAMMA = 0.001
$CEOFF = 0.01

# Construct the argument list
$args = @(
    "--dataset", $DATASET
    "--total_batch_size_str", $TOTAL_BATCH_SIZE_STR
    "--batch_size", $BATCH_SIZE
    "--max_iters", $MAX_ITERS
    "--learning_rate", $LEARNING_RATE
    "--warmup_steps", $WARMUP_STEPS
    "--grad_clip", $GRAD_CLIP
    "--eval_interval", $EVAL_INTERVAL
    "--eval_iters", $EVAL_ITERS
    "--file_name", $FILE_NAME
    "--wandb_project", $WANDB_PROJECT
    "--wandb_run_name", $WANDB_RUN_NAME
    "--ckpt_interval", $CKPT_INTERVAL
    "--n_layer", $N_LAYER
    "--norm", $NORM
    "--n_embd", $N_EMBD
    "--vocab_size", $VOCAB_SIZE
    "--block_size", $BLOCK_SIZE
    "--dropout", $DROPOUT
    "--pos_emb", $POS_EMB
    "--up_dim", $UP_DIM
    "--non_linearity", $NON_LINEARITY
    "--attn", $ATTN
    "--n_head", $N_HEAD
    "--n_kv_heads", $N_KV_HEADS
    "--q_latent_dim", $Q_LATENT_DIM
    "--kv_latent_dim", $KV_LATENT_DIM
    "--rope_head_dim", $ROPE_HEAD_DIM
    "--n_exp", $N_EXP
    "--n_shared", $N_SHARED
    "--n_act", $N_ACT
    "--alpha", $ALPHA
    "--gamma", $GAMMA
    "--coeff", $CEOFF
)

# Conditional flags
if ($SAVE_MODEL)   { $args += "--save_model" }
if ($EVAL)         { $args += "--eval" }
if ($MOE)          { $args += "--moe" }
if ($ACT_RECOMP)   { $args += "--act_recomp" }
if ($AUX_FREE)     { $args += "--aux_free" }
if ($WANDB_LOG)    { $args += "--wandb_log" }

# Run the command
python train.py @args
