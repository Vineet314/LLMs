#!/bin/bash

# Define default arguments
MODEL_PATH="6_sinusoidal_gqa/mqa_gqa_llm_model.pt"
MAX_NEW_TOKENS=500
START_TEXT="crazy bro"
DEVICE="cuda"
TEMPERATURE=1.0
TOP_K=""  # Leave blank to omit top_k
COMPILE=false

# Run the training script with arguments
python model_sample.py \
  --model_path $MODEL_PATH \
  --max_new_tokens $MAX_NEW_TOKENS \
  --start_text "$START_TEXT" \
  --device $DEVICE \
  --temperature $TEMPERATURE \
  $( [ -n "$TOP_K" ] && echo "--top_k $TOP_K" ) \
  $( [ "$COMPILE" = true ] && echo "--compile" )
