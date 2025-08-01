# Model and Training Configuration Parameters

This document outlines all the configurable parameters for the LLM model and the training script. These parameters are defined in the `LLMconfig` and `Trainconfig` dataclasses and can be overridden via command-line arguments in `train.py`.

Example usage : `python train.py --save_model --n_layer=8 ...`

> The default configurations do not apply to `train.sh`

## Training Configuration (`Trainconfig`)

These parameters control the training loop, optimization, and evaluation process.

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--total_batch_size_str`| `'2**11'` (`2048`) | int | Total number of tokens to process in one optimization step. This determines the gradient accumulation steps. |
| `--batch_size` | `2` (`2**1`) | int | The micro-batch size. Number of sequences processed in parallel on the GPU before a gradient accumulation step. |
| `--max_iters` | `2500` | int | The total number of training iterations (optimization steps) to run. |
| `--learning_rate` | `3e-4` | float | The maximum learning rate for the AdamW optimizer. |
| `--warmup_steps` | `100` | int | Number of initial steps to linearly warm up the learning rate from 0 to its maximum value. |
| `--grad_clip` | `1.0` | float | The value for gradient clipping to prevent exploding gradients. `0.0` disables it. |
| `--eval` | `False` | bool | If `True`, periodically evaluates the model on a validation set. |
| `--eval_interval` | `100` | int | The number of iterations between each evaluation run. |
| `--eval_iters` | `100` | int | The number of batches to use when estimating the evaluation loss. |
| `--save_model` | `False` | bool | If `True`, saves the model's state dictionary to `llm_model.pt` after training is complete. |

## Model Configuration (`LLMconfig`)

These parameters define the architecture of the Language Model itself.

### Core Architecture

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--n_layer` | `6` | int | The number of transformer blocks (layers) in the model. |
| `--n_embd` | `256` | int | The dimensionality of the token embeddings and the hidden states. |
| `--vocab_size` | `50304` | int | The size of the vocabulary. Default is for the `gpt2` tokenizer. |
| `--block_size` | `1024` (`2**10`) | int | The maximum context length (sequence length) the model can process. |
| `--dropout` | `0.2` | float | The dropout rate applied to various layers for regularization. |
| `--pos_emb` | `'rope'` | str | The type of positional embedding. Options: `'learn'`, `'sin'`, `'rope'`. |

### Feed-Forward / MLP Configuration

These settings apply to the standard MLP block or individual experts within an MoE layer.

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--up_dim` | `384` | int | The dimensionality of the hidden layer within the feed-forward network (MLP). |
| `--non_linearity` | `'gelu'` | str | The activation function to use. Options include `'relu'`, `'gelu'`, `'silu'`, etc. |

### Attention Mechanism

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--attn` | `'mla'` | str | The attention mechanism to use. Options: `'mha'`, `'mqa'`, `'gqa'`, `'mla'`. |
| `--n_head` | `8` | int | The number of attention heads for the queries. |
| `--n_kv_heads` | `4` | int | The number of heads for keys and values. Used for Grouped-Query Attention (`gqa`). Ignored for `mha` and `mqa`. |
| `--q_latent_dim` | `32` | int | **MLA Only.** The latent dimension for the down-projected query. |
| `--kv_latent_dim`| `32` | int | **MLA Only.** The latent dimension for the down-projected key/value context. |
| `--rope_head_dim`| `16` | int | **MLA with RoPE Only.** The head dimension used for the decoupled rotary embeddings. |

### Mixture of Experts (MoE)

These parameters are only active if `moe` is set to `True`.

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--moe` | `False` | bool | If `True`, replaces the MLP block in each transformer layer with a sparse MoE layer. |
| `--n_exp` | `16` | int | The total number of experts in each MoE layer. |
| `--n_shared` | `2` | int | The number of experts (out of `n_exp`) that are "shared" and process every token. |
| `--n_act` | `8` | int | The number of experts to activate per token. This total **includes** the shared experts. |
| `--aux_free` | `False` | bool | If `True`, uses the "Aux-Loss-Free" load balancing strategy with dynamic expert bias. |
| `--alpha` | `0.0001` | float | **Aux-Free Only.** The coefficient for the complementary auxiliary loss. |
| `--gamma` | `0.001` | float | **Aux-Free Only.** The update speed for the dynamic expert bias. |
| `--coeff` | `0.01` | float | **If `aux_free=False`**. The coefficient for the standard load-balancing auxiliary loss. |