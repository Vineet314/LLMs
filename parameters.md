# Model and Training Configuration Parameters

This document outlines all the configurable parameters for the LLM model and the training script. These parameters are defined in the `LLMconfig` and `Trainconfig` dataclasses and can be overridden via command-line arguments in `train.py`. The default values can be found in the `train.py` script.

## Training Configuration (`Trainconfig`)

These parameters control the training loop, optimization, and evaluation process.

### Core Training

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--dataset` | str | The dataset to be used for training. Options: `'shakespeare'`, `'tinystories'`,`fineweb`, `wikitext`. |
| `--total_batch_size_str`| str | Total number of tokens to process in one optimization step, passed as a string expression (e.g., `'2**11'`). This determines the gradient accumulation steps. |
| `--batch_size` | int | The micro-batch size. Number of sequences processed in parallel on the GPU before a gradient accumulation step. |
| `--max_iters` | int | The total number of training iterations (optimization steps) to run. |
| `--learning_rate` | float | The maximum learning rate for the AdamW optimizer. |
| `--warmup_steps` | int | Number of initial steps to linearly warm up the learning rate from 0 to its maximum value. |
| `--grad_clip` | float | The value for gradient clipping to prevent exploding gradients. A value of `0.0` disables it. |
| `--act_recomp` | bool | If `True`, uses activation recomputation during the backward pass to save memory. |
| `--compile` | bool | If `True`, compiles the model using `torch.compile` for a potential speed-up. |

### Evaluation

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--eval` | bool | If `True`, periodically evaluates the model on a validation set. |
| `--eval_interval` | int | The number of iterations between each evaluation run. |
| `--eval_iters` | int | The number of batches to use when estimating the evaluation loss. |

### Logging and Checkpointing

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--save_model` | bool | If `True`, saves the model's state dictionary to a checkpoint file after training. |
| `--ckpt_interval` | int | The number of iterations between each checkpoint save. |
| `--file_name` | str | The base name of the checkpoint file to be saved. |
| `--wandb_log` | bool | If `True`, logs training metrics to Weights and Biases. |
| `--wandb_project` | str | The Weights and Biases project name to log to. |
| `--wandb_run_name`| str | The Weights and Biases run name. |

## Model Configuration (`LLMconfig`)

These parameters define the architecture of the Language Model itself.

### Core Architecture

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--n_layer` | int | The number of transformer blocks (layers) in the model. |
| `--n_embd` | int | The dimensionality of the token embeddings and the hidden states. |
| `--vocab_size` | int | The size of the vocabulary. The default in `train.py` is for the `gpt2` tokenizer. |
| `--block_size` | int | The maximum context length (sequence length) the model can process. |
| `--dropout` | float | The dropout rate applied to various layers for regularization. |
| `--pos_emb` | str | The type of positional embedding. Options: `'learn'`, `'sin'`, `'rope'`. |

### Normalization

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--norm` | str | The type of normalization layer to use. Options: `'layer'`, `'rms'`. |

### Feed-Forward / MLP Configuration

These settings apply to the standard MLP block or individual experts *within* an MoE layer.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--up_dim` | int | The dimensionality of the hidden layer within the feed-forward network (MLP). |
| `--non_linearity` | str | The activation function to use. Options include `'relu'`, `'gelu'`, `'swiglu'`, etc. |

### Attention Mechanism

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--attn` | str | The attention mechanism to use. Options: `'mha'`, `'mqa'`, `'gqa'`, `'mla'`. |
| `--n_head` | int | The number of attention heads for the queries. |
| `--n_kv_heads` | int | **GQA Only.** The number of heads for keys and values. |
| `--q_latent_dim` | int | **MLA Only.** The latent dimension for the down-projected query. |
| `--kv_latent_dim`| int | **MLA Only.** The latent dimension for the down-projected key/value context. |
| `--rope_head_dim`| int | **MLA with RoPE Only.** The head dimension used for the decoupled rotary embeddings. |

### Mixture of Experts (MoE)

These parameters are only active if `--moe` is passed in (i.e., set to `True`).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--moe` | bool | If `True`, replaces the MLP block in each transformer layer with a sparse MoE layer. |
| `--n_exp` | int | The total number of experts in each MoE layer. |
| `--n_shared` | int | The number of experts (out of `n_exp`) that are "shared" and process every token. |
| `--n_act` | int | The number of experts to activate per token. This total **includes** the shared experts. |
| `--aux_free` | bool | If `True`, uses the "Aux-Loss-Free" load balancing strategy with dynamic expert bias. |
| `--alpha` | float | **Aux-Free Only.** The coefficient for the complementary auxiliary loss. |
| `--gamma` | float | **Aux-Free Only.** The update speed for the dynamic expert bias. |
| `--coeff` | float | **If `aux_free=False`**. The coefficient for the standard load-balancing auxiliary loss. |

### Custom Layer Configuration

For building heterogeneous models where each layer can have a different configuration. If used, the model is considered `CUSTOM_LAYERS=True`.
***ONLY ONE OF THESE IS TO BE PROVIDED***

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--layer_configs` | str | A JSON string representing a list of dictionaries, where each dictionary defines the configuration for a specific layer. |
| `--layer_configs_jsonl` | str | The file path to a JSONL file where each line is a JSON object defining a layer's configuration. Checkout `layer_configs.jsonl` as an example |

Following are the keys that MUST be provided when using this: 
```python
>>> dict_keys(['attn', 'n_head', 'moe', 'up_dim', 'non_linearity'])
```
Depending upong `attn` and `moe`, following are also to be provided:
```python
>>> dict_kets(['n_kv_heads', 'q_latent_dim', 'kv_latent_dim', 'rope_head_dim', 'n_exp', 'n_shared', 'n_act', 'coeff', 'aux_free', 'alpha', 'gamma'])
```