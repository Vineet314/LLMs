# Model, Training, & Inference Configuration Parameters

This document outlines all the configurable parameters for the `model.py`, `train.py` and the `sample.py` scripts. These parameters are defined in the `LLMconfig`, `BlockConfig` and `Trainconfig` dataclasses and can be overridden via command-line arguments in `train.py` and `sample.py` scripts. The default values can be found in the respective scripts.

## Training Configuration (`Trainconfig`)

These parameters control the training loop, optimization, and evaluation process.

### Core Training

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `'shakespeare'` | str | The dataset to be used for training. Options: `'shakespeare'`, `'tinystories'`,`fineweb`, `wikitext`. |
| `--total_batch_size_str`| `'1024'` | str | Total number of tokens to process in one optimization step, passed as a string expression (e.g., `'2**11'`). This determines the gradient accumulation steps. |
| `--batch_size` | `2` | int | The micro-batch size. Number of sequences processed in parallel on the GPU before a gradient accumulation step. |
| `--max_iters` | `500` | int | The total number of training iterations (optimization steps) to run. |
| `--learning_rate` | `3e-4` | float | The maximum learning rate for the AdamW optimizer. |
| `--warmup_steps` | `100` | int | Number of initial steps to linearly warm up the learning rate from 0 to its maximum value. |
| `--grad_clip` | `1.0` | float | The value for gradient clipping to prevent exploding gradients. A value of `0.0` disables it. |
| `--act_recomp` | `False` | bool | If `True`, uses activation recomputation during the backward pass to save memory. |

### Evaluation

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--eval` | `False` | bool | If `True`, periodically evaluates the model on a validation set. |
| `--eval_interval` | `100` | int | The number of iterations between each evaluation run. |
| `--eval_iters` | `100` | int | The number of batches to use when estimating the evaluation loss. |

### Logging and Checkpointing

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--save_model` | `False` | bool | If `True`, saves the model's state dictionary to a checkpoint file after training. |
| `--ckpt_interval` | `250` | int | The number of iterations between each checkpoint save. |
| `--file_name` | `'llm_model'` | str | The base name of the checkpoint file to be saved. |
| `--wandb_log` | `False` | bool | If `True`, logs training metrics to Weights and Biases. |
| `--wandb_project` | `'llms'` | str | The Weights and Biases project name to log to. |
| `--wandb_run_name`| Current Timestamp | str | The Weights and Biases run name. Defaults to the current unix timestamp. |

## Model Configuration (`LLMconfig`)

These parameters define the architecture of the Language Model itself.

### Core Architecture

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--n_layer` | `6` | int | The number of transformer blocks (layers) in the model. |
| `--n_embd` | `256` | int | The dimensionality of the token embeddings and the hidden states. |
| `--vocab_size` | `50304` | int | The size of the vocabulary. The default in `train.py` is for the `gpt2` tokenizer. |
| `--block_size` | `512` | int | The maximum context length (sequence length) the model can process. |
| `--dropout` | `0.0` | float | The dropout rate applied to various layers for regularization. |
| `--pos_emb` | `'learn'` | str | The type of positional embedding. Options: `'learn'`, `'sin'`, `'rope'`. |

### Normalization

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--norm` | `'layer'` | str | The type of normalization layer to use. Options: `'layer'`, `'rms'`. |

### Feed-Forward / MLP Configuration

These settings apply to the standard MLP block or individual experts *within* an MoE layer.

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--up_dim` | `256` | int | The dimensionality of the hidden layer within the feed-forward network (MLP). |
| `--non_linearity` | `'gelu'` | str | The activation function to use. Options include `'relu'`, `'gelu'`, `'swiglu'`, etc. |

### Attention Mechanism

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--attn` | `'mqa'` | str | The attention mechanism to use. Options: `'mha'`, `'mqa'`, `'gqa'`, `'mla'`. |
| `--n_head` | `8` | int | The number of attention heads for the queries. |
| `--n_kv_heads` | `8` | int | **GQA Only.** The number of heads for keys and values. |
| `--q_latent_dim` | `None` | int | **MLA Only.** The latent dimension for the down-projected query. |
| `--kv_latent_dim`| `None` | int | **MLA Only.** The latent dimension for the down-projected key/value context. |
| `--rope_head_dim`| `None` | int | **MLA with RoPE Only.** The head dimension used for the decoupled rotary embeddings. |

### Mixture of Experts (MoE)

These parameters are only active if `--moe` is passed in (i.e., set to `True`).

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--moe` | `False` | bool | If `True`, replaces the MLP block in each transformer layer with a sparse MoE layer. |
| `--n_exp` | `None` | int | The total number of experts in each MoE layer. |
| `--n_shared` | `None` | int | The number of experts (out of `n_exp`) that are "shared" and process every token. |
| `--n_act` | `None` | int | The number of experts to activate per token. This total **includes** the shared experts. |
| `--aux_free` | `False` | bool | If `True`, uses the "Aux-Loss-Free" load balancing strategy with dynamic expert bias. |
| `--alpha` | `None` | float | **Aux-Free Only.** The coefficient for the complementary auxiliary loss. |
| `--gamma` | `None` | float | **Aux-Free Only.** The update speed for the dynamic expert bias. |
| `--coeff` | `None` | float | **If `aux_free=False`**. The coefficient for the standard load-balancing auxiliary loss. |

### Custom Layer Configuration

For building heterogeneous models where each layer can have a different configuration. If used, the model is considered `CUSTOM_LAYERS=True`.
***ATMOST ONE OF THESE IS TO BE PROVIDED***

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--layer_configs` | `None` | str | A JSON string representing a list of dictionaries, where each dictionary defines the configuration for a specific layer. |
| `--layer_configs_jsonl` | `None` | str | The file path to a JSONL file where each line is a JSON object defining a layer's configuration. Checkout `layer_configs.jsonl` as an example |

Following are the keys that MUST be provided when using this:
```python
>>> dict_keys(['attn', 'n_head', 'moe', 'up_dim', 'non_linearity'])
```
Depending upong `attn` and `moe`, following are also to be provided:
```python
>>> dict_kets(['n_kv_heads', 'q_latent_dim', 'kv_latent_dim', 'rope_head_dim', 'n_exp', 'n_shared', 'n_act', 'coeff', 'aux_free', 'alpha', 'gamma'])
```
## Inference Configuration (`sample.py`)

| Parameter | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `--model_path` | `'llm_model_best.pt'` | str | Path to the saved model checkpoint (.pt file). |
| `--prompt` | `'Once upon a time'` | str | The starting prompt for text generation. |
| `--max_new_tokens`| `300` | int | Maximum number of new tokens to generate. |
| `--temperature` | `0.9` | float | Generation temperature. `>1.0` is more creative, `<1.0` is more deterministic. Set to `0` for greedy sampling. |
| `--top_k` | `200` | int | Top-k sampling, which limits sampling to the k most likely tokens. |
