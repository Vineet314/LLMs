# Large Language Models (LLMs)

Build a custom LLM, and then train it.
Written entirely in python, using just PyTorch.

For now, training can be done one of 4 datsets: [Tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~304k tokens), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-v1)(~118M tokens), [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)(~470M tokens), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B tokens).

Key Architectures:
   - Attention mechanisms : Multi Head, Multi Query, Grouped Query and Multi Head Latent Attention. All have KV caching enabled by default.  
   - Feed Forward Network : Multi-Layer Perceptrons, Mixture of Experts (with or without DeepSeek's [auxiliary-loss-free load balancing](https://arxiv.org/pdf/2408.15664))
   - Heterogeneous models : Why keep all the transformer blocks same? Vary attention mechanisms, FFN types *within* the same model. Pass in `--layer_configs`. 
   - Check *[parameters](https://github.com/Vineet314/LLMs/blob/main/parameters.md)* for a full list of Model and training Parameters.

## Stepwise Quickstart
1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Run `pip install -r requirements.txt` (a virtual environment is highly suggested)
3. Choose one of the available datasets. cd into the data directorty and run `python prepare.py`
4. For logging, run `wandb login` and do the needful.
5. Then,
   - On a Windows Laptop with a CPU you can run:
   ```powershell
   PS> python train.py
   ```
   - Or a slightly complex one, if you have a GPU:
   ```powershell
   PS> python train.py --max_iters=5000 --save_model --attn='gqa' --pos_emb='rope' --n_head=16 --n_kv_heads=4 --eval --moe --aux_free --save_model
   ```
   - It is highly suggest to run it on a Linux-based OS, like Ubuntu, or on WSL as it enables you to use the Pytorch compiler. 
   Check out `train.sh` if you want to make any changes in settings and run:
   ```bash
   ~$ ./train.sh
   ```
6. Research. Change parameters, mechanisms. Sky is the limit. Document your findings.
7. Sample from your model. Run: 
   ```bash
   python sample.py \
      --model_path="llm_model_best.pt" \
      --prompt="why are we doing this" \
      --max_new_tokens=500 \
      --temperature=0.9 \
      --top_k=100
   ```


## Architectures
> Dev environement for implementing newer model architectures, starting from [NanoGPT](https://github.com/karpathy/nanoGPT) as the Base.
> Might Remove this dir totally.

This repository contains examples and scripts for training Large Language Models (LLMs) using PyTorch.
It includes various implementations of LLMs, focusing on understanding the architecture and training techniques.
The project is structured to progressively build up from basic concepts to more advanced techniques.

For training on a single GPU, this project firstly aims at understanding the core of the LLM architecture: The attention mechanism.
  - `basic_llm`: Begin with the basic implementation from scratch, as per [Andrej Karpthy's nanoGPT](https://youtu.be/l8pRSuU81PU).
  - `flash_llm`: Identify and attend the inffefcienies in the previous code, and implement [flash attention](https://arxiv.org/abs/2205.14135) using `torch.scaled_dot_product_attention`
  - `kv_cache_llm`: Now that training is fast enough, Inference is highly inefficient. To handle that, we implement caching of Key Value vectors.
  - `mqa_gqa_llm`: KV Caching introduces significant memory bottlenecks. To reduce, we group (duplicate) Keys and Values, reducing computation at the cost of quality.
  - `naive_mhla_llm`: Compress KV vectors, way better than GQA, improves training and inference efficiency. Introduced by deepseek in [Deepseek V2](https://arxiv.org/abs/2405.04434). Currently the RoPE less implementation, that's what Naive.
  - `sinusoidal_llm` : So far have been using learnable encodings, time to upgrade to sinusoidal, fixed encodings for positions. 
  - `rope_llm.py` : Implements the Rotary Postional Encodings, as per the [RoFormer](https://arxiv.org/pdf/2104.09864v1) on a model with Grouped Query Attention.
  - `rope_mhla_llm` : Implements the Decoupled Rotary Positional Encodings, as in the [DeepSeek V2](https://arxiv.org/abs/2405.04434).
  - `flash_mhla_llm` : (TODO) Implement the goodiness of Flash Attention, but for Multi Head Latent Attention as per [Flash MLA](https://github.com/deepseek-ai/FlashMLA) by DeepSeek (This probably won't work on Kaggle)
  - `moe` : Implements a traditional Mixture of Experts model using the Auxilary Load Balancing Loss technique
  - `deepseek_moe` : Upgrades the standard MoE architecture along the lines of [DeepSeek MoE](https://arxiv.org/abs/2401.06066), mainly the Fine-Grained Expert Segmentation and Shared Expert Isolation.
  - `aux_loss_free_moe` : Replaces the Aux Loss by a much smaller complimentry loss, introducing bias correction in router logits. Introduced in [DeepSeek V3](https://arxiv.org/abs/2412.19437)

## Experiments 
> Research Environment; search for optimal configuration, parameters and hyper parameters.
   - Exp1 : Comparison of different Attention Mechanisms (GQA, MLA), each with RoPE and SinusoidalPositional Embedding. (4 dense models) 
   - Exp2 : Previous experiment, but with auxiliary loss free MoE models. 
   - Exp3 : Compare simple LLM performence with heterogeneous LLMs

## Multi GPU training and DeepSpeed
[DeepSpeed](deepspeed.ai) provides a robust framework for training optimization, for single-node and multi-node systems. Yet to be explored.

For training on multiple GPUs, check out my repository [Distributed Pytorch](https://github.com/Vineet314/Distributed-Pytorch) which explores distributed training.
For a try, one can run the `kaggle-train.py` script as per the instructions given in the docstring. Or, it can be run on a single-node multi-gpu server as follows: 
```
# use kaggle only for Demo
!torchrun --standalone --nproc_per_node=8 kaggle-train.py --max_iters=5000 --moe --aux_free --eval --eval_interval=20 --max_iters=150
```
## TODO
- Make and use own tokenizer instead of `tiktoken`
- Run tests using different configurations (and perhaps make a script for that)
- Explore SLURM for experimentation.
- Explore Parallelism mainly on the other repo
- Add ALiBi Positional encodings
- Implement Flash MLA as per [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- (Much Later) Explore fine tuning
