# Large Language Models (LLMs) 

Build a custom LLM, and then train it. for now, training is done on Tiny Shakespeare data, as we are limited to a single GPU

## Quickstart
On a Windows Laptop with a CPU you can run:
```powershell
PS C:\> python train.py --attn='mqa' --pos_emb='sin'
```
Or a slightly complex one, if you have a GPU:
```powershell
PS C:\> python train.py --max_iters=5000 --eval --save_model --attn='gqa' --pos_emb='rope' --n_head=16 --n_kv_heads=4
<#
Attention Mechanisms : --attn : 'mha','mqa','gqa','mla'
Positional Encodings : --pos_emb : 'learn','sin','rope'
#> 
```
It is highly suggest to run it on a Linux-based OS, like Ubuntu, or on WSL as it enables you to use the Pytorch compiler. 
Check out `train.sh` if you want to make any changes in settings and run:
```bash
~$ chmod +x train.sh
~$ ./train.sh
# Or you can also change settings directly in CLI:
# python train.py \
# --compile --eval --save_model \
# --attn='mla' --pos_emb='rope' \
# --max_iters=5000
```

## Architectures
This repository contains examples and scripts for training Large Language Models (LLMs) using PyTorch.
It includes various implementations of LLMs, focusing on understanding the architecture and training techniques.
The project is structured to progressively build up from basic concepts to more advanced techniques.

For training on a single GPU, this project firstly aims at understanding the core of the LLM architecture: The attention mechanism.
  - `basic_llm.py`: Begin with the basic implementation from scratch, as per [Andrej Karpthy's nanoGPT](https://youtu.be/l8pRSuU81PU).
  - `flash_llm.py`: Identify and attend the inffefcienies in the previous code, and implement [flash attention](https://arxiv.org/abs/2205.14135) using `torch.scaled_dot_product_attention`
  - `kv_cache_llm.py`: Now that training is fast enough, Inference is highly inefficient. To handle that, we implement caching of Key Value vectors.
  - `mqa_gqa_llm.py`: KV Caching introduces significant memory bottlenecks. To reduce, we group (duplicate) Keys and Values, reducing computation at the cost of quality.
  - `naive_mhla_llm.py`: Compress KV vectors, way better than GQA, improves training and inference efficiency. Introduced by deepseek in [Deepseek V2](https://arxiv.org/abs/2405.04434). Currently the RoPE less implementation, that's what Naive.
  - `sinusoidal_llm.py` : So far have been using learnable encodings, time to upgrade to sinusoidal, fixed encodings for positions. 
  - `rope_llm.py` : Implements the Rotary Postional Encodings, as per the [RoFormer](https://arxiv.org/pdf/2104.09864v1) on a model with Grouped Query Attention.
  - `rope_mhla_llm.py` : Implements the Decoupled Rotary Positional Encodings, as in the [DeepSeek V2 ](https://arxiv.org/abs/2405.04434).
  - `flash_mhla_llm.py` : (TODO) Implement the goodiness of Flash Attention, but for Multi Head Latent Attention as per [Flash MLA](https://github.com/deepseek-ai/FlashMLA) by DeepSeek (This probably won't work on Kaggle)

## Multi GPU training
For training on multiple GPUs, check out my repository [Distributed Pytorch](https://github.com/Vineet314/Distributed-Pytorch) which explores distributed training.
For a try, one can run the `kaggle-train.py` script as per the instructions given in the docstring. Or, it can be run on a single-node multi-gpu server as follows: 
```bash
torchrun --standalone --nproc_per_node=8 train.py --max_iters=5000
# compile is enabled by default
```
## TODO
- Explore and implement MoE architectures
- Implement Flash MLA as per [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- Add ALiBi Positional encodings
- Fix the eval in training script
- Add a sample.py and bash script
- (Much Later) Explore fine tuning
