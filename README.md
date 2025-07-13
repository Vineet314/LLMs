## Large Language Model (LLM) 

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
  - `rope_llm.py` : (TODO) Implement the Rotary Postional Encodings, as per the [RoFormer](https://arxiv.org/pdf/2104.09864v1)
  - `rope_mhla_llm.py` : (TODO) Implement the Decoupled Rotary Positional Encodings, as in the [DeepSeek V2 ](https://arxiv.org/abs/2405.04434)
  - `flash_mhla_llm.py` : (TODO) Implement the goodiness of Flash Attention, but for Multi Head Latent Attention as per [Flash MLA](https://github.com/deepseek-ai/FlashMLA) by DeepSeek (This probably won't work on Kaggle)

For training on multiple GPUs, check out my repository [Distributed Pytorch](https://github.com/Vineet314/Distributed-Pytorch) which explores distributed training.