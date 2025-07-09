## Large Language Model (LLM) 

This repository contains examples and scripts for training Large Language Models (LLMs) using PyTorch.
It includes various implementations of LLMs, focusing on understanding the architecture and training techniques.
The project is structured to progressively build up from basic concepts to more advanced techniques.

For training on a single GPU, this project firstly aims at understanding the core of the LLM architecture: The attention mechanism.
  - `basic_llm.py`: Begin with the basic implementation from scratch, as per [Andrej Karpthy's nanoGPT](https://youtu.be/l8pRSuU81PU).
  - `flash_llm.py`: Identify and attend the inffefcienies in the previous code, and implement [flash attention](https://arxiv.org/abs/2205.14135) using `torch.scaled_dot_product_attention`
  - `kv_cache_llm.py`: Now that training is fast enough, Inference is highly inefficient. To handle that, we implement caching of Key Value vectors.
  - `mqa_gqa_llm.py`: KV Caching introduces significant memory bottlenecks. To reduce, we group (duplicate) Keys and Values, reducing computation at the cost of quality.
  - `mhla.py`: Compress KV vectors, way better than GQA, improves training and inference efficiency. Introduced by deepseek in [Deepseek V2](https://arxiv.org/abs/2405.04434) 
  - TODO: After MHLA is implemented, update the Multi GPU script.

For training on multiple GPUs, check out my repository [Distributed Pytorch](https://github.com/Vineet314/Distributed-Pytorch) which explores distributed training.