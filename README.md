## Large Language Model (LLM) 

This repository contains examples and scripts for training Large Language Models (LLMs) using PyTorch.
It includes various implementations of LLMs, focusing on understanding the architecture and training techniques.
The project is structured to progressively build up from basic concepts to more advanced techniques.

For training on a single GPU, this project aims at understanding the core of the LLM architecture: The attention mechanism.
  - Begins with the basic implementation from scratch, spelled out in python in `basic_llm.py`
  - Then it identifies and attends the inffefcienies in the previous code, and implements flash attention in `flash_llm.py`
  - Now that training is fast enough, Inference is highly inefficient. To handle that, we implement caching of Key Value vectors, in `kv_cache_llm.py`
  - TODO: implement MQA, GQA and MHLA from scratch.
  - TODO: After MHLA is implemented, update the Multi GPU script.

For training on multiple GPUs, this project aims at understanding the core of the LLM architecture: The attention mechanism.
Check out my repository [Distributed Pytorch](https://github.com/Vineet314/Distributed-Pytorch) for more details on distributed training.