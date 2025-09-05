# Large Language Models (LLMs)

Build a custom LLM, and then train it.
Written entirely in python, using just PyTorch.

For now, training can be done one of 4 datsets: [Tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~304k tokens), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-v1) (~118M tokens), [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~470M tokens), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B tokens).

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
## Meet your scripts 
### train
- Device setup. If an Nvidia GPU is available, pytorch gives its greetings.
- Command Line arguments are parsed: \
   - (Model arguments) : For a simple LLM, pass in arguments as you wish.
  For custom layers, or heterogeneous LLM, pass either a JSONL file under `--layer_configs`, or a string of list of dictionares as a string under `--layer_configs_jsonl`, but not both.
  If using custom LLM layers, any Attention/FFN param passed outside of the aforementioned arguments will be ignored. Check required and optional arguments in `parameters.md` when using custom layers.

   - (Training arguments) : dataset is expected to be prepared before training. 
  For gradient accumulation, `total_batch_size_str` is to be passed in as a sting of a math expression, eg: "2**12".
  Gradient accumulation steps are calculated as:
  $grad accum steps =$ $$\frac{total batch size}{batch size * block size}$$
  Thus ensure, TBS is a multiple of (B*T)
   If `act_recomp` is passed, Selective activation recomputation is used - All block/layer activations are recomputed on the fly.
- Data: Uses highly efficient memmory mapping. Data is pre-tokenized and stored in `.bin` files. For very large scales, and on distributed systems, this wont work.
- Model compilation: If on a Linux based system, and pytorch 2.0+ is used, model is compiled used `torch.compile()`
- Cosine Decay of learning rate is used.
- Auto Mixed Precision Training: used `torch.amp` with bfloat16 if available, or float16 with gradient scaling.
- if `save_model` is enanled, checkpoints are saved at specified intervals, and on the last step.
- After atleast one validation run, if `save_model` is enanled, best checkpoints are saved after a best validation loss is beat.

### model
> Not much to see here, unsure what too add.
- for MHA/GQA/MQA - the highly optimized `torch.nn.functional.scaled_dot_product_attention` is used.
- for MHLA - custom implementation from scratch.
- for inference, KV caching is enabled by default.
- Rolling context windows is used: only the last `block_size` tokens are used for context. KV cache is similarly trunacted.
- temperature can be anything from 0 (greedy sampling/deterministic/hardmax) to, well, infinite. However, models wildly hallucinate after ~1.75, and using a value >2 is not suggested.

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
- add multi-token prediction.
- Run tests using different configurations (and perhaps make a script for that)
- Explore SLURM for experimentation.
- Explore Parallelism mainly on the other repo
- Add ALiBi Positional encodings
- Implement Flash MLA as per [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- (Much Later) Explore fine tuning

## Acknowledgements
This repo was built to understand and implement all advancements in the field of LLMs after GPT2. Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) does the heavylifting of implementing all advancements before and upto GPT2. A significant amount of boilerplate code is taken from that repo. \
Thoroughly understanding a concept just from research papers could be extremly difficult and time consuming. Thanks to [Vizura AI](https://www.youtube.com/@vizuara), particularly their [DeepSeek from scratch playlist](https://youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms&si=0aa6DcNgnjCxhmUa), understanding the theory is far simpler. 
 
## Refrences
- (TODO) gather list of all research paper used in the project
