'''This trains a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

This script is to be run from as ./train.sh
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tiktoken
import argparse
import torch
import math

from dataclasses import dataclass
from time import time

from sinusoidal_llm import LLM

torch.set_float32_matmul_precision("high") # OPTIM 1 brought dt from 230 to 170

@dataclass
class LLMconfig:
    # hyperparameters
    block_size : int  # what is the maximum context length for predictions?
    vocab_size : int # OPTIM 4 (along with grad clipping) brought dt from 95 to 90
    n_kv_heads : int
    n_embd : int
    n_head : int
    n_layer: int
    dropout: float

@dataclass
class Trainconfig:
    total_batch_size : int
    batch_size : int
    max_iters : int
    eval : bool
    eval_interval : int
    learning_rate : float
    warmup_steps : int
    max_decay_steps : int
    device : str
    eval_iters : int
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool

ModelConfig = LLMconfig(
    # hyperparameters
    block_size = 1024, # what is the maximum context length for predictions?
    vocab_size = 50304, # OPTIM 4 (along with grad clipping) brought dt from 95 to 90
    n_kv_heads = 4,
    n_embd = 768,
    n_head = 8,
    n_layer= 6,
    dropout= 0.2)

TrainingConfig = Trainconfig(
    total_batch_size = 2**16,
    batch_size = 4, # how many independent sequences will we process in parallel?
    max_iters = 500,
    eval = False,
    eval_interval = 50,
    learning_rate = 3e-4,
    warmup_steps = 25,
    max_decay_steps = 75,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    eval_iters = 200,
    compile = False if os.name != 'posix' else True,
    save_model = True)

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        # training data
        with open('../_data/shakesphere.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+(B*T+1)]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        # advance the position
        self.current_position += B*T

        if self.current_position + (B*T+1)  > len(self.tokens):
            self.current_position = 0
        return x,y

def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    # 1) linear warump for warmup_steps:
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > TrainingConfig.max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (TrainingConfig.max_decay_steps - TrainingConfig.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)  # ensure it does
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model:LLM, TrainingConfig:Trainconfig, eval_loader:DataLoader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(TrainingConfig.eval_iters)
        for k in range(TrainingConfig.eval_iters):
            X, Y = eval_loader.next_batch()
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--device',        type=str,   default=TrainingConfig.device,        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--max_decay_steps',type=int,  default=TrainingConfig.max_decay_steps,help='Maximum decay steps for learning rate')

    parser.add_argument('--n_embd',        type=int,   default=ModelConfig.n_embd,          help='Number of embedding dimensions')
    parser.add_argument('--n_head',        type=int,   default=ModelConfig.n_head,          help='Number of attention heads')
    parser.add_argument('--n_kv_heads',    type=int,   default=ModelConfig.n_kv_heads,      help='Number of key/value heads for GQA')
    parser.add_argument('--n_layer',       type=int,   default=ModelConfig.n_layer,         help='Number of layers in the model')
    parser.add_argument('--dropout',       type=float, default=ModelConfig.dropout,         help='Dropout rate')
    parser.add_argument('--vocab_size',    type=int,   default=ModelConfig.vocab_size,      help='Vocabulary size for the model')
    parser.add_argument('--block_size',    type=int,   default=ModelConfig.block_size,      help='Block size for training')
    parser.add_argument('--total_batch_size_str', type=str,      default='2**16',         help='Total batch size for training passed in as a string expression')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile()')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')
    return parser.parse_args()

def main(model:LLM, TrainingConfig:Trainconfig, ModelConfig:LLMconfig, optimizer:torch.optim.Optimizer):
    B = TrainingConfig.batch_size
    T = ModelConfig.block_size
    assert TrainingConfig.total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = TrainingConfig.total_batch_size // (B * T)

    train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size)
    eval_loader  = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size)

    for iter in range(TrainingConfig.max_iters):
        t3 = t4 = 0
        t0 = time() 
        # every once in a while evaluate the loss on train and val sets
        # if iter % ModelConfig.eval_interval == 0 or iter == ModelConfig.max_iters - 1:
            # t3 = time()
        #     losses = estimate_loss(model, ModelConfig, eval_loader)
            # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # t4 = time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0 

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x,y = x.to(device=TrainingConfig.device), y.to(device=TrainingConfig.device)
            # evaluate the loss
            if torch.cuda.is_bf16_supported(): # OPTIM 2 brought dt from 170 to 130
                with torch.autocast(device_type=TrainingConfig.device, dtype=torch.bfloat16):
                    _, loss, _ = model(x,y)
            else: # need to learn about gradient scalers 
                _, loss, _ = model(x,y)
            loss = loss/grad_accum_steps
            loss_accum += loss.detach()  
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(iter, TrainingConfig) # OPTIM 5 : i plugged in,now its almost 68ms
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time()
        dt = (t1-t0-(t4-t3))*1000
        print(f"step: {iter} | train loss:{loss_accum:.4f} | dt: {dt:.2f}ms | grad_acum_steps:{grad_accum_steps}")

    if TrainingConfig.save_model:
        torch.save(model, 'mqa_gqa_llm_model.pt')
        print("\nsaved run to mqa_gqa_llm_model.pt")

if __name__ == '__main__':
    args = parse_args()
    for key, value in vars(args).items():
        # need to eval the total_batch_size to get the grad_accum_steps
        if key == 'total_batch_size_str':
            value = eval(value)
            setattr(TrainingConfig, 'total_batch_size', value)
        else:
            if hasattr(TrainingConfig, key):
                setattr(TrainingConfig, key, value)
            else:
                setattr(ModelConfig, key, value)

    model = LLM(ModelConfig).to(TrainingConfig.device)
    if TrainingConfig.compile: # OPTIM 3 brought dt from 130 to 95ms
        print("Compiling the model with torch.compile()")
        model = torch.compile(model)

    # Training
    print(f"total parameters = {model.get_num_params():,}")
    optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=TrainingConfig.device,prints=False)

    # start training
    main(model=model, TrainingConfig=TrainingConfig, ModelConfig=ModelConfig, optimizer=optimizer)
