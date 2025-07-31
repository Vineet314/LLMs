'''This script trains an LLM model based on the user's CLI inputs. 

This code is highly inspired by Andrej Karpathy's work on his nanoGPT : https://github.com/karpathy/nanoGPT/

This script is made to be run on a single GPU(preferred)/CPU. For a more sophisticated run involving distributed systems,
checkout : https://github.com/Vineet314/Distributed-Pytorch/blob/master/LLMs/kaggle-train-v2.py

To run this, either use a bash script, or run:
python train.py --compile --max_iters=100 --attn='gqa' --pos_emb='sin'

For details about arguments, see the LLMConfig and TrainConfig classes.'''

# import warnings ; warnings.filterwarnings("ignore")
import math
import torch
import argparse
import tiktoken
import requests

from time import time
from typing import Literal
from dataclasses import dataclass
from contextlib import nullcontext

from model import LLM

# ______________DEVICE and DTYPE SETUP_________________
torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision('medium')   # Not sure if this has any effect when used with Auto Mixed Precision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ctx = torch.amp.autocast(device_type=device, dtype=getattr(torch, dtype))
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16')) if device == 'cuda' else nullcontext()

# ____________PARAMS-CONFIG_________________

@dataclass
class Trainconfig:
    total_batch_size : int
    batch_size : int
    max_iters : int
    eval : bool
    eval_interval : int
    eval_iters : int
    learning_rate : float
    warmup_steps : int
    grad_clip : int
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool

@dataclass
class LLMconfig:
    # token params
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']

    # Neural Network
    up_dim  : int
    non_linearity : str | Literal['elu','lrelu','relu', 'gelu', 'swish', 'mish', 'silu', 'selu','celu']
    dropout : float
    n_layer : int
    n_exp : int
    n_act : int
    coeff : int
    
    # Attention
    attn : str | Literal['mha', 'mqa', 'gqa', 'mla', 'fmla']
    # kv_cache : bool
    n_head : int
    n_kv_heads : int 
        # Only for mla 
    q_latent_dim  : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None

ModelConfig = LLMconfig(
    # token params
    vocab_size = 50304, 
    block_size = 2**10, 
    n_embd = 256, 
    pos_emb = 'rope',
    # MoE
    up_dim = 1024, 
    non_linearity = 'gelu',  
    dropout=0.2,
    n_layer = 6,
    n_exp = 1,
    n_act = 1,
    coeff = 0.05,
    # Attention
    attn = 'mla', 
    # kv_cache = True, 
    n_head = 8,
    n_kv_heads = 4, 
    # MHLA
    q_latent_dim = 32, 
    kv_latent_dim = 32,
    rope_head_dim = 16)

TrainingConfig = Trainconfig(
    
    total_batch_size = 2**13,
    batch_size = 2**3, # how many independent sequences will we process in parallel?
    max_iters = 2500,
    eval = False,
    eval_interval=100,
    eval_iters=100,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0,    
    compile = True,
    save_model = True)

# ___________ CLI-OVERRIDE__________________

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--grad_clip',     type=float,  default=TrainingConfig.grad_clip,    help='Gradient Clip value')

    parser.add_argument('--vocab_size',  type=int,   default=ModelConfig.vocab_size,  help='Vocabulary size for the model')
    parser.add_argument('--block_size',  type=int,   default=ModelConfig.block_size,  help='Block size for the model')
    parser.add_argument('--n_embd',      type=int,   default=ModelConfig.n_embd,      help='Embedding dimension for the model')
    parser.add_argument('--pos_emb',     type=str,   default=ModelConfig.pos_emb,     help='Type of positional encoding (learn, sin, rope)')
    parser.add_argument('--up_dim',      type=int,   default=ModelConfig.up_dim,      help='Up dimension for the Expert in the model')
    parser.add_argument('--non_linearity',type=str,   default=ModelConfig.non_linearity,help='Non-linearity for the Expert in the model')
    parser.add_argument('--n_exp',       type=int,   default=ModelConfig.n_exp,       help='Number of Experts in the model')
    parser.add_argument('--n_act',       type=int,   default=ModelConfig.n_act,       help='Number of Active Experts in the model')
    parser.add_argument('--coeff',       type=int,   default=ModelConfig.coeff,       help='Loss Coefficient for the MoE')
    parser.add_argument('--dropout',     type=float, default=ModelConfig.dropout,     help='Dropout rate for the model')
    parser.add_argument('--n_layer',     type=int,   default=ModelConfig.n_layer,     help='Number of layers in the model')
    parser.add_argument('--attn',        type=str,   default=ModelConfig.attn,         help='Type of attention mechanism (mha, mqa, gqa, mla)')
    parser.add_argument('--n_head',      type=int,   default=ModelConfig.n_head,      help='Number of attention heads in the model')
    parser.add_argument('--n_kv_heads',  type=int,   default=ModelConfig.n_kv_heads,  help='Number of KV heads in the model (only for gqa)')
    parser.add_argument('--q_latent_dim',  type=int, default=ModelConfig.q_latent_dim,help='Query latent dimension (only for mla)')
    parser.add_argument('--kv_latent_dim', type=int, default=ModelConfig.kv_latent_dim,help='KV latent dimension (only for mla)')
    parser.add_argument('--rope_head_dim', type=int, default=ModelConfig.rope_head_dim,help='RoPE head dimension (only for mla)')
    
    parser.add_argument('--total_batch_size_str', type=str, default='2**13', help='Total batch size for training passed in as a string expression')
    parser.add_argument('--compile',    action='store_true', help='Whether to compile the model with torch.compile()')
    parser.add_argument('--eval',       action='store_true', help='Wheter to perform Evalutions once a while')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')

    return parser.parse_args()

args = parse_args()
for key, value in vars(args).items():
    # need to eval the total_batch_size to get the grad_accum_steps
    if key == 'total_batch_size_str':
        value = eval(value)
        setattr(TrainingConfig, 'total_batch_size', value)
    else:
        if isinstance(value, str) and key !='non_linearity':
            value = value.lower().strip()
        if hasattr(TrainingConfig, key):
            setattr(TrainingConfig, key, value)
        else:
            setattr(ModelConfig, key, value)
if ModelConfig.attn == 'mha':
    ModelConfig.n_kv_heads = ModelConfig.n_head
elif ModelConfig.attn == 'mqa':
    ModelConfig.n_kv_heads = 1
elif ModelConfig.attn == 'mla':
    req = ModelConfig.q_latent_dim is not None and ModelConfig.kv_latent_dim is not None
    assert req, "Either q_latent_dim or kv_latent_dim is missing"
    if ModelConfig.pos_emb == 'rope':
        assert ModelConfig.rope_head_dim is not None, "Need dim of Rotary heads"

# _______________ DATASET _________________

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        # training data
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+(B*T+1)]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        # advance the position
        self.current_position += B*T 

        if self.current_position + (B*T*+1)  > len(self.tokens):
            self.current_position = 0
        return x,y

# ____________ UTIL FUNCTIONS _________________

def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters + 2 # avoid division by zero
    # 1) linear warump for warmup_steps:
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (max_decay_steps - TrainingConfig.warmup_steps)
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

#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)

#___________CREATE YOUR MODEL_____________
model = LLM(ModelConfig).to(device)
print(f"total parameters = {model.get_num_params():,}")

if TrainingConfig.compile :  
    print("Using compiled model")
    model = torch.compile(model)

#______________________________________________ TRAINING ______________________________________________

optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)
train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size)
eval_loader  = None

for iter in range(TrainingConfig.max_iters+1):
    t0 = time()

    lr = get_lr(iter, TrainingConfig) 
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)
    if TrainingConfig.eval:
        pass
        # every once in a while evaluate the loss on train and val sets
        # if iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters - 1:
        #     losses = estimate_loss()
        #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    for micro_step in range(grad_accum_steps):

        x,y = train_loader.next_batch()
        x,y = x.to(device=device), y.to(device=device)

        with ctx:
            _, loss, _ = model(x,y) #logits, loss, kv cache
            loss = loss/grad_accum_steps

        scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()    

    if "cuda" in device : torch.cuda.synchronize()
    dt  = (time()-t0)*1000
    print(f"step: {iter} | train loss:{loss*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps}")

if TrainingConfig.save_model:
    torch.save(model.state_dict(), 'llm_model.pt')
