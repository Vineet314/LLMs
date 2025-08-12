import warnings
warnings.filterwarnings("ignore")
import os
import math
import json
import argparse
import torch
import numpy as np
from time import perf_counter
from dataclasses import dataclass

import deepspeed
from model import LLM

# _______________ Config Dataclasses _______________
@dataclass
class Trainconfig:
    dataset : str
    # total_batch_size : int
    # batch_size : int
    max_iters : int
    eval : bool
    eval_interval : int
    eval_iters : int
    learning_rate : float
    warmup_steps : int
    grad_clip : float
    compile : bool
    save_model : bool
    file_name : str

@dataclass
class LLMconfig:
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str
    up_dim : int
    non_linearity : str
    dropout : float
    n_layer : int
    moe : bool
    n_exp : int
    n_shared : int
    n_act : int
    coeff : float
    aux_free : bool
    alpha : float
    gamma : float
    attn : str
    n_head : int
    n_kv_heads : int
    q_latent_dim : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None

# Defaults
ModelConfig = LLMconfig(
    vocab_size = 50304,
    block_size = 1024,
    n_embd = 256,
    pos_emb = 'rope',
    up_dim = 384,
    non_linearity = 'swiglu',
    dropout = 0.0,
    n_layer = 6,
    moe = True,
    n_exp = 16,
    n_shared = 2,
    n_act = 8,
    coeff = 0.01,
    aux_free = True,
    alpha = 0.0001,
    gamma = 0.001,
    attn = 'mla',
    n_head = 8,
    n_kv_heads = 4,
    q_latent_dim = 32,
    kv_latent_dim = 32,
    rope_head_dim = 16,
)

TrainingConfig = Trainconfig(
    dataset = 'tinystories',
    # total_batch_size = 2048,
    # batch_size = 2,
    max_iters = 2500,
    eval = False,
    eval_interval = 100,
    eval_iters = 100,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0,
    compile = False if os.name != 'posix' else True,
    save_model = True,
    file_name = 'llm_model'
)

# _______________ CLI OVERRIDE _______________
def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    # Training Parameters
    parser.add_argument('--dataset',       type=str,   default=TrainingConfig.dataset,       help='The data set to be used for training')
    # parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--grad_clip',     type=float,  default=TrainingConfig.grad_clip,    help='Gradient Clip value')
    # Model Parameters
    parser.add_argument('--vocab_size',  type=int,   default=ModelConfig.vocab_size,  help='Vocabulary size for the model')
    parser.add_argument('--block_size',  type=int,   default=ModelConfig.block_size,  help='Block size for the model')
    parser.add_argument('--n_embd',      type=int,   default=ModelConfig.n_embd,      help='Embedding dimension for the model')
    parser.add_argument('--pos_emb',     type=str,   default=ModelConfig.pos_emb,     help='Type of positional encoding (learn, sin, rope)')
    parser.add_argument('--n_layer',     type=int,   default=ModelConfig.n_layer,     help='Number of layers in the model')
    parser.add_argument('--dropout',     type=float, default=ModelConfig.dropout,     help='Dropout rate for the model')
    # MLP Params
    parser.add_argument('--up_dim',      type=int,   default=ModelConfig.up_dim,      help='Up dimension for the Expert in the model')
    parser.add_argument('--non_linearity',type=str,   default=ModelConfig.non_linearity,help='Non-linearity for the Expert in the model')
    # MoE Params
    parser.add_argument('--n_exp',       type=int,   default=ModelConfig.n_exp,       help='Number of Experts in the model')
    parser.add_argument('--n_shared',    type=int,   default=ModelConfig.n_shared,    help='Number of Shared Experts in the model')
    parser.add_argument('--n_act',       type=int,   default=ModelConfig.n_act,       help='Number of Active Experts in the model')
    parser.add_argument('--coeff',       type=float, default=ModelConfig.coeff,       help='Aux Loss Coefficient for the MoE if not using Aux Free')
    parser.add_argument('--alpha',       type=float, default=ModelConfig.alpha,       help='Complementry Loss Coefficient for the MoE if using Aux Free')
    parser.add_argument('--gamma',       type=float, default=ModelConfig.gamma,       help='Bias Update speed in Aux loss free MoE if using Aux Free')
    # Attention Params
    parser.add_argument('--attn',        type=str,   default=ModelConfig.attn,        help='Type of attention mechanism (mha, mqa, gqa, mla)')
    parser.add_argument('--n_head',      type=int,   default=ModelConfig.n_head,      help='Number of attention heads in the model')
    parser.add_argument('--n_kv_heads',  type=int,   default=ModelConfig.n_kv_heads,  help='Number of KV heads in the model (only for gqa)')
    parser.add_argument('--q_latent_dim',  type=int, default=ModelConfig.q_latent_dim,help='Query latent dimension (only for mla)')
    parser.add_argument('--kv_latent_dim', type=int, default=ModelConfig.kv_latent_dim,help='KV latent dimension (only for mla)')
    parser.add_argument('--rope_head_dim', type=int, default=ModelConfig.rope_head_dim,help='RoPE head dimension (only for mla)')
    # parser.add_argument('--total_batch_size_str', type=str, default=str(TrainingConfig.total_batch_size), help='Total batch size for training passed in as a string expression')
    parser.add_argument('--moe',        action='store_true', help='Whether to use Mixture of Experts in the model')
    parser.add_argument('--aux_free',   action='store_true', help='Whether to use Aux Loss Free MoE')
    parser.add_argument('--eval',       action='store_true', help='Wheter to perform Evalutions once a while')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')
    parser.add_argument('--file_name', type=str, default=TrainingConfig.file_name, help='Name of the checkpoint to be saved')
    # DeepSpeed config path + overrides
    parser.add_argument('--ds_config', type=str, default="ds_config.json")
    parser.add_argument('--offload', action='store_true', help="Enable CPU offloading")

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

# _______________ Dataset Loader _______________
class DataLoader:
    def __init__(self, B, T, file_path, device):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        if self.B * self.T + 1 > self.N:
            raise ValueError("Batch size too large for dataset length")

    def next_batch(self):
        B, T = self.B, self.T
        start_indices = torch.randint(0, self.N - T - 1, (B,))
        x_list, y_list = [], []
        for start in start_indices:
            seq = self.tokens[start:start+T+1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])
        x = torch.tensor(np.stack(x_list), dtype=torch.long)
        y = torch.tensor(np.stack(y_list), dtype=torch.long)
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        return x, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(TrainingConfig.batch_size, ModelConfig.block_size, os.path.join('data', TrainingConfig.dataset, 'train.bin'), device)
val_loader   = DataLoader(TrainingConfig.batch_size, ModelConfig.block_size, os.path.join('data', TrainingConfig.dataset, 'val.bin'), device)

# _______________ LR Schedule _______________
def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters + 2
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    elif iter > max_decay_steps:
        return min_lr
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (max_decay_steps - TrainingConfig.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model_engine, TrainingConfig:Trainconfig, train_loader:DataLoader, val_loader:DataLoader, ds_config):
    out = {}
    model_engine.eval()
    model_engine.module.VAL_RUN = True  # Access the underlying nn.Module

    dtype = torch.bfloat16 if ds_config["bf16"]["enabled"] else torch.float16
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    ctx = torch.autocast(device_type=device_type, dtype=dtype)

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(TrainingConfig.eval_iters)
        for k in range(TrainingConfig.eval_iters):
            X, Y = loader.next_batch()
            with ctx:
                _, loss, _ = model_engine(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model_engine.train()
    model_engine.module.VAL_RUN = False
    return out

#  _______________ Load & Update DeepSpeed Config _______________
with open(args.ds_config) as f:
    ds_config = json.load(f)

if args.offload: 
    ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

with open(args.ds_config, "w") as f:
    json.dump(ds_config, f, indent=2)

# _______________ Model Init _______________
model = LLM(ModelConfig).to(device)
total, active = model.get_num_params()
print(f"total parameters = {total:,}, acitive parameters = {active:,}")
if TrainingConfig.compile:
    model = torch.compile(model)

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config=args.ds_config)

# _______________ Training Loop _______________
x, y = train_loader.next_batch()
train_loss_stats = []
valrun_val_loss_stats = []
valrun_train_loss_stats = []

for it in range(TrainingConfig.max_iters+1):
    t0 = perf_counter()
    
    lr = get_lr(it, TrainingConfig)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if ds_config["bf16"]["enabled"] else torch.float16):
        _, loss, _ = model_engine(x, y)

    x, y = train_loader.next_batch()
    model_engine.backward(loss)
    
    if TrainingConfig.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), TrainingConfig.grad_clip)
    
    model_engine.step()

    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader, ds_config)
        valrun_val_loss_stats.append(losses['val'])
        valrun_train_loss_stats.append(losses['train'])
        b = perf_counter()
        print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b

    if "cuda" in device : torch.cuda.synchronize()
    dt = (perf_counter() - t0) * 1000
    print(f"step {it} | loss: {loss.item():.4f} | dt: {dt:.2f} ms | accum: {model_engine.gradient_accumulation_steps()}")

# _______________ Save _______________
if TrainingConfig.save_model:
    model_engine.save_checkpoint(TrainingConfig.file_name + "_ckpt_dir")
    
    # save stats seperatley
    loss_stats = {'train':train_loss_stats, 'valrun_val':valrun_val_loss_stats, 'valrun_train':valrun_train_loss_stats}
    stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}
    torch.save(stats, TrainingConfig.file_name+'_stats.pt')
    print("Stats and config saved to {}.pt".format(TrainingConfig.file_name + '_stats'))
