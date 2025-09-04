import warnings; warnings.filterwarnings('ignore')
import os
import math
import torch
import argparse
import numpy as np

from pathlib import Path
from typing import Literal
from time import perf_counter, time
from dataclasses import dataclass, fields
from contextlib import nullcontext

from model import LLM, LLMconfig, BlockConfig

# ______________DEVICE and DTYPE SETUP_________________
torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision('medium')   # Not sure if this has any effect when used with Auto Mixed Precision

if torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.device_count() > 1: device="cuda:0"; import warnings; warnings.warn("You have MULTIPLE GPUs, but utilizing only ONE")
else: device = "cpu"

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ctx = torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype)) if device == 'cuda' else nullcontext()
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# ____________PARAMS-CONFIG_________________

@dataclass
class Trainconfig:
    dataset : str | Literal['shakespeare', 'tinystories', 'fineweb']
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
    ckpt_interval : int
    file_name : str
    act_recomp : bool
    wandb_log : bool
    wandb_project : str
    wandb_run_name : str

TrainingConfig = Trainconfig(
    dataset='tinystories',
    total_batch_size = 2**11,
    batch_size = 2**1, # how many independent sequences will we process in parallel?
    max_iters = 2500,
    eval = False,
    eval_interval=100,
    eval_iters=100,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0,    
    compile = False if os.name != 'posix' else True,
    save_model = True,
    ckpt_interval=250,
    file_name='llm_model',
    act_recomp=False,  # Default to False
    wandb_log = False, # Default to False
    wandb_project = 'llms',
    wandb_run_name = str(int(time())))

ModelConfig = LLMconfig(
    # token params
    vocab_size = 50304, 
    block_size = 2**10,
    n_embd = 256, 
    pos_emb = 'learn',
    dropout=0.0,
    n_layer = 6,
    norm = 'layer',
    
    act_recomp=TrainingConfig.act_recomp,   # Link the activation recomputation from the TRaining params           
    CUSTOM_LAYERS=False,                    # Whether to use custom layer configuration
    layer_configs=None)                     # List of layer configurations if using custom layers

# ___________ CLI-OVERRIDE__________________

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    # Training Parameters
    parser.add_argument('--dataset',       type=str,   default=TrainingConfig.dataset,       help='The data set to be used for training')
    parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--grad_clip',     type=float, default=TrainingConfig.grad_clip,     help='Gradient Clip value')
    parser.add_argument('--ckpt_interval', type=int,   default=TrainingConfig.ckpt_interval, help='Interval for checkpointing')
    parser.add_argument('--file_name',     type=str,   default=TrainingConfig.file_name,     help='Name of the checkpoint to be saved')
    parser.add_argument('--wandb_project', type=str,   default=TrainingConfig.wandb_project, help='Weights and Biases project name')
    parser.add_argument('--wandb_run_name',type=str,   default=TrainingConfig.wandb_run_name, help='Weights and Biases run name')
    parser.add_argument('--eval',         action='store_true', help='Wheter to perform Evalutions once a while')
    parser.add_argument('--act_recomp',   action='store_true', help='Whether to use (selective) activation recomputation')
    parser.add_argument('--save_model',   action='store_true', help='Whether to save the model after training')
    parser.add_argument('--wandb_log',    action='store_true', help='Whether to log training to Weights and Biases')

    # Model Parameters
    parser.add_argument('--vocab_size',  type=int,   default=ModelConfig.vocab_size,  help='Vocabulary size for the model')
    parser.add_argument('--block_size',  type=int,   default=ModelConfig.block_size,  help='Block size for the model')
    parser.add_argument('--n_embd',      type=int,   default=ModelConfig.n_embd,      help='Embedding dimension for the model')
    parser.add_argument('--pos_emb',     type=str,   default=ModelConfig.pos_emb,     help='Type of positional encoding (learn, sin, rope)')
    parser.add_argument('--n_layer',     type=int,   default=ModelConfig.n_layer,     help='Number of layers in the model')
    parser.add_argument('--dropout',     type=float, default=ModelConfig.dropout,     help='Dropout rate for the model')
    parser.add_argument('--norm',        type=str,   default=ModelConfig.norm,        help='Type of normalization (layer, rms)')
    # MLP Params
    parser.add_argument('--up_dim',      type=int,   default=512,    help='Up dimension for the Expert in the model')
    parser.add_argument('--non_linearity',type=str,  default='gelu', help='Non-linearity for the Expert in the model')
    # MoE Params
    parser.add_argument('--n_exp',       type=int,   default=None,   help='Number of Experts in the model')
    parser.add_argument('--n_shared',    type=int,   default=None,   help='Number of Shared Experts in the model')
    parser.add_argument('--n_act',       type=int,   default=None,   help='Number of Active Experts in the model')
    parser.add_argument('--coeff',       type=float, default=None,   help='Aux Loss Coefficient for the MoE if not using Aux Free')
    parser.add_argument('--alpha',       type=float, default=None,   help='Complementry Loss Coefficient for the MoE if using Aux Free')
    parser.add_argument('--gamma',       type=float, default=None,   help='Bias Update speed in Aux loss free MoE if using Aux Free')
    # Attention Params
    parser.add_argument('--attn',        type=str,   default='mha',  help='Type of attention mechanism (mha, mqa, gqa, mla)')
    parser.add_argument('--n_head',      type=int,   default=8,      help='Number of attention heads in the model')
    parser.add_argument('--n_kv_heads',  type=int,   default=8,      help='Number of KV heads in the model (only for gqa/mqa)')
    parser.add_argument('--q_latent_dim',  type=int, default=None,   help='Query latent dimension (only for mla)')
    parser.add_argument('--kv_latent_dim', type=int, default=None,   help='KV latent dimension (only for mla)')
    parser.add_argument('--rope_head_dim', type=int, default=None,   help='RoPE head dimension (only for mla)')
    
    parser.add_argument('--total_batch_size_str', type=str, default=str(TrainingConfig.total_batch_size), help='Total batch size for training passed in as a string expression')
    parser.add_argument('--moe',        action='store_true', help='Whether to use Mixture of Experts in the model')
    parser.add_argument('--aux_free',   action='store_true', help='Whether to use Aux Loss Free MoE')

    # Custom Layer configuration
    # dict_keys(['attn', 'n_head', 'moe', 'up_dim', 'non_linearity', 'n_kv_heads', 'q_latent_dim', 'kv_latent_dim', 'rope_head_dim', 'n_exp', 'n_shared', 'n_act', 'coeff', 'aux_free', 'alpha', 'gamma'])
    # OMG THIS SPINS MY HEAD
    # for simple LLMs with all layers having same params, pass-in normally.
    # if passing in custom layers, you still need to pass in `n_layer`. Other FFN/ATTN arguments will be inferred from the layer configs.
    # ONLY one of the following is to be provided if using custom layers configurations. 
    parser.add_argument('--layer_configs', nargs='+', default=None, help='List of layer configs. Refer to `parameters.md` for details.')
    parser.add_argument('--layer_configs_jsonl', type=str, default=None, help='Path for the JSON file containing the layer configurations')

    return parser.parse_args()

args = parse_args()

model_params    = [param.name for param in fields(LLMconfig)]
block_params    = [param.name for param in fields(BlockConfig)]; [block_params.remove(i) for i in ('n_embd', 'pos_emb', 'dropout')] #block_params.remove('n_embd'); block_params.remove('pos_emb'); block_params.remove('dropout')
training_params = [param.name for param in fields(Trainconfig)]

single_layer_config = {param:None for param in block_params} # for simple LLM
for key,val in vars(args).items():
    if   key == 'act_recomp' : setattr(ModelConfig, key, val); setattr(TrainingConfig, key, val)
    elif key == 'total_batch_size_str': value = eval(val); setattr(TrainingConfig, 'total_batch_size', value)
    
    elif key in training_params: setattr(TrainingConfig, key, val)
    elif key in model_params: setattr(ModelConfig, key, val)
    
    elif key == 'layer_configs' and val is not None:
        setattr(ModelConfig, 'CUSTOM_LAYERS', True)
        assert vars(args)['layer_configs_jsonl'] is None, "\nOnly one of `layer_configs` or `layer_configs_jsonl` can be passed-in."
        if isinstance(val, list) and all([isinstance(valoo, dict) for valoo in val]): 
            multi_layer_configs:list[dict] = val
        else: 
            raise TypeError(f"Expected a python list of dictionaries")
    
    elif key == 'layer_configs_jsonl' and val is not None:
        setattr(ModelConfig, 'CUSTOM_LAYERS', True)
        assert vars(args)['layer_configs'] is None, "\nOnly one of `layer_configs` or `layer_configs_jsonl` can be passed-in."
        import json; from pathlib import Path
        with open(Path(val), "r", encoding="utf-8") as f: multi_layer_configs = [json.loads(line) for line in f]
    
    else: # means block params are passed seperatley, intened for simple-LLM
        # Homogeneous model.
        setattr(ModelConfig, 'CUSTOM_LAYERS', False)
        if key in block_params: single_layer_config[key] = val

if not ModelConfig.CUSTOM_LAYERS: # for simple LLM
    print("Using simple homogeneous LLM")
    multi_layer_configs = [single_layer_config for _ in range(ModelConfig.n_layer)]
else:
    print("Using custom heterogeneous LLM")
    # `multi_layer_config` varible exists and is of the type list[dict]
    assert ModelConfig.n_layer == len(multi_layer_configs), f"\nNumber of layers ({ModelConfig.n_layer}) must match the length of layer_configs ({len(multi_layer_configs)})"

args = [ModelConfig.n_embd, ModelConfig.pos_emb, ModelConfig.dropout]
ModelConfig.layer_configs = [BlockConfig(*args, **kwargs) for kwargs in multi_layer_configs]

# run a few checks 
for l_num, layer in enumerate(ModelConfig.layer_configs):
    if layer.attn == 'mla':
        req = layer.q_latent_dim is not None and layer.kv_latent_dim is not None
        assert req, f"Either q_latent_dim or kv_latent_dim is missing in layer{l_num}"
        if ModelConfig.pos_emb == 'rope':
            assert layer.rope_head_dim is not None, f"Need dim of Rotary heads in layer{l_num}" 

# _______________ DATASET _________________

class DataLoader:
    def __init__(self, B, T, file_path, device):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'

        # Keep the memory-mapped file open persistently
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        if self.B * self.T + 1 > self.N:
            raise ValueError(f"Batch size {B} and block size {T} are too large for dataset of length {self.N}")

    def next_batch(self):
        """
        Returns (x, y) where:
        - x is (B, T) input tokens
        - y is (B, T) target tokens (shifted by one)
        """
        B, T = self.B, self.T

        # Sample B random starting positions independently
        start_indices = torch.randint(0, self.N - T - 1, (B,))

        # Gather sequences
        x_list = []
        y_list = []
        for start in start_indices:
            seq = self.tokens[start : start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])

        # Stack into tensors
        x = torch.tensor(np.stack(x_list), dtype=torch.long)
        y = torch.tensor(np.stack(y_list), dtype=torch.long)

        # Move to device (with pinned memory if CUDA)
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

data_dir = os.path.join('data', TrainingConfig.dataset)
print(f"Using Dataset {Path(data_dir).stem}")
train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path=os.path.join(data_dir, "train.bin"), device=device)
val_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path=os.path.join(data_dir, "val.bin"), device=device)

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
def estimate_loss(model:LLM, TrainingConfig:Trainconfig, train_loader:DataLoader, val_loader:DataLoader):
    out = {}
    model.eval() ; model.VAL_RUN = True
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(TrainingConfig.eval_iters)
        for k in range(TrainingConfig.eval_iters):
            X, Y = loader.next_batch()
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train(); model.VAL_RUN = False
    return out

#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)

#___________CREATE YOUR MODEL_____________
model = LLM(ModelConfig).to(device)
total, active = model.get_num_params()
print(f"total parameters = {total:,}, acitive parameters = {active:,}")

if ModelConfig.norm =='rms': print("Using RMSNorm")
if TrainingConfig.compile :  
    print("Using compiled model")
    model = torch.compile(model)

# ___________ WANDB INITIALIZATION __________________

if TrainingConfig.wandb_log:
    import wandb ; from copy import deepcopy

    dummy_config = deepcopy(ModelConfig)
    if dummy_config.CUSTOM_LAYERS:
        for attr in block_params: setattr(dummy_config, attr, 'custom')
    
    wandb_config = {'total':total, 'active':active, **vars(TrainingConfig), **vars(dummy_config)}

    wandb.init(project=TrainingConfig.wandb_project, 
               name=TrainingConfig.wandb_run_name,
               config=wandb_config)
    
    del wandb_config, dummy_config # free up RAM

#______________________________________________ TRAINING ______________________________________________

optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)
x,y = train_loader.next_batch() # get the first batch of training data
train_loss_stats, val_loss_stats = [], []
best_val_loss = float('inf') # ~1.797 * 10**308 ; float64 upper bound

for iter in range(TrainingConfig.max_iters+1):
    t0 = perf_counter() # timer stats

    # ____________ UPDATE LEARNING RATE ____________
    lr = get_lr(iter, TrainingConfig) 
    for param_grp in optimizer.param_groups: param_grp['lr'] = lr
    
    # ____________ ACTUAL TRAINING LOOP ____________
    # fwd pass -> loss -> scale loss if fp16 -> bwd pass -> clip grads -> step -> update sclaer 
    for micro_step in range(grad_accum_steps):
        with ctx: # auto-mixed precision
            _, loss, _ = model(x,y) #logits, loss, kv cache
            loss:torch.Tensor = loss/grad_accum_steps

        x,y = train_loader.next_batch() # Async prefetch the next batch of data
        train_loss_stats.append(loss.item())
        scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0: # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ____________ LOGGING ____________
    if "cuda" not in device: mem = 0
    else: torch.cuda.synchronize() ; mem = torch.cuda.memory_reserved()

    dt  = (perf_counter()-t0)*1000
    
    if iter!=0: # skip the logging for the first iteration
        print(f"step: {iter} | train loss:{loss.item()*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")
        
        # log to wandb
        if TrainingConfig.wandb_log:
            wandb.log({
                'train/loss': loss.item()*grad_accum_steps,
                'train/lr': lr,
                'train/grad_accum_steps': grad_accum_steps,
                'train/iter_time_ms': dt,
                'train/GPU_RAM_GB': mem/1024**3
            }, step=iter)

    # ____________ PERFORM EVAL RUN ____________
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        val_loss_stats.append(losses['val'])
        b = perf_counter()
        print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")

        # log to wandb
        if TrainingConfig.wandb_log:
            wandb.log({
                'eval/train_loss': losses['train'],
                'eval/val_loss': losses['val']
            }, step=iter)

    # ____________ SAVE LATEST CHECKPOINT ____________
    if TrainingConfig.save_model and (iter % TrainingConfig.ckpt_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        # model checkpoint
        loss_stats = {'iter':iter, 'train':train_loss_stats, 'val':val_loss_stats}

        checkpoint = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'model_state': model.state_dict()}
        stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}

        torch.save(checkpoint, TrainingConfig.file_name+'_ckpt.pt')
        torch.save(stats, TrainingConfig.file_name+'_stats.pt')
        
        del checkpoint, stats, loss_stats # del big variables from RAM
        print(f"---------------- At iter {iter} : Model Checkpoint and Stats saved")

    # ____________ SAVE BEST CHECKPOINT ____________
    if (TrainingConfig.save_model) and (iter>TrainingConfig.eval_interval) and (losses['val'] < best_val_loss):
        # best checkpoint, after atleast 1 val run
        best_val_loss = losses['val']
        loss_stats = {'iter':iter, 'train':train_loss_stats, 'val':val_loss_stats}

        checkpoint = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'model_state': model.state_dict()}
        stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}

        torch.save(checkpoint, TrainingConfig.file_name+'_best.pt')
        torch.save(stats, TrainingConfig.file_name+'_best_stats.pt')
        
        del checkpoint, stats, loss_stats # del big variables from RAM
        print(f"At iter {iter} : val loss {losses['val']:.4f} - New Best Model saved")
