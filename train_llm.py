'''This script builds and trains an LLM model based on the user's CLI inputs. Available settings to choose from : 
1. Attention Type (with  KV caching): 
    - Multi Head Attention (mha)
    - Multi Query Attention (mqa)
    - Grouped Query Attention (gqa)
    - Multi Head Latent Attention (mla)
    - (Work in progress) Flash Multi Head Latent Attention (fmla)

2. Positional Encodings:
    - Learnable PE
    - Sinusoidal PE
    - Rotary PE (RoPE)

This script uses Pytorch's Distributed Data Parallel, meaning the model can be trained on multi-GPU systems.
This can be either run using a bash script (recommended) or by using a terminal command as : 

torchrun --standalone --nproc_per_node=2 /path/to/this/scripty.py --arg1=val1 --arg2=val2
For details about arguments, see the LLMConfig and TrainConfig classes.

'''
### ----------- Model Script -----------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from dataclasses import dataclass 

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
    
    # Attention
    typ : str | Literal['mha', 'mqa', 'gqa', 'mla', 'fmla']
    # kv_cache : bool
    n_head : int
    n_kv_heads : int 
        # Only for mla 
    q_latent_dim  : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None

    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

        # H below is either the number of total query heads(nh)
        # hs is the embedding dimension for the query/key, given by n_embd//nh
        B,T,H,_ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)          # (B, T, H, hs)       -> (B, T, H, hs//2, 2)    -> creates the two pairs in the embd dim
        x_re, x_im = x_.unbind(-1)                      # (B, T, H, hs//2, 2) -> (B, T, H, hs//2)       -> splits those two pairs
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (T, hs//2)          -> (1, T, 1, hs//2)       -> this has dtype complex64, so last dim has two parts, real and imaginary
        # freq_cis has two parts : real and imaginary (cosθ, sinθ)
        
        # Perform the rotation (vector * rotation matrix)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)

class Attention(nn.Module):
    """ Implements the attention mechanism with different types of attention as specified in the config. """

    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = config.n_embd // config.n_head

        # common between all mechanisims
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if config.typ in ('mha','mqa','gqa'):
            # TODO make sure config.kv_heads is set during argparse
            self.n_kv_heads = config.n_kv_heads
            assert config.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
            self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.n_kv_heads * self.head_size, bias=False)

        elif config.typ == 'mla':
            self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim , False)
            self.W_uq  = nn.Linear(config.q_latent_dim , config.n_embd, False)
            self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
            self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
            self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
            self.W_o   = self.c_proj

            self.register_buffer('_k_absorbed_inference', None, persistent=False)
            self.register_buffer('_v_absorbed_inference', None, persistent=False)

            if config.pos_emb == 'rope':
                self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim,  False)
                self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)
            
    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        assert self.config.typ == 'mla', "This is only for absorption trick in MLA"
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, nlkv, hs, nlq = self.config.n_head, self.config.kv_latent_dim, self.head_size, self.config.q_latent_dim
        with torch.no_grad():
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)   
            
            if self.config.pos_emb == 'rope':
                self._k_absorbed_inference = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            else:
                self._k_absorbed_inference = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, nlkv).unsqueeze(0)

    def forward(self, x:torch.Tensor, freq_cis:torch.Tensor|None = None, kv_cache=None):
        B,T,C = x.size()
        hs = C//self.config.n_head
        assert self.head_size == hs, "Mismtach in head size (Should Never Happen)"
        rope = self.config.pos_emb == 'rope'

        if self.config.typ in ('mha','mqa', 'gqa'):
        # Calculate Q, K, V by projecting input x
            q_proj_size = self.config.n_embd
            kv_proj_size = self.n_kv_heads * hs
            # q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # this was for MHA
            q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
            q:torch.Tensor = q.view(B, T, self.config.n_head,hs) # (B, T, nh, hs)
            k:torch.Tensor = k.view(B, T, self.n_kv_heads,   hs) # (B, T, n_kvh, hs)
            v:torch.Tensor = v.view(B, T, self.n_kv_heads,   hs).transpose(1, 2) # (B, n_kvh, T, hs)

            if rope:
                q = LLMconfig.apply_rotary_emb(q, freq_cis)
                k = LLMconfig.apply_rotary_emb(k, freq_cis)

            q, k = q.transpose(1, 2), k.transpose(1, 2) # (B, nh, T, hs) , (B, n_kvh, T, hs)

            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)
            updated_kv_cache = (k, v)

            if self.n_kv_heads != self.config.n_head:
                num_repeats = self.config.n_head // self.n_kv_heads
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
            y = y.transpose(1,2).contiguous().view(B,T,C)
            # output projection
            y = self.dropout(self.c_proj(y))
        
        elif self.config.typ == 'mla':
            nh,nlkv,nlq = self.config.n_head, self.config.kv_latent_dim, self.config.q_latent_dim

            if self.training:
                if rope:
                    k_eff = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
                else:
                    k_eff = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, nlkv).unsqueeze(0)
                v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)
            else:
                if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                    self._precompute_absorbed_matrices()
                k_eff = self._k_absorbed_inference
                v_eff = self._v_absorbed_inference

            new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

            if kv_cache is None:
                c_kv  = new_c_kv
                if not rope: updated_kv_cache = c_kv
            else:
                if rope:
                    c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1) # append cache
                else:
                    c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # append cache
                    updated_kv_cache = c_kv

            T_full = c_kv.size(1)

            c_q:torch.Tensor = self.W_dq(x)  # (B,T,nlq)

            if not rope:
                q:torch.Tensor = self.W_uq(c_q)
                q = q.view(B, T, nh, hs).transpose(1, 2)
                attn:torch.Tensor = (q @ k_eff @ c_kv.transpose(1,2).unsqueeze(1)) / math.sqrt(hs)
            else:
                dhr = self.config.rope_head_dim
                attn_c = c_q.unsqueeze(1) @ k_eff @ c_kv.transpose(-1,-2).unsqueeze(1)
                c_kr:torch.Tensor = self.W_kr(x).unsqueeze(2)
                k_r = LLMconfig.apply_rotary_emb(c_kr, freq_cis).transpose(1,2)

                if kv_cache is not None:
                    k_r = torch.cat([kv_cache['k_r'], k_r], dim=2)
                
                updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}

                c_qr:torch.Tensor = self.W_qr(c_q).view(B,T,nh,dhr)
                q_r = LLMconfig.apply_rotary_emb(c_qr, freq_cis).transpose(1,2)

                attn_r = q_r @ k_r.transpose(-1,-2)
                attn:torch.Tensor = (attn_c + attn_r)/math.sqrt(hs+dhr)

            query_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
            key_indices = torch.arange(T_full, device=x.device).unsqueeze(0)
            mask = (query_indices >= key_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)
            attn = attn.masked_fill(mask == 0, float('-inf')) 

            attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)
            y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
            y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        return y, updated_kv_cache
    
class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig):
        super().__init__()
        non_linearity_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish(),
            'silu': nn.SiLU(),
            'selu': nn.SELU(),
            'celu': nn.CELU(),
            'elu': nn.ELU(),
            'lrelu': nn.LeakyReLU(negative_slope=0.01)}

        self.c_fc = nn.Linear(config.n_embd, config.up_dim*config.n_embd, bias=False)
        self.non_linearity = non_linearity_map.get(config.non_linearity, nn.GELU())
        self.c_proj = nn.Linear(config.up_dim*config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.non_linearity(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp  = MLP(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

    def forward(self, x:torch.Tensor, freq_cis:torch.Tensor|None = None, kv_cache=None):
        # Layer Norm + Attention
        attn, updated_kv_cache = self.attn.forward(self.ln1(x), freq_cis, kv_cache)
        x = x + attn
        # Feed-forward network with residual connection
        x = x + self.mlp(self.ln2(x))
        return x, updated_kv_cache

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config

        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb  = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            self.register_buffer("freq_cis", self._precompute_freqs_cis())
    
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)

    def _precompute_freqs_cis(self):
        """Precomputes the rotary frequencies for RoPE."""
        d = self.config.rope_head_dim
        assert d % 2 == 0, "rope_head_dim must be even"
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d)) # 1.0 / (base^(2i/d))
        seq = torch.arange(self.config.block_size)
        freqs = torch.outer(seq, theta)

        # Convert to complex numbers: r * e^(i*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
        
    def _init_weights(self, module):
        """Initializes model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Returns the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, device, prints=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
            print("Using Fused AdamW")
        except:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        if prints:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def forward(self, idx: torch.Tensor, targets=None, kv_caches=None):
        B, T = idx.size()
        start_pos = 0

        if kv_caches is not None and kv_caches[0] is not None:
            if self.config.typ in ('mha', 'mqa', 'gqa'):
                # cache is (key, value) tensors: (B, n_kv_heads, T_cache, hs)
                start_pos = kv_caches[0][0].shape[-2]
            elif self.config.typ == 'mla':
                if self.config.pos_emb == 'rope':
                    # Cache is {'c_kv': (B, T_cache, nlkv), 'k_r': (B, 1, T_cache, dhr)}
                    start_pos = kv_caches[0]['c_kv'].shape[1]
                else:
                    # Cache is the c_kv tensor: (B, T_cache, nlkv)
                    start_pos = kv_caches[0].shape[1]

        tkn_emb = self.tkn_emb(idx)  # Shape: (B, T, n_embd)
        
        freqs_cis = None
        
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
        pos = pos.clamp(max=self.config.block_size - 1)

        if self.config.pos_emb == 'rope':
            freqs_cis = self.freq_cis.to(idx.device)[pos]
            x = tkn_emb
        elif self.config.pos_emb == 'learn':
            pos_emb = self.pos_emb(pos)
            x = tkn_emb + pos_emb
        elif self.config.pos_emb == 'sin':
            x = tkn_emb + self.pos_emb.to(idx.device)[pos, :]

        x = self.transformer.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        updated_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x, updated_kv_cache = block(x, freqs_cis, kv_caches[i])
            updated_kv_caches.append(updated_kv_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # For generation, we only need logits of the last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()
        kv_caches = [None] * self.config.n_layer

        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                input_for_forward = idx_cond
            else:
                input_for_forward = idx[:, -1:]

            if kv_caches[0] is not None:
                if self.config.typ in ('mha', 'mqa', 'gqa'):
                    cache_len = kv_caches[0][0].shape[-2]
                elif self.config.typ == 'mla':
                     cache_len = kv_caches[0]['c_kv'].shape[1] if self.config.pos_emb == 'rope' else kv_caches[0].shape[1]

                if cache_len >= self.config.block_size:
                    # Keep the most recent (block_size - 1) tokens to make space for the new one
                    keep_len = self.config.block_size - 1
                    for layer_idx in range(self.config.n_layer):
                        layer_cache = kv_caches[layer_idx]
                        if self.config.typ in ('mha', 'mqa', 'gqa'):
                            k, v = layer_cache
                            kv_caches[layer_idx] = (k[..., -keep_len:, :], v[..., -keep_len:, :])
                        elif self.config.typ == 'mla':
                            if self.config.pos_emb == 'rope':
                                layer_cache['c_kv'] = layer_cache['c_kv'][:, -keep_len:, :]
                                layer_cache['k_r']  = layer_cache['k_r'][:, :, -keep_len:, :] # Seq len is dim 2
                            else: # c_kv
                                kv_caches[layer_idx] = layer_cache[:, -keep_len:, :]

            logits, _, kv_caches = self.forward(input_for_forward, kv_caches=kv_caches)
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
    
### ----------- Training Script -----------

import os
import argparse
import tiktoken
import requests

from time import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import IterableDataset, DataLoader

assert torch.cuda.is_available()

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
    max_decay_steps : int
    device : str
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool

ModelConfig = LLMconfig(
    # token params
    vocab_size = 50304, 
    block_size = 1024, 
    n_embd = 768, 
    pos_emb = 'rope',
    # FFN
    up_dim = 4, 
    non_linearity = 'gelu',  
    dropout=0.2,
    n_layer = 12, 
    # Attention
    typ = 'mla', 
    # kv_cache = True, 
    n_head = 8,
    n_kv_heads = 4, 
    # MHLA
    q_latent_dim = 256, 
    kv_latent_dim = 256,
    rope_head_dim = 64)                

TrainingConfig = Trainconfig(
    total_batch_size = 2**13,
    batch_size = 4, # how many independent sequences will we process in parallel?
    max_iters = 500,
    eval_interval = 50,
    learning_rate = 3e-4,
    warmup_steps = 25,
    max_decay_steps = 75,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    eval = False,
    eval_iters = 200,
    compile = False if os.name != 'posix' else True,
    save_model = True)

device_type = 'cuda'
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("Using BFloat16")
ctx = torch.autocast(device_type=device_type, dtype=dtype)
# ___________ CLI-OVERRIDE__________________

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--max_decay_steps',type=int,  default=TrainingConfig.max_decay_steps,help='Maximum decay steps for learning rate')
    parser.add_argument('--device',        type=str,   default=TrainingConfig.device,        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')

    parser.add_argument('--vocab_size',  type=int,   default=ModelConfig.vocab_size,  help='Vocabulary size for the model')
    parser.add_argument('--block_size',  type=int,   default=ModelConfig.block_size,  help='Block size for the model')
    parser.add_argument('--n_embd',      type=int,   default=ModelConfig.n_embd,      help='Embedding dimension for the model')
    parser.add_argument('--pos_emb',     type=str,   default=ModelConfig.pos_emb,     help='Type of positional encoding (learn, sin, rope)')
    parser.add_argument('--up_dim',      type=int,   default=ModelConfig.up_dim,      help='Up dimension for the MLP in the model')
    parser.add_argument('--non_linearity',type=str,   default=ModelConfig.non_linearity,help='Non-linearity for the MLP in the model')
    parser.add_argument('--dropout',     type=float, default=ModelConfig.dropout,     help='Dropout rate for the model')
    parser.add_argument('--n_layer',     type=int,   default=ModelConfig.n_layer,     help='Number of layers in the model')
    parser.add_argument('--typ',         type=str,   default=ModelConfig.typ,         help='Type of attention mechanism (mha, mqa, gqa, mla, fmla)')
    parser.add_argument('--n_head',      type=int,   default=ModelConfig.n_head,      help='Number of attention heads in the model')
    parser.add_argument('--n_kv_heads',  type=int,   default=ModelConfig.n_kv_heads,  help='Number of KV heads in the model (only for gqa)')
    parser.add_argument('--q_latent_dim',  type=int, default=None,                    help='Query latent dimension (only for mla)')
    parser.add_argument('--kv_latent_dim', type=int, default=None,                    help='KV latent dimension (only for mla)')
    parser.add_argument('--rope_head_dim', type=int, default=None,                    help='RoPE head dimension (only for mla)')
    
    parser.add_argument('--total_batch_size_str', type=str, default='2**16', help='Total batch size for training passed in as a string expression')
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

if ModelConfig.typ == 'mha':
    ModelConfig.n_kv_heads = ModelConfig.n_head
elif ModelConfig.typ == 'mqa':
    ModelConfig.n_kv_heads = 1
elif ModelConfig.typ == 'mla':
    req = ModelConfig.q_latent_dim is not None and ModelConfig.kv_latent_dim is not None
    assert req, "Either q_latent_dim or kv_latent_dim is missing"
    if ModelConfig.pos_emb == 'rope':
        assert ModelConfig.rope_head_dim is not None, "Need dim of Rotary heads"

# _______________ DATASET _________________

class MyDataset(IterableDataset):
    def __init__(self, tokens, B, T):
        super().__init__()
        self.tokens = tokens
        self.B = B
        self.T = T

    def __iter__(self):
        # Get the rank and world size to ensure each GPU gets a unique shard of data
        worker_info = torch.utils.data.get_worker_info()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Create a shard of the data for the current process
        num_tokens = len(self.tokens)
        tokens_per_process = num_tokens // world_size
        start = rank * tokens_per_process
        end = start + tokens_per_process

        # This is a simplified sharding. More robust methods exist.
        data = self.tokens[start:end]

        while True: # Loop indefinitely for epochs
            # Generate random starting points for batches
            ix = torch.randint(0, len(data) - self.T, (self.B,))
            x = torch.stack([data[i:i+self.T] for i in ix])
            y = torch.stack([data[i+1:i+self.T+1] for i in ix])
            yield x, y

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
enc = tiktoken.get_encoding('gpt2')
all_tokens = torch.tensor(enc.encode(text), dtype=torch.long)
train_dataset = MyDataset(all_tokens, TrainingConfig.batch_size, ModelConfig.block_size)
# eval_dataset  = MyDataset(all_tokens, TrainingConfig.batch_size, ModelConfig.block_size)

# ____________ UTIL FUNCTIONS _________________

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

# _______________DDP setup_________________

init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = ddp_rank == 0

#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size    # sequence length 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

#___________CREATE YOUR MODEL_____________
model = LLM(ModelConfig).to(device)
print(f"total parameters = {model.get_num_params():,}")

if TrainingConfig.compile : model = torch.compile(model)

model = DDP(model, device_ids=[ddp_local_rank])
raw_model:LLM = model.module

#______________________________________________ TRAINING ______________________________________________

optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=3e-4,device=device,prints=False)
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, pin_memory=True)
# eval_loader = DataLoader(eval_dataset, batch_size=None, num_workers=2, pin_memory=True) 
train_iter = iter(train_loader)

scaler = torch.amp.GradScaler(enabled=(device_type == 'cuda'))

for iter in range(TrainingConfig.max_iters):
    t0 = time()
    lr = get_lr(iter, TrainingConfig) 
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    if TrainingConfig.eval:
        pass
        # every once in a while evaluate the loss on train and val sets
        # if iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters - 1:
        #     losses = estimate_loss()
        #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    loss_accum = torch.tensor(0.0, device=device)

    for micro_step in range(grad_accum_steps):
        x,y = next(train_iter)
        x,y = x.to(device=device), y.to(device=device)
        # sync gradients only on the last micro step
        # this is to avoid unnecessary synchronization overhead as we are accumulating gradients
        # this is a hack to avoid the DDP warning about gradient synchronization
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # evaluate the loss
        with ctx:
            _, loss, _ = model(x,y)
            loss = loss/grad_accum_steps

        loss_accum += loss.detach()  
        scaler.scale(loss).backward()

    dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
    # grad clipping
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()    
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time()-t0)*1000
    if master_process : print(f"step: {iter} | train loss:{loss_accum.item():.4f} | dt: {dt:.2f}ms")

destroy_process_group()
if TrainingConfig.save_model and master_process:
    torch.save(model.state_dict(), 'ddp_model.pt')
    torch.save(raw_model.state_dict(), 'llm_model.pt')