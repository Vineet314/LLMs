'''This script builds and trains an LLM model based on the user's CLI inputs. Available settings to choose from : 
1. Attention Type (with or without KV caching): 
    - Basic Multi Head Attention (basic)
    - Flash Multi Head Attention (mha)
    - Flash Attention based on Multi Query Attention (mqa)
    - Flash Attention based on Grouped Query Attention (gqa)
    - Multi Head Latent Attention (mla)
    - (Work in progress) Flash Multi Head Latent Attention (fmla)

2. Positional Encodings:
    - Learnable PE
    - Sinusoidal PE
    - Rotary PE (RoPE)

The following arguments can be provided. If not, the default values are set : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from dataclasses import dataclass 
from typing import Literal

@dataclass
class LLMconfig:
    # token params
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']

    # Neural Network
    up_dim  : int
    non_lin : str | Literal['relu', 'gelu', 'swish', 'mish', 'silu', 'selu','celu']
    dropout : float
    n_layer : int
    
    # Attention
    typ : str | Literal['basic','mha', 'mqa', 'gqa', 'mla', 'fmla']
    kv_cache : bool
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

        # common between all mechanisims
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if config.typ.lower().strip() in ('basic','mha','mqa','gqa'):
            # TODO make sure config.kv_heads is set during argparse
            self.n_kv_heads = config.n_kv_heads
            assert config.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
            self.head_size = config.n_embd // config.n_head
            self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.n_kv_heads * self.head_size, bias=False)

            if config.typ.lower().strip() == 'basic':
                self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        elif config.typ.lower().strip() == 'mla':
            self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim , False)
            self.W_uq  = nn.Linear(config.q_latent_dim , config.n_embd, False)
            self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
            self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
            self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
            self.W_o   = self.c_proj

            self.register_buffer('_k_absorbed_inference', None, persistent=False)
            self.register_buffer('_v_absorbed_inference', None, persistent=False)

            if config.pos_emb.lower().strip() == 'rope':
                self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim,  False)
                self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)
            
    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        assert self.config.typ == 'mla', "This is only for absorption trick in MLA"
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, nlkv, hs, nlq = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head, self.config.q_latent_dim
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)   


    




    
