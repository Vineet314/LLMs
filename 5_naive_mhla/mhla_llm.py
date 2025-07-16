'''
This code implements the latest advancement in the transformer acrhitecture: The Multi Head Latent Attention. 
Introduced by DeepSeek in : https://arxiv.org/abs/2405.04434

This code builds a transformer based LLM which uses the 'Low-Rank Key-Value Joint Compression' MHLA algorithm as per the above paper.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class LLMconfig:
    # hyperparameters
    block_size : int  # what is the maximum context length for predictions?
    vocab_size : int 
    q_latent_dim : int
    kv_latent_dim : int
    n_embd : int
    n_head : int
    n_layer: int
    dropout: float

class CausalSelfAttention(nn.Module):
    """ A fully parallel implementation of the MHLA algorithm. No for loops. 
    Currently does not support RoPE encodings. Thus Naive MHLA."""
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.head_size = config.n_embd // config.n_head

        self.W_dq  = nn.Linear(config.n_embd,        config.q_latent_dim,  bias=False)
        self.W_uq  = nn.Linear(config.q_latent_dim,  config.n_embd,        bias=False)
        self.W_dkv = nn.Linear(config.n_embd,        config.kv_latent_dim, bias=False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_o   = nn.Linear(config.n_embd,       config.n_embd,         bias=False)
        # self.ln  = nn.LayerNorm(config.kv_latent_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Attributes to store pre-computed matrices for inference (now using register_buffer)
        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)
        # self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0))

    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh , n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, kv_cache=None) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, C = x.size()
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head

        # k_eff and v_eff based on training or inference
        if self.training:
            k_eff = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

        if kv_cache is None:
            c_kv = new_c_kv # (B,T,n_kvl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # append cache

        T_full = c_kv.size(1) # Current total sequence length (including cache)

        q:torch.Tensor = self.W_uq(self.W_dq(x)) # query projection : (B,T,C) -> (B,T,n_ql) -> (B,T,C)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B, nh, T, hs)

        # Attention score using the effective k_eff
        attn:torch.Tensor = (q @ k_eff @ c_kv.transpose(1,2).unsqueeze(1)) / sqrt(hs)

        # Causal masking adapted for KV caching during training or inference
        # query_global_indices: The global index of each query in the current input 'x'
        # key_global_indices: The global index of each key in the full cached sequence 'c_kv'
        # T is the length of the current input `x`. T_full is `T_prev_cache + T`.
        # For a new query at relative index `i` (0 to T-1), its global index is `(T_full - T) + i`.
        # It can attend to keys up to its global index.
        query_global_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
        key_global_indices   = torch.arange(T_full, device=x.device).unsqueeze(0)
        causal_mask_expanded = (query_global_indices >= key_global_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)

        attn = attn.masked_fill(causal_mask_expanded == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, n_kvl) * (1, nh, n_kvl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        return y, c_kv
       
class MLP(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config:LLMconfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config:LLMconfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

    def forward(self, x, kv_cache=None):
        attn_output, updated_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x, updated_kv_cache

class LLM(nn.Module):
    """ A simple GPT-like language model """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tkn_emb.weight  = self.lm_head.weight

        self.apply(self._init_weights)

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        except:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        if prints:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def forward(self, idx:torch.Tensor, targets=None, kv_caches:torch.Tensor=None):
        b,t = idx.size()
        # assert t<=self.block_size, f"Maximum context window is {self.block_size} but got length {t}"       

        if kv_caches is not None:
            # During generation with KV caching, 'idx' will contain only the new token(s).
            # The position starts from the length of the existing cache.
            # kv_caches[0][0].shape[-2] gives the current sequence length in the cache
            # The input idx has length 't', so positions should be [cache_len, cache_len + 1, ..., cache_len + t - 1]
            pos_offset = kv_caches[0][0].shape[-2] 
            pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=idx.device).unsqueeze(0)
        else:
            # Initial forward pass (no cache)
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        tkn_emb = self.tkn_emb(idx)
        pos_emb = self.pos_emb(pos % self.block_size)
        x = self.transformer.drop(tkn_emb+pos_emb)

        updated_kv_caches = [] # C_KV for each Block
        for i, block in enumerate(self.transformer.h):
            x, updated_kv_cache = block(x, kv_cache=kv_caches[i] if kv_caches is not None else None)
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
    def generate(self, idx:torch.Tensor, max_new_tokens:int, temperature=1.0, top_k=None):
        """
        Takes a conditioning sequence of indices idx (LongTensor of shape (b,t)) and
        completes the sequence max_new_tokens times, feeding the predictions back into the model.
        It manages the KV cache to ensure the context window (block_size) is respected.
        """
        kv_caches = None # Initialize
        initial_input_length = idx.size(1)
        
        if initial_input_length > self.block_size:
            # Use only the last block_size tokens from the prompt
            current_input_for_model = idx[:, -self.block_size:]
        else:
            current_input_for_model = idx
        
        # one forward pass to initialize KV caches.
        logits, _, kv_caches = self(current_input_for_model, kv_caches=None)

        if initial_input_length > self.block_size:
            idx = idx[:, -self.block_size:]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] 

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            
            # Ensure kv cache doesn't exceed block_size
            current_cache_length = kv_caches[0][0].shape[-2] 
            
            if current_cache_length > self.block_size:
                for i in range(len(kv_caches)):
                    kv_caches[i] = kv_caches[i][..., -self.block_size:, :]
                                    
            logits = logits[:, -1, :] / temperature # (B, vocab_size)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx