'''
This code implements the latest advancement in the transformer acrhitecture: The Multi Head Latent Attention. 
Introduced by DeepSeek in : https://arxiv.org/abs/2405.04434

This code builds a transformer based LLM which uses the 'Low-Rank Key-Value Joint Compression' MHLA algorithm as per the above paper.
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt
import inspect

class config:
    # hyperparameters
    batch_size : int # how many independent sequences will we process in parallel?
    block_size : int  # what is the maximum context length for predictions?
    vocab_size : int # OPTIM 4 (along with grad clipping) brought dt from 95 to 90

    max_iters : int
    eval_interval : int
    learning_rate : float
    warmup_steps : int
    max_decay_steps : int

    device : str
    eval_iters : int
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool

    latent_dim : int
    n_embd : int
    n_head : int
    n_layer : int
    dropout : float
    total_batch_size : int

class NaiveMHLA(nn.Module):
    """ A fully parallel implementation of the MHLA algorithm. No for loops. 
    Currently does not support RoPE encodings. Thus Naive MHLA."""
    def __init__(self, config:config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.head_size = config.n_embd // config.n_head
        # Projection layers
        self.W_dq  = nn.Linear(config.n_embd,     config.latent_dim, bias=False)  # Query down projection
        self.W_uq  = nn.Linear(config.latent_dim, config.n_embd,     bias=False)  # Query up projection
        self.W_dkv = nn.Linear(config.n_embd,     config.latent_dim, bias=False)  # Compress into latent KV space
        self.W_uk  = nn.Linear(config.latent_dim, config.n_embd,     bias=False)  # Decompress K
        self.W_uv  = nn.Linear(config.latent_dim, config.n_embd,     bias=False)  # Decompress V
        self.W_o   = nn.Linear(config.n_embd,     config.n_embd,     bias=False)  # Final output projection
        # self.ln  = nn.LayerNorm(config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.k_abs = None
        self.v_abs = None
        # self.register_buffer('k_abs', None)
        # self.register_buffer('v_abs', None)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x:torch.Tensor, kv_cache=None) -> torch.Tensor:

        B, T, C = x.size()
        nh , nl, hs = self.config.n_head, self.config.latent_dim, self.config.n_embd//self.config.n_head
        # if self.k_abs is None:
        k_absorbed = self.W_dq.weight.T @ self.W_uk.weight.T @ self.W_uk.weight # (C,nl) x (nl,C) x (C,nl) = (C,nl)
        self.k_abs = k_absorbed.view(nh, hs, nl).unsqueeze(0) # (1, nh, hs, nl)

        # if self.v_abs is None:
        v_absorbed = self.W_uv.weight.T @ self.W_o.weight.T   # (nl, C) x (C, C) = (nl, C)
        self.v_abs = v_absorbed.view(nl, nh, hs).transpose(0,1).unsqueeze(0) # (1, nh, nl, hs)
        
        new_c_kv = self.W_dkv(x)  # down projection : (B,T,C) -> (B,T,nl)
        if kv_cache is None:
            c_kv = new_c_kv # (B,T,nl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # append cache
        
        # Q*K^T = x * k_abs * c_kv^T   ### for variables, let q replace x, as q appears no where
        # x -> (B,T,C)
        q = x.view(B,T,nh,hs).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B, nh, T, hs)

        # now we have everything to compute attention scores, attn = q * k_abs * c_kv^T 
        # (B, nh, T, hs) * (1, nh, hs, nl) * (B, 1, nl, T) = (B, nh, T, T)
        # the following three steps can be made significatnly faster by avioding VRAM bottlenecks, perhaps by FlashMLA
        attn = (q @ self.k_abs @ c_kv.transpose(1,2).unsqueeze(1)) / sqrt(hs) # (B, nh, T, T) # significatnly faster than **-0.5
        attn = attn.masked_fill(self.tril[...,:T,:T] == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))  # (B, nh, T, T)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, nl) * (1, nh, nl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ self.v_abs  #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        return y, c_kv
       
class MLP(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config:config):
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

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.attn = NaiveMHLA(config)
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
    def __init__(self, config:config):
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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        if prints:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def forward(self, idx, targets=None, kv_caches=None):
        b,t = idx.size()
        assert t<=self.block_size, f"Maximum context window is {self.block_size} but got length {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        if kv_caches is not None:
            # During generation with KV caching, we are only passing the new token
            # so the position embedding should be for the current timestep
            pos = torch.tensor([[kv_caches[0][0].shape[-2]]], dtype=torch.long, device=idx.device)

        tkn_emb = self.tkn_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.transformer.drop(tkn_emb+pos_emb)

        updated_kv_caches = [] # list of tuples (k,v) for head of each Block
        for i, block in enumerate(self.transformer.h):
            x, updated_kv_cache = block(x, kv_cache=kv_caches[i] if kv_caches is not None else None)
            updated_kv_caches.append(updated_kv_cache)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss, updated_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        kv_caches = None
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop the KV cache
            if kv_caches is not None and kv_caches[0][0].shape[-2] >= self.block_size:
                for i in range(len(kv_caches)):
                    # Chop the cache to the block size
                    k, v = kv_caches[i]
                    # Slice sequence dimension T
                    kv_caches[i] = (k[..., -self.block_size+1:, :], v[..., -self.block_size+1:, :])
                # Pass only the last token
                idx_cond = idx[:, -1:]
            
            elif kv_caches is None:
                # First pass, no cache exists yet, crop the input if it's too long
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            else:
                # Subsequent passes, we have a cache, so only pass the last token
                idx_cond = idx[:, -1:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            # --- After the forward pass, prune the cache if it's too long ---
            if kv_caches[0][0].shape[-2] > self.block_size:
                for i in range(len(kv_caches)):
                    k, v = kv_caches[i]
                    # Slice the sequence dimension (T) to be at most block_size
                    kv_caches[i] = (k[..., -self.block_size:, :], v[..., -self.block_size:, :])

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx