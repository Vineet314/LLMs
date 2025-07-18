'''
This code explores the sinusoidal positional encodings, as introduced in Vaswani et. al.,
"Attention is all you need" : https://arxiv.org/pdf/1706.03762/v1

The previous models used learnable embeddings, but they increase computation are not very accurate.
Thus, fixed encodings are preffered.

It was decided to skip exploring integer and binary encodings, because sinusoidal encodigns are just better, 
and that is one step closer to RoPE.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMconfig:
    # hyperparameters
    block_size : int  # what is the maximum context length for predictions?
    vocab_size : int # OPTIM 4 (along with grad clipping) brought dt from 95 to 90
    n_kv_heads : int
    n_embd : int
    n_head : int
    n_layer: int
    dropout: float

class CausalSelfAttention(nn.Module):
    """ Grouped-Query Attention """

    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head  = config.n_head

        self.n_kv_heads = config.n_kv_heads
        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        head_size = self.n_embd // self.n_head

        # k,q,v in a btach
        # Total size for Q is n_embd. Total size for K and V is n_kv_heads * head_size each.
        self.c_attn = nn.Linear(self.n_embd, self.n_embd + 2 * self.n_kv_heads * head_size)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


    def forward(self, x, kv_cache=None):
        # input of size (batch, time-step, channels) or (Batch, tokens, embeddings)
        # output of size (batch, time-step, head size)
        B,T,C = x.size()
        head_size = C //self.n_head
        # Calculate Q, K, V by projecting input x
        q_proj_size = self.n_embd
        kv_proj_size = self.n_kv_heads * head_size
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # this was for MHA
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        q = q.view(B, T, self.n_head,     head_size).transpose(1, 2) # (B, n_kvh, T, hs)
        k = k.view(B, T, self.n_kv_heads, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_kv_heads, head_size).transpose(1, 2) # (B, n_kvh, T, hs)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        updated_kv_cache = (k, v)

        # Before attention, repeat K and V heads to match Q heads
        # basically copying K,V to perform MQA and GQA
        # i could set enable_gqa=True in F.scaled_dot_product_attention but i am using repeat_interleave instead, as per:
        # https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
        if self.n_kv_heads != self.n_head:
            num_repeats = self.n_head // self.n_kv_heads
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache
       
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

        pos_emb = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1) 

        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
        #                   |________[0,2,4...n_embd]________|         * |______log(10000^(-1/d_model))_____| = e^log(10000^(-2i/d_model)) = 1 / 10000^(2i/d_model) 
        # Calculate the sinusoidal encodings
        pos_emb[:, 0::2] = torch.sin(position * div_term) # Even indices
        pos_emb[:, 1::2] = torch.cos(position * div_term) # Odd indices

        self.register_buffer('pos_emb', pos_emb)

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tkn_emb.weight  = self.lm_head.weight

        self.apply(self._init_weights)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # Since sinusoidal embeddings are fixed and not parameters, no subtraction is needed.
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
        use_fused = True and ("cuda" in device)
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)
        except:
            use_fused = False
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        print(f"using fused AdamW: {use_fused}")

        if prints:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def forward(self, idx, targets=None, kv_caches=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Maximum context window is {self.block_size} but got length {t}"
        # Determine the starting position for positional encoding.
        # If there's a KV cache, the new tokens are being added after the cached sequence.
        start_pos = kv_caches[0][0].shape[-2] if kv_caches is not None else 0

        # Get token embeddings
        tkn_emb = self.tkn_emb(idx) # (B, T, n_embd)
        
        # Get the correct positional embeddings for the current time steps
        pos_emb = self.pos_emb[start_pos : start_pos + t, :] # (T, n_embd)
        
        x = self.transformer.drop(tkn_emb + pos_emb)

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
            # # --- After the forward pass, prune the cache if it's too long ---
            # if kv_caches[0][0].shape[-2] > self.block_size:
            #     for i in range(len(kv_caches)):
            #         k, v = kv_caches[i]
            #         # Slice the sequence dimension (T) to be at most block_size
            #         kv_caches[i] = (k[..., -self.block_size:, :], v[..., -self.block_size:, :])

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx