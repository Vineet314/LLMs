import torch
import triton
import triton.language as tl

@triton.jit
def FullMHLA_kernel(
    # Data Pointers
    C_Q_ptr, Q_R_ptr, C_KV_ptr, K_R_ptr, K_EFF_ptr, V_EFF_ptr, Out_ptr,
    
    # Dimension Constants
    T, T_full, N_EMBD, Q_LATENT_DIM, KV_LATENT_DIM, HEAD_SIZE, ROPE_HEAD_DIM,
    
    # Stride Constants for Tensor Traversal
    stride_cq_b, stride_cq_t, stride_cq_nlq,
    stride_qr_b, stride_qr_h, stride_qr_t, stride_qr_dhr,
    stride_ckv_b, stride_ckv_t, stride_ckv_nlkv,
    stride_kr_b, stride_kr_h, stride_kr_t, stride_kr_dhr,
    stride_keff_h, stride_keff_nlq, stride_keff_nlkv,
    stride_veff_h, stride_veff_nlkv, stride_veff_hs,
    stride_out_b, stride_out_h, stride_out_t, stride_out_hs,
    
    # Other Parameters
    q_start_pos,  # The starting position of queries in the full sequence for causal masking

    # Meta-parameters for tuning
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr) :

    """
    Triton Kernel for DeepSeek-V2's Multi-Head Latent Attention.
    Computes attention based on two components:
    1. NoPE (No Position Embedding) attention via latent compression.
    2. RoPE (Rotary Position Embedding) attention on a low-rank projection.
    """
    # ## 1. Grid and Offset Calculation ##
    # Get the program IDs for the batch, head, and query block dimensions.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    # Calculate offsets for the current block of queries.
    t_offsets = pid_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T

    # ## 2. Pointer Setup and Initial Loads ##
    # Pointers to the current query block's data
    cq_block_ptr = C_Q_ptr + pid_b * stride_cq_b + t_offsets[:, None] * stride_cq_t + tl.arange(0, Q_LATENT_DIM)[None, :]
    qr_block_ptr = Q_R_ptr + pid_b * stride_qr_b + pid_h * stride_qr_h + t_offsets[:, None] * stride_qr_t + tl.arange(0, ROPE_HEAD_DIM)[None, :]
    
    # Pointers to the head-specific "effective" weight matrices
    keff_head_ptr = K_EFF_ptr + pid_h * stride_keff_h + tl.arange(0, Q_LATENT_DIM)[:, None] * stride_keff_nlq + tl.arange(0, KV_LATENT_DIM)[None, :]
    veff_head_ptr = V_EFF_ptr + pid_h * stride_veff_h + tl.arange(0, KV_LATENT_DIM)[:, None] * stride_veff_nlkv + tl.arange(0, HEAD_SIZE)[None, :]

    # ## 3. Initialization of Accumulators ##
    # Initialize accumulators for the online softmax algorithm.
    acc = tl.zeros([BLOCK_T, HEAD_SIZE], dtype=tl.float32)
    m_i = tl.full([BLOCK_T], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Load the necessary query and effective K-matrix data for this block.
    c_q_block = tl.load(cq_block_ptr, mask=t_mask[:, None], other=0.0)
    q_r_block = tl.load(qr_block_ptr, mask=t_mask[:, None], other=0.0)
    k_eff_head = tl.load(keff_head_ptr)
    
    # Pre-calculate the NoPE component of the query.
    q_eff_block = tl.dot(c_q_block, k_eff_head)

    # Define the attention scaling factor.
    scale = (HEAD_SIZE + ROPE_HEAD_DIM)**-0.5

    # ## 4. Main Loop over Key/Value Sequence ##
    # Iterate over the key/value sequence in blocks.
    for start_k in range(0, T_full, BLOCK_K):
        k_offsets = start_k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < T_full

        # Load Key-related data for the current block.
        ckv_block_ptr = C_KV_ptr + pid_b * stride_ckv_b + k_offsets[None, :] * stride_ckv_t + tl.arange(0, KV_LATENT_DIM)[:, None]
        # K_R is broadcast across the head dimension.
        kr_block_ptr = K_R_ptr + pid_b * stride_kr_b + 0 * stride_kr_h + k_offsets[None, :] * stride_kr_t + tl.arange(0, ROPE_HEAD_DIM)[:, None]
        
        c_kv_block = tl.load(ckv_block_ptr, mask=k_mask[None, :], other=0.0)
        k_r_block = tl.load(kr_block_ptr, mask=k_mask[None, :], other=0.0)

        # --- Compute Attention Scores ---
        # RoPE component
        s_r = tl.dot(q_r_block, k_r_block)
        # NoPE component
        s_c = tl.dot(q_eff_block, c_kv_block)
        
        s_ij = (s_r + s_c) * scale

        # --- Causal Masking ---
        causal_mask = (q_start_pos + t_offsets[:, None]) >= k_offsets[None, :]
        s_ij = tl.where(causal_mask, s_ij, -float('inf'))

        # --- Online Softmax Update ---
        m_ij = tl.maximum(m_i, tl.max(s_ij, 1))
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_ij = tl.sum(p_ij, 1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        # --- Update Accumulator with Value ---
        v_eff_head = tl.load(veff_head_ptr)
        v_block_eff = tl.dot(tl.trans(c_kv_block), v_eff_head)

        p_ij = p_ij.to(v_block_eff.dtype)
        acc += tl.dot(p_ij, v_block_eff)

        l_i += l_ij
        m_i = m_ij

    # ## 5. Finalization and Store ##
    # Normalize the accumulator. Add epsilon for stability.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Pointer to the output block
    out_block_ptr = Out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + t_offsets[:, None] * stride_out_t + tl.arange(0, HEAD_SIZE)[None, :]
    tl.store(out_block_ptr, acc.to(Out_ptr.dtype.element_ty), mask=t_mask[:, None])

def FullMHLA_triton(module: torch.nn.Module, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache=None):
    """
    Entrypoint function for the Triton-based FullMHLA forward pass.
    
    This function orchestrates the computation by:
    1. Performing initial linear projections on the input `x`.
    2. Managing the Key-Value (KV) cache for `c_kv` and `k_r`.
    3. Applying Rotary Position Embeddings (RoPE).
    4. Launching the main Triton kernel `FullMHLA_kernel`.
    5. Reshaping the output to the final desired format.
    """
    # ## 1. Setup and Projections ##
    B, T, C = x.shape
    cfg = module.config
    nh, nlkv, nlq = cfg.n_head, cfg.kv_latent_dim, cfg.q_latent_dim
    hs = C // nh
    dhr = cfg.rope_head_dim

    c_q = module.W_dq(x)
    new_c_kv = module.W_dkv(x)
    
    # ## 2. KV Cache Management ##
    if kv_cache is None:
        c_kv = new_c_kv
    else:
        c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1)
    T_full = c_kv.size(1)

    # ## 3. RoPE Application ##
    # The original static method from your LLMconfig is used here.
    c_qr = module.W_qr(c_q).view(B, T, nh, dhr)
    q_r = apply_rotary_emb(c_qr, freqs_cis).transpose(1, 2)
    
    c_kr = module.W_kr(x).unsqueeze(2)
    k_r_new = apply_rotary_emb(c_kr, freqs_cis).transpose(1, 2)
    
    # Append new rotary keys to the cache.
    if kv_cache is not None:
        k_r = torch.cat([kv_cache['k_r'], k_r_new], dim=2)
    else:
        k_r = k_r_new

    # ## 4. Prepare "Effective" Matrices and Output Tensor ##
    # These matrices absorb subsequent weight transformations.
    # Note: Using .training attribute to decide which branch to take, as in original code
    if module.training:
        k_eff = (module.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ module.W_uk.weight.view(1,nh,hs,nlkv))
        v_eff = (module.W_uv.weight.T @ module.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)  
    else:
        # In inference, we would ideally use pre-computed matrices.
        # This implementation computes them on-the-fly for simplicity, matching the training path.
        k_eff = (module.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ module.W_uk.weight.view(1,nh,hs,nlkv))
        v_eff = (module.W_uv.weight.T @ module.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)
    
    k_eff = k_eff.squeeze(0) # Shape: (nh, nlq, nlkv)
    v_eff = v_eff.squeeze(0) # Shape: (nh, nlkv, hs)
    
    y = torch.empty((B, nh, T, hs), device=x.device, dtype=x.dtype)

    # ## 5. Grid Definition and Kernel Launch ##
    # Define block sizes. These should be tuned for optimal performance.
    BLOCK_T = 64
    BLOCK_K = 64
    
    grid = (B, nh, triton.cdiv(T, BLOCK_T))
    
    FullMHLA_kernel[grid](
        c_q, q_r, c_kv, k_r, k_eff, v_eff, y,
        T, T_full, C, nlq, nlkv, hs, dhr,
        c_q.stride(0), c_q.stride(1), c_q.stride(2),
        q_r.stride(0), q_r.stride(1), q_r.stride(2), q_r.stride(3),
        c_kv.stride(0), c_kv.stride(1), c_kv.stride(2),
        k_r.stride(0), k_r.stride(1), k_r.stride(2), k_r.stride(3),
        k_eff.stride(0), k_eff.stride(1), k_eff.stride(2),
        v_eff.stride(0), v_eff.stride(1), v_eff.stride(2),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        T_full - T,  # q_start_pos
        BLOCK_T=BLOCK_T,
        BLOCK_K=BLOCK_K,
    )
    
    # ## 6. Final Output Processing ##
    # Reshape and apply dropout to match the original module's output.
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = module.dropout(y)
    
    updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}

    return y, updated_kv_cache

def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
    ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

    # H below is either the number of total query heads(nh)
    # hs is the embedding dimension for the query/key, given by n_embd//nh
    B,T,H,_ = x.size()
    x_ = x.float().reshape(B, T, H, -1, 2)          # (B, T, H, hs)       -> (B, T, H, hs//2, 2)    -> creates the two pairs in the embd dim
    x_re, x_im = x_.unbind(-1)                      # (B, T, H, hs//2, 2) -> (B, T, H, hs//2)       -> splits those two pairs
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (T, hs//2)          -> (1, T, 1, hs//2)       -> this has dtype complex64, so last dim has two parts, real and imaginary
    # freqs_cis has two parts : real and imaginary (cosθ, sinθ)
    # import code ; code.interact(local=locals())
    # Perform the rotation (vector * rotation matrix)
    x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
    x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
    
    # Stack the real and imaginary parts back together
    x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

    return x_out.type_as(x)