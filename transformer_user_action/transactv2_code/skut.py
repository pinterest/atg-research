from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

"""
Triton kernel for Single Kernel Unified Transformer (SKUT) that fuses attention, layer norm, feed-forward network (FFN) and post FFN layer norm.
The kernel is designed to be used for fast inference of transformer models, that have some special constraints.
The Q, K, V weight matrices are small enough 64 dimensions to fit into SRAM. This allows to load a single sequence X and perform inplace tiled attention computation.
For more details refer to: https://arxiv.org/abs/2506.02267
"""

@triton.jit
def skut_fwd(
    X,  # [Z, H, N_CTX, D]
    Q,  # [D, D]
    K,  # [D, D]
    V,  # [D, D]
    W1,  # [D, D_FF]
    W2,  # [D_FF, D]
    kpm_ptr,  # [B, N_CTX]
    cm_ptr,  # [N_CTX, N_CTX]
    stride_kpm_0,
    stride_kpm_1,
    sm_scale,
    Out,  # [Z, H, N_CTX, D]
    stride_xz,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,  # Block size over Q sequence
    BLOCK_DMODEL: tl.constexpr,  # Block size over D (Should be equal to D)
    BLOCK_N: tl.constexpr,  # Block size over K, V sequence
    BLOCK_W0: tl.constexpr,  # Block size over W1 should be equal to D
    BLOCK_W1: tl.constexpr,  # Block size over W2 should be equal to D_FF
):
    """
    Triton kernel for fused attention, layer norm, feed-forward network (FFN) and post FFN layer norm.
    The kernel is designed to be used for fast inference of transformer models, that have some special constraints.
    The Q, K, V weight matrices are small enough -  64 dimensions to fit into SRAM. This allows to load a single sequence X and perform inplace tiled attention computation.
    The current kernel uses a model dimension of 64 and FF dimension of 32, for a batch size and seq length of 256 this works out to roughly 6 MB of memory usage, which is less than the 6 MB SRAM of A10G GPUs.
    For different model dimensions and GPU types, the block sizes may have to be modified to fit all inputs into SRAM.
    The kernel also fuses the layer norm and FFN operations to reduce the number of memory accesses and improve performance.
    Z - Batch size
    N_CTX - Sequence length
    D - Model dimension
    D_FF - Feed forward dimension

    Requirements:
        1. The Q, K, V weight matrices are small enough <64 dimensions to fit into SRAM.
        2. The tensors are of correct shape, dtype and contiguous.
        3. The kernel is designed to work with 1 head only.
        4. The sequence length N_CTX should be a multiple of 16 (for optimization)
        5. All other tensor dimensions are also a multiple of 16

    Args:
        X (torch.Tensor): Tensor of shape [Z, 1, N_CTX, D] representing the input sequence, currently only works for 1 head.
        Q (torch.Tensor): Tensor of shape [D, D] representing the query weight matrix.
        K (torch.Tensor): Tensor of shape [D, D] representing the key weight matrix.
        V (torch.Tensor): Tensor of shape [D, D] representing the value weight matrix.
        W1 (torch.Tensor): Tensor of shape [D, D_FF] representing the first weight matrix of the FFN.
        W2 (torch.Tensor): Tensor of shape [D_FF, D] representing the second weight matrix of the FFN.
        kpm_ptr (torch.Tensor): Tensor of shape [B, N_CTX] representing the key padding mask
        cm_ptr (torch.Tensor): Tensor of shape [N_CTX, N_CTX] representing the causal mask.
        stride_kpm_* (int): Stride for kpm_ptr.
        sm_scale (float32): Scale factor for the attention scores, should be 1/sqrt(D) for scaled dot product attention.
        Out (torch.Tensor): Tensor of shape [Z, 1, N_CTX, D] representing the output sequence.
        stride_x* (int): Strides for X
        stride_o* (int): Strides for Out
        BLOCK_M (tl.constexpr): Block size over the Q sequence.
        BLOCK_DMODEL (tl.constexpr): Block size over D (Should be equal to D).
        BLOCK_N (tl.constexpr): Block size over the K, V sequence.
        BLOCK_W0 (tl.constexpr): Block size over W1 (should be equal to D).
        BLOCK_W1 (tl.constexpr): Block size over W2 (should be equal to D_FF).
    """
    # Compiler hints
    N_CTX = tl.multiple_of(N_CTX, BLOCK_M)
    N_CTX = tl.multiple_of(N_CTX, BLOCK_N)

    # 2D grid of blocks
    # We use the first dimension to process the sequence length and the second dimension to process the batch size x heads
    # This reason for this slightly counter intuitive design is that the first dimension thread launch in cuda is faster than the second dimension and this potentially allows better memory locality - https://stackoverflow.com/questions/46660053/is-blockidx-correlated-to-the-order-of-block-execution
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Note: This kernel technically works with multiple heads but will use the same Q, K, V weight matrices for all heads so it's not useful for multi-head attention
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)
    qvk_offset = off_z * stride_xz + off_h * stride_xh

    # Setup block pointers, this allows easy pointer advances without complex pointer arithmetic
    kpm_kv_block_ptr = tl.make_block_ptr(
        base=kpm_ptr + off_z * stride_kpm_0,
        shape=(1, N_CTX),
        strides=(1, stride_kpm_1),
        offsets=(0, 0),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    cm_block_ptr = tl.make_block_ptr(
        base=cm_ptr,
        shape=(N_CTX, N_CTX),
        strides=(N_CTX, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # There are two block pointers for the X sequence, Xq advances over the Q sequence and Xkv advances over the K, V sequence
    Xq_block_ptr = tl.make_block_ptr(
        base=X + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_xm, stride_xk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    Xkv_block_ptr = tl.make_block_ptr(
        base=X + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_xm, stride_xk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    # Note: We load the whole Q, K, V, W1, W2 weight matrix into SRAM, this is possible because the Q, K, V, W1, W2 weight matrix is small enough to fit into SRAM
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    W1_block_ptr = tl.make_block_ptr(
        base=W1,
        shape=(BLOCK_W0, BLOCK_W1),
        strides=(BLOCK_W1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_W0, BLOCK_W1),
        order=(1, 0),
    )
    W2_block_ptr = tl.make_block_ptr(
        base=W2,
        shape=(BLOCK_W1, BLOCK_W0),
        strides=(BLOCK_W0, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_W1, BLOCK_W0),
        order=(1, 0),
    )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2) to account for using exp2 instead of exp
    # load q: it will stay in SRAM throughout
    q_mat = tl.load(Q_block_ptr)
    x0 = tl.load(Xq_block_ptr)
    q = tl.dot(x0, q_mat, allow_tf32=False, out_dtype=tl.float16)
    kt_mat = tl.load(K_block_ptr)
    v_mat = tl.load(V_block_ptr)
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    # Note: One caveat is that the KV^T part is calculated multiple times for each block of the Q sequence, this is a tradeoff to reduce memory usage
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kpm = tl.load(kpm_kv_block_ptr)
        cm = tl.load(cm_block_ptr)
        # Compute qk
        x = tl.load(Xkv_block_ptr)
        k = tl.trans(tl.dot(x, kt_mat, allow_tf32=False, out_dtype=tl.float16))
        v = tl.dot(x, v_mat, out_dtype=tl.float16)

        # Note: Using tf32 cores for dot leads to imprecise results
        # By forcing the out_dtype to be float16, we can use fp16 cores, which are more precise. The allow_tf32 flags are technically not needed, but are added for clarity.
        # The kpm and cm block are directly added to the qk matrix, it is important that they have -inf values in the correct places before invoking the kernel.
        qk = tl.dot(q, k, allow_tf32=False, out_dtype=tl.float16) + kpm + cm

        # This part computes the attention scores and updates the accumulator in the style of flash attention, see paper for more details
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # update m_i and l_i
        alpha = tl.math.exp2((m_i - m_ij))
        l_i = l_i * alpha + l_ij
        # update output accumulator
        acc = acc * alpha[:, None]
        # update acc
        acc += tl.dot(p.to(tl.float16), v, allow_tf32=False, out_dtype=tl.float16)
        # update m_i and l_i
        m_i = m_ij

        # Block pointer advance
        Xkv_block_ptr = tl.advance(Xkv_block_ptr, (BLOCK_N, 0))
        kpm_kv_block_ptr = tl.advance(kpm_kv_block_ptr, (0, BLOCK_N))
        cm_block_ptr = tl.advance(cm_block_ptr, (0, BLOCK_N))

    acc = acc / l_i[:, None]
    y = acc + x0

    # Layer norm
    y = y - tl.sum(y / BLOCK_DMODEL, axis=1)[:, None]
    y = y / tl.sqrt(tl.sum(y * y / BLOCK_DMODEL, axis=1) + 1e-5)[:, None]
    z = y

    # The FFN is fused as well
    w1 = tl.load(W1_block_ptr)
    w2 = tl.load(W2_block_ptr)
    y = tl.dot(y.to(w1.dtype), w1, allow_tf32=False, out_dtype=tl.float16)
    y = tl.where(y > 0, y, tl.zeros_like(y))
    y = tl.dot(y.to(w2.dtype), w2, allow_tf32=False, out_dtype=tl.float16)
    y = y + z

    # Post FFN Layer norm
    y = y - tl.sum(y / BLOCK_DMODEL, axis=1)[:, None]
    y = y / tl.sqrt(tl.sum(y * y / BLOCK_DMODEL, axis=1) + 1e-5)[:, None]

    tl.store(
        O_block_ptr,
        y.to(Out.type.element_ty),
    )


def run_skut(
    x,
    q,
    k,
    v,
    w1,
    w2,
    causal_mask,
    key_padding_mask,
    BLOCK_M=64,
    BLOCK_N=64,
    num_warps=4,
    num_stages=2,
):
    x = x.unsqueeze(1)
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    Lk = q.shape[-1]
    o = torch.empty_like(x).to(torch.float16)

    def kernel_grid(meta):
        return (triton.cdiv(x.shape[2], meta["BLOCK_M"]), x.shape[0] * x.shape[1], 1)

    skut_fwd[kernel_grid](
        X=x.to(torch.float16),
        Q=q.to(torch.float16),
        K=k.to(torch.float16),
        V=v.to(torch.float16),
        W1=w1.to(torch.float16),
        W2=w2.to(torch.float16),
        sm_scale=sm_scale,
        Out=o,
        kpm_ptr=key_padding_mask.to(torch.float16),
        cm_ptr=causal_mask.to(torch.float16),
        stride_kpm_0=key_padding_mask.stride(0),
        stride_kpm_1=key_padding_mask.stride(1),
        stride_xz=x.stride(0),
        stride_xh=x.stride(1),
        stride_xm=x.stride(2),
        stride_xk=x.stride(3),
        stride_oz=o.stride(0),
        stride_oh=o.stride(1),
        stride_om=o.stride(2),
        stride_on=o.stride(3),
        Z=x.shape[0],
        H=x.shape[1],
        N_CTX=x.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        BLOCK_W0=64,
        BLOCK_W1=32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o.squeeze(1)


class SKUTBlock(torch.nn.Module):
    """A Merged implememntation of triton and torch transformer block. This module uses a slow implementation of self-attention and feedforward for training and a fast fused triton implementation for inference.
    1. The implementation uses torch's scaled_dot_product_attention and layer_norm functions. The functional layer norm doesn't have a rescaling and bias parameter.
    2. The implementation uses torch's relu function for feedforward.
    3. The sequence length has to be a multiple of 16 for the fast implementation to work (triton block alignment optimization)
    4. The weights are stored in float32 and the computation is done in float16 for the fast implementation, they do not use the tf32 cores
    5. The input X, Q, K, W should not contain NaNs or Infs. The mask should use -inf and 0 for the padding and causal mask.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.q = torch.nn.Parameter(torch.empty(d_model, d_model, dtype=torch.float32))
        self.k = torch.nn.Parameter(torch.empty(d_model, d_model, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.empty(d_model, d_model, dtype=torch.float32))
        self.w1 = torch.nn.Parameter(torch.empty(d_model, d_ff, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.empty(d_ff, d_model, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q)
        torch.nn.init.xavier_uniform_(self.k)
        torch.nn.init.xavier_uniform_(self.v)
        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.w2)
    
    def _sa_slow(self, x, merged_mask):
        x = x + F.scaled_dot_product_attention(
            (x @ self.q).unsqueeze(1),
            (x @ self.k).unsqueeze(1),
            (x @ self.v).unsqueeze(1),
            merged_mask.unsqueeze(1),
        ).squeeze(1)
        x = torch.layer_norm(x, (x.size(-1),))
        return x
    
    def _ff_slow(self, x):
        x = x + F.relu(x @ self.w1, inplace=True) @ self.w2
        return torch.layer_norm(x, (x.size(-1),))

    def _sa_ff_fast(self, x, key_padding_mask, causal_mask):
        return run_skut(
            x.to(torch.float16),
            self.q.to(torch.float16),
            self.k.to(torch.float16),
            self.v.to(torch.float16),
            self.w1.to(torch.float16),
            self.w2.to(torch.float16),
            causal_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
    
    def forward(self, x, merged_mask=None, causal_mask=None, key_padding_mask=None, use_slow=True):
        if use_slow:
            z = self._sa_slow(x, merged_mask)
            z = self._ff_slow(z)
        else:
            z = self._sa_ff_fast(x, key_padding_mask, causal_mask)
        return z

class SKUTransformer(torch.nn.Module):
    def __init__(self, nlayer, d_model=64, d_ff=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([SKUTBlock(d_model, d_ff) for _ in range(nlayer)])

    def forward(self, x, key_padding_mask=None, causal_mask=None, use_slow=True):
        B, S, D = x.shape
        merged_mask = key_padding_mask.view(B, 1, S).expand(-1, S, -1) + causal_mask
        for layer in self.layers:
            x = layer(x, merged_mask=merged_mask if use_slow else None,
                      key_padding_mask=key_padding_mask if not use_slow else None,
                      causal_mask=causal_mask if not use_slow else None,
                      use_slow=use_slow)
        return x


def main():
    # Example usage
    nlayer = 2
    d_model = 64
    d_ff = 32
    model = SKUTransformer(nlayer, d_model, d_ff).to("cuda")

    B, S = 256, 256 
    x = torch.nn.init.xavier_uniform_(torch.empty((B, S, d_model), device="cuda")).contiguous()
    key_padding_mask = torch.where(torch.rand_like(x[..., 0]) < 0.25, -float("inf"), 0.0).contiguous()
    causal_mask = torch.where(torch.rand(S, S, device=x.device) < 0.25, -float("inf"), 0.0).contiguous()

    slow_output = model(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask, use_slow=True)
    fast_output = model(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask, use_slow=False)
    print("Device: ", torch.cuda.get_device_name())
    print("Outputs match Slow v/s Fast ?: ", torch.allclose(slow_output, fast_output.to(torch.float32), atol=1e-2, rtol=3e-2))

    print("Triton benchmark (ms): ", triton.testing.do_bench(lambda: model(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask, use_slow=False), warmup=5, rep=10))
    print("Slow benchmark (ms): ", triton.testing.do_bench(lambda: model(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask, use_slow=True), warmup=5, rep=10))

    """
    Expected Output:
    Device:  NVIDIA L40S
    Outputs match Slow v/s Fast ?:  True
    Triton benchmark (ms):  0.3244096040725708
    Slow benchmark (ms):  1.3330433368682861
    """


if __name__ == "__main__":
    main()