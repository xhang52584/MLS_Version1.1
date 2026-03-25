"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

# ============================================================================
# Triton Kernel for FlashAttention
# ============================================================================

@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr,
    scale,
    seq_q,seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, # Query 分块
    BLOCK_N: tl.constexpr, # Key/Value 分块
    BLOCK_D: tl.constexpr, # Head Dim 分块 (可选，默认为 64)
):
    # 获取当前的 Batch*Head 索引和 Q 的分块索引
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # 定义偏移量
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)

    # 指针基地址
    q_base = q_ptr + pid_bh * stride_q0 + rm[:, None] * stride_q1 + rd[None, :] * stride_q2
    k_base = k_ptr + pid_bh * stride_k0 + rn[None, :] * stride_k1 + rd[:, None] * stride_k2
    v_base = v_ptr + pid_bh * stride_v0 + rn[:, None] * stride_v1 + rd[None, :] * stride_v2

    # --- Online Softmax 状态变量 (数学优化关键) ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 最大值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # 累加和
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)      # 结果累加器

# 加载 Q 块 (保持在 SRAM 中)
    q = tl.load(
        q_base,
        mask=(rm[:, None] < q_ptr.shape[1]) & (rd[None, :] < head_dim),
        other=0.0,
)

    # 循环遍历 K/Value 的分块
    for start_n in range(0, seq_k, BLOCK_N):
        # 1. 加载 K 块 (注意转置以配合 tl.dot)
        k_ptrs = k_ptr + pid_bh * stride_k0 + (start_n + rn[None, :]) * stride_k1 + rd[:, None] * stride_k2
        k = tl.load(k_ptrs) # 形状 [head_dim, BLOCK_N]
        
        # 2. 计算分块 Scores
        qk = tl.dot(q, k) * scale
        
        # 3. 因果掩码
        if IS_CAUSAL:
            qk += tl.where(rm[:, None] >= (start_n + rn[None, :]), 0, -1.0e9)

        # 4. Online Softmax 核心更新
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # 更新归一化分母
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # 5. 更新累加器 (确保 V 的加载和乘法正确)
        v_ptrs = v_ptr + pid_bh * stride_v0 + (start_n + rn[:, None]) * stride_v1 + rd[None, :] * stride_v2
        v = tl.load(
            v_ptrs,
            mask=((start_n + rn[:, None]) < seq_k) & (rd[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc) # 使用 Tensor Core

        # 6. 更新最大值状态
        m_i = m_i_new

    # 最终归一化并存储结果
    acc = acc / l_i[:, None]
    out_base = output_ptr + pid_bh * stride_o0 + rm[:, None] * stride_o1 + rd[None, :] * stride_o2
    tl.store(
    out_base,
    acc,
    mask=(rm[:, None] < seq_q) & (rd[None, :] < head_dim),
    )


# ============================================================================
# Attention 
# ============================================================================

class MultiHeadAttention:
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape
        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)
        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1

def scaled_dot_product_attention(q, k, v, attention_mask=None, is_causal=False, scale=None) -> torch.Tensor:
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape
    if scale is None: scale = 1.0 / np.sqrt(head_dim)

    # 保持原有的 Padding 逻辑
    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    # 数据展平与转换
    q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
    k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
    v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()

    # FlashAttention 不再需要分配巨大的 scores 显存！(效率提升点)
    output = torch.empty((batch * num_heads, seq_q, head_dim), dtype=torch.float32, device=q.device)

    # 定义分块大小 (FlashAttention 的核心参数)
    BLOCK_M = 64
    BLOCK_N = 32

    # Grid 设置: 处理所有 Head，并行处理 Q 的不同分块
    grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_M))

    flash_attn_kernel[grid](
        q_flat, k_flat, v_flat, output,
        float(scale), seq_q, seq_k, head_dim,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=head_dim_padded
        num_warps=4,
        num_stages=2
    )

    return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(2, 4, 128, 64, device=device)
    k = torch.randn(2, 4, 128, 64, device=device)
    v = torch.randn(2, 4, 128, 64, device=device)
    output = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"Output shape: {output.shape} | FlashAttention optimized.")
