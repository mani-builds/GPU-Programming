#!/usr/bin/env python

import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    # 1. Map 1D program ID to 2D block position (pid_m, pid_n)
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_SIZE)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # 2. Compute 1D offset vectors
    off_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_k = tl.arange(0, BLOCK_SIZE)

    # 3. Initialize 2D pointers for A and B blocks
    a_ptrs = a_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn

    # 4. Initialize FP32 accumulator for numerical precision
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # 5. Iterate along K dimension
    for k_start in tl.range(0, k, BLOCK_SIZE):
        # Load A and B tiles with boundary masks along K dimension
        a_mask = (off_m[:, None] < m) & (off_k[None, :] < k - k_start)
        b_mask = (off_k[:, None] < k - k_start) & (off_n[None, :] < n)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate matrix multiplication results
        acc = tl.dot(a, b, acc)

        # Advance pointers along K dimension
        a_ptrs += BLOCK_SIZE * stride_ak
        b_ptrs += BLOCK_SIZE * stride_bk

    # 6. Convert to output precision (float16 / float32)
    c = acc.to(tl.float32)

    # 7. Write output block back to DRAM safely with masks
    c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    c_mask = (off_m[:, None] < m) & (off_n[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor):
    m, k = a.shape
    k_b, n = b.shape
    assert k == k_b, f"Incompatible dimensions: {a.shape} and {b.shape}"
    assert a.device == DEVICE and b.device == DEVICE

    output = torch.empty((m, n), device=DEVICE, dtype=a.dtype)

    BLOCK_SIZE = 64  # Square tile size (64x64)

    # Compute grid size: total 2D tiles required
    grid_m = triton.cdiv(m, BLOCK_SIZE)
    grid_n = triton.cdiv(n, BLOCK_SIZE)
    grid = (grid_m * grid_n, )

    matmul_kernel[grid](
        a, b, output,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

torch.manual_seed(0)
M, N, K = 1024, 4096, 2048

a = torch.randn([M, K], device=DEVICE, dtype=torch.float32)
b = torch.randn([K, N], device=DEVICE, dtype=torch.float32)

output_torch = torch.matmul(a, b)
output_triton = matmul(a, b)
print(output_torch)
print(output_triton)

# Check precision with typical GPU floating point tolerances
assert torch.allclose(output_triton, output_torch, atol=1e-1, rtol=1e-3)
print("SUCCESS: Triton output matches PyTorch!")
