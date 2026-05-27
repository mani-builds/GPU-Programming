import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Map program ID to row index
    row_idx = tl.program_id(0)

    # Pointers to the current row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x_ptr = X_ptr + row_idx * stride + cols
    y_ptr = Y_ptr + row_idx * stride + cols

    # Load input and compute variance
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store rstd for the backward pass
    tl.store(RSTD_ptr + row_idx, rstd)

    # Load weight, normalize, and scale
    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    y = x * rstd * w

    # Store output
    tl.store(y_ptr, y.to(Y_ptr.dtype.element_ty), mask=mask)

@triton.jit
def rmsnorm_bwd_kernel(
    DY_ptr, X_ptr, W_ptr, RSTD_ptr, DX_ptr, DW_ptr,
    stride_x, stride_dy,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to row index
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row data
    dy_ptr = DY_ptr + row_idx * stride_dy + cols
    x_ptr = X_ptr + row_idx * stride_x + cols

    dy = tl.load(dy_ptr, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_idx).to(tl.float32)

    # Compute gradient for weight (dw)
    # y = x * rstd * w -> dy/dw = x * rstd
    dw_row = dy * x * rstd
    tl.atomic_add(DW_ptr + cols, dw_row, mask=mask)

    # Compute gradient for input (dx)
    # dx = (dy * w * rstd) - (x * rstd^3 / N) * sum(dy * w * x)
    m_dy_w = dy * w
    sum_dy_w_x = tl.sum(m_dy_w * x, axis=0)

    dx = rstd * (m_dy_w - (x * rstd * rstd / N) * sum_dy_w_x)

    # Store dx
    dx_ptr = DX_ptr + row_idx * stride_x + cols
    tl.store(dx_ptr, dx.to(DX_ptr.dtype.element_ty), mask=mask)
