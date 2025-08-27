import torch
import triton
import triton.language as tl


@triton.jit
def kernel_softmax_online(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm_new = tl.maximum(mm, x)
        if i:
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new

    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new

    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_online(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y

