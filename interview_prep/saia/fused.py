# matmul + bias + activation (relu)

import torch

def naive_kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor):
    m, k = a.shape
    k, n = b.shape
    c = torch.empty(m,n, dtype=a.dtype, device=a.device)

    # matmul
    c = torch.matmul(a, b) #[m, n]

    # bias
    c = c + bias.reshape((m,1)) #[m, n] + [m, ] = [m, n]

    # activation
    c = torch.relu(c)

    return c
