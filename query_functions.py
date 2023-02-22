import torch
from kernels import *
def variance(X1, X2, y1, noise, kernel, device, mean_only=False, variance_only=False):
    if kernel == dot_kernel:
        assert noise != 0
    S12 = kernel(X1, X2) # (N, M)
    S11 = kernel(X1, X1) # (N, N)
    Minv = S12.t() @ torch.inverse(S11 + noise**2 * torch.eye(X1.shape[0]).to(device)) # (N, N)
    mean = Minv @ y1
    if mean_only:
        return mean
    S22 = kernel(X2, X2) # (M, M)
    variance = S22 - Minv @ S12
    if variance_only:
        return variance
    return mean, variance