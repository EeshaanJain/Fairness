import torch

def dot_kernel(xa, xb):
    return xa @ xb.t()

def rbf_kernel(xa, xb, sigma=1.):
    return torch.exp((-0.5/sigma**2) * torch.cdist(xa, xb, p=2.0)**2)


