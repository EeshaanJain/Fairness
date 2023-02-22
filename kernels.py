import torch

def dot_kernel(X1, X2):
    return X1 @ X2.t()

def rbf_kernel(X1, X2, var=1., scale=1.):
    val = -0.5 * torch.cdist(X1, X2) ** 2
    return scale * torch.exp(val / var)


