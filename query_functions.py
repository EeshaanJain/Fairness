import torch

def variance(x_new, X_old, noise, kernel, device):
    K_no = kernel(x_new, X_old) #k_(x*,x)
    K_oo = kernel(X_old, X_old) # K
    return kernel(x_new, x_new) + noise ** 2 - K_no.t() @ torch.inverse(K_oo + noise**2 * torch.eye(K_oo.shape[0]).to(device))


def naive_variance(x_new, X, noise, kernel='dot', device='cpu'):
    assert kernel == 'dot'
    A = x_new@(X.t())
    K =  X@(X.t())
    K_plus_I_inv = torch.inverse(K+(noise**2)*torch.eye(K.shape[0]).to(device))
    return torch.sum(x_new * x_new) + (noise ** 2) - A@K_plus_I_inv@A.t() 