import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mean_quad_error_loss(output, target):
    return ((output - target) ** 4).sum() / output.data.nelement()