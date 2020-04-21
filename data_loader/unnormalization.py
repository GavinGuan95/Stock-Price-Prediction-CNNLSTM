import numpy as np

def unnormalize(np_array, target=True):
    # apply the inverse transform in here
    # load the transformation params from the saved file
    fname = 'input_norm_para.npz'
    if target:
        fname = 'target_norm_para.npz'

    with np.load(fname) as para:
        mean, std = [para[i] for i in ('mean', 'std')]
    unnormalized_np = (np_array*std) + mean
    return unnormalized_np
