import torch
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def regression_binary_pred(output, target):
    with torch.no_grad():
        """
        This method needs to be fixed because because the sign of the normalized data does not
        necessarily match the sign of the un-normalized data
        """

        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()

        # apply the inverse transform in here
        # load the transformation params from the saved file
        with np.load('norm_para.npz') as para:
            mean, var = [para[i] for i in ('mean', 'var')]

        output_np = (output_np*var)+mean
        target_np = (target_np*var)+mean

        sign_match = np.sign(output_np) == np.sign(target_np)
        sign_match_percent = np.sum(sign_match)/np.size(sign_match)
    # print("regression_binary_pred: {}, {}, {}".format(sign_match_percent, output, target))
    return sign_match_percent
