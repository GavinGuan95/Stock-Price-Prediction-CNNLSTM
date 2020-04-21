import torch
import numpy as np
from data_loader.unnormalization import unnormalize

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
        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()

        # apply the inverse transform in here
        output_np = unnormalize(output_np)
        target_np = unnormalize(target_np)

        output_positive_pct = np.sum(np.sign(output_np)) / np.size(output_np)
        target_positive_pct = np.sum(np.sign(target_np)) / np.size(target_np)
        # print("output positive pct: {}".format(output_positive_pct))
        # print("target positive pct: {}".format(target_positive_pct))

        sign_match = np.sign(output_np) == np.sign(target_np)
        sign_match_percent = np.sum(sign_match)/np.size(sign_match)\
        # if TN!=0:
        #     print(TN)
    # import random
    # a = random.randrange(0, 5)
    # if a == 0:
    #     print("regression_binary_pred: {}, {}, {}".format(sign_match_percent, output, target))
    return sign_match_percent


def f1_score(output, target):
    with torch.no_grad():
        """
        This method needs to be fixed because because the sign of the normalized data does not
        necessarily match the sign of the un-normalized data
        """

        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()

        # apply the inverse transform in here
        # load the transformation params from the saved file
        with np.load('target_norm_para.npz') as para:
            mean, std = [para[i] for i in ('mean', 'std')]

        output_np = (output_np * std) + mean
        target_np = (target_np * std) + mean

        # calculate the precision, recall and F1-score here
        TP = np.sum(np.sign(output_np) + np.sign(target_np) == 2)
        TN = np.sum((np.sign(output_np) + np.sign(target_np)) == -2)
        FP = np.sum((np.sign(target_np) - np.sign(output_np)) == -2)
        FN = np.sum((np.sign(target_np) - np.sign(output_np)) == 2)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_1_score = 2 * precision * recall / (precision + recall)

        # np.savez("results.npz", F_1_score=F_1_score)

    return F_1_score

