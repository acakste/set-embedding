

import torch
from src.training.distortion.dist_metrics import wasserstein

def fn_quadratic(d, delta, reduction):
    distortion = (d - delta)**2

    return reduce_loss(distortion, reduction)

def fn_quadratic_weighted(d, delta, reduction):
    weight_func = (1./(d+1e-4))**2
    distortion = weight_func*(d - delta)**2

    return reduce_loss(distortion, reduction)


def fn_soft_fractional(d, delta, reduction, gamma=1e-5):
    _tmp = torch.exp(gamma*(d/(1e-4+delta))) + torch.exp(gamma*(delta/(1e-4+d)))
    distortion = (1/gamma)*torch.log(_tmp/(2*torch.exp(torch.tensor(gamma))))

    return reduce_loss(distortion, reduction)


def fn_huber(d,delta, reduction, tau=0.5):
    _tmp = torch.abs(d-delta)

    expression_1 = (d - delta)**2
    expression_2 = tau*(2*_tmp-tau)

    mask = (_tmp < tau)*1.0
    distortion = expression_1*mask + expression_2*(1.- mask)

    return reduce_loss(distortion, reduction)



distortion_fn_mappings = {
    "fn_quadratic": fn_quadratic
}

def select_distortion_function(fn_str):
    try:
        return distortion_fn_mappings[fn_str]
    except KeyError:
        print(f"{fn_str} is an invalid function.")

def reduce_loss(distortion, reduction):
    if reduction == "mean":
        return torch.mean(distortion)
    elif reduction == "sum":
        return torch.sum(distortion)
    else:
        """Equal to no reduction"""
        return distortion


def distortion_loss(x, x_embed, distortion_func=fn_quadratic, reduction="mean"):
    N, S, _ = x.shape

    input_distances = torch.zeros(size=(int(N*(N-1)/2),))
    embedding_distances = torch.zeros(size=(int(N*(N-1)/2),))
    i = 0
    for index_1 in range(N-1):
        for index_2 in range(index_1+1, N):
            input_distances[i] = wasserstein(x[index_1], x[index_2])
            embedding_distances[i] = torch.sum((x_embed[index_1] - x_embed[index_2])**2)**(1/2)
            i += 1

    return distortion_func(embedding_distances, input_distances, reduction)


class DistortionCriterion():
    def __init__(self, distortion_func, reduction):
        """
        Just a wrapper for the function distortion_loss. MOVE THIS to that file!
        :param distortion_func: fn_quadratic, fn_huber, fn_weighted_quadratic etc
        :param reduction: "none", "sum", "mean"
        """
        self.distortion_func = distortion_func
        self.reduction = reduction

    def __call__(self, x, target):
        """
        :param x: the input instance(s)
        :param target: the target instance(s)
        """
        return self.distortion_func(x, target, reduction=self.reduction)




