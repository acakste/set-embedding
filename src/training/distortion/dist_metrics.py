import math
import torch
import torch.linalg as linalg
from time import time
import matplotlib.pyplot as plt

from typing import Union
import pykeops.torch as keops
import torch
import tqdm

def euc_dist_centroids(points_1, points_2):
    mass_centre_1 = points_1.mean(axis=0)
    mass_centre_2 = points_2.mean(axis=0)
    return torch.sum((mass_centre_1 - mass_centre_2)**2)**(1/2)

def wasserstein(X,Y,p=4, verbose=False, max_iters=100):
    d,_,_ = sinkhorn(X,Y,p, verbose=verbose, max_iters=max_iters)
    return d

def sinkhorn(x: torch.Tensor, y: torch.Tensor, p: float = 2,
             w_x: Union[torch.Tensor, None] = None,
             w_y: Union[torch.Tensor, None] = None,
             eps: float = 1e-3,
             max_iters: int = 100, stop_thresh: float = 1e-5,
             verbose=False):
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm. This code will use the GPU if you pass in GPU tensors.
    Note that this algorithm can be backpropped through
    (though this may be slow if using many iterations).

    :param x: A [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: A [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param p: Which norm to use. Must be an integer greater than 0.
    :param w_x: A [n,] shaped tensor of optional weights for the points x (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param w_y: A [m,] shaped tensor of optional weights for the points y (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param eps: The reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: The maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: Stop if the maximum change in the parameters is below this amount
    :param verbose: Print iterations
    :return: a triple (d, corrs_x_to_y, corr_y_to_x) where:
      * d is the approximate p-wasserstein distance between point clouds x and y
      * corrs_x_to_y is a [n,]-shaped tensor where corrs_x_to_y[i] is the index of the approximate correspondence in point cloud y of point x[i] (i.e. x[i] and y[corrs_x_to_y[i]] are a corresponding pair)
      * corrs_y_to_x is a [m,]-shaped tensor where corrs_y_to_x[i] is the index of the approximate correspondence in point cloud x of point y[j] (i.e. y[j] and x[corrs_y_to_x[j]] are a corresponding pair)
    """

    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(p, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                         f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}")

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(f"w_x must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_x must match the shape of x in dimension 0 but got "
                             f"x.shape = {x.shape} and w_x.shape = {w_x.shape}")
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(f"w_y must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_y must match the shape of y in dimension 0 but got "
                             f"y.shape = {y.shape} and w_y.shape = {w_y.shape}")

    # Distance matrix [n, m]
    x_i = keops.Vi(x)  # [n, 1, d]
    y_j = keops.Vj(y)  # [i, m, d]
    if p == 1:
        M_ij = ((x_i - y_j) ** p).abs().sum(dim=2) # [n, m]
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p)  # [n, m]

    # Weights [n,] and [m,]
    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
        # w_y *= (w_x.shape[0] / w_y.shape[0])

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(f"Weights w_x and w_y do not sum to the same value, "
                         f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                         f"(absolute difference = {abs(sum_w_x - sum_w_y)}")

    log_a = torch.log(w_x)  # [n]
    log_b = torch.log(w_y)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)

    u_i = keops.Vi(u.unsqueeze(-1))
    v_j = keops.Vj(v.unsqueeze(-1))

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        summand_u = (-M_ij + v_j) / eps
        u = eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
        u_i = keops.Vi(u.unsqueeze(-1))

        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
        v_j = keops.Vj(v.unsqueeze(-1))

        max_err_u = torch.max(torch.abs(u_prev-u))
        max_err_v = torch.max(torch.abs(v_prev-v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})
        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()

    approx_corr_1 = P_ij.argmax(dim=1).squeeze(-1)
    approx_corr_2 = P_ij.argmax(dim=0).squeeze(-1)

    if u.shape[0] > v.shape[0]:
        distance = (P_ij * M_ij).sum(dim=1).sum()
    else:
        distance = (P_ij * M_ij).sum(dim=0).sum()
    return distance, approx_corr_1, approx_corr_2


def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()




"""
    *   Scales sort of with 1/d^2 wrt dimension for fewer samples with the same distribution
    *   Taking mu(A) = d*[1], mu(B) = d*[-1], the distance is always close to d*4 = \sum_{i}^d (1-(-1))^2. Nice!
"""