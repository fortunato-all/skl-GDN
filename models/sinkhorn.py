import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy import sparse

def WasserstainLossLog(loga, logb, C, ϵ, max_iter):
    """
    Calculate the Wasserstain distance between two measures in log space.

    Args:
        loga (n,) or (n, 1): log measure.
        logb (m,) or (m, 1): log measure.
        C (n, m): cost matrix.
    Returns:
        Non-negative scalar.
    """
    if loga.dim() == 1: loga = loga.unsqueeze(1)
    if logb.dim() == 1: logb = logb.unsqueeze(1)

    logu = torch.zeros_like(loga)
    logv = torch.zeros_like(logb)
    for i in range(max_iter):
        logu = loga - torch.logsumexp(-C/ϵ + logv.t(), dim=1, keepdim=True)
        logv = logb - torch.logsumexp(-C/ϵ + logu, dim=0, keepdim=True).t()

    P = torch.exp(logu - C/ϵ + logv.t())
    return torch.sum(P * C)

def approxSinkhornLoss(y, logit, C, ϵ, max_iter, **kwargs):
    """
    Calculate Wasserstain loss with the specified k indices.

    Args:
        y (m, ): the ground truth measure, only a small fraction of elements are nonzeros.
        logits (m,): the model output logit.
        C (m, m): the pre-computed cost matrix.
    kwargs:
        indice (m,): boolean index vector.
        k (int)
    """
    indice, k = kwargs.get("indice", None), kwargs.get("k", None)
    """if indice is not None: # specified logit units
        # logb, idx_b = torch.log_softmax(logit, dim=0)[indice], torch.where(indice)[0]
        logb, idx_b = logit[indice], torch.where(indice)[0]
    elif k is not None: # top-k logit units
        # logb, idx_b = torch.topk(torch.log_softmax(logit, dim=0), k)
        logb, idx_b = torch.topk(logit, k)
    else: # entire logit units
        # logb, idx_b = torch.log_softmax(logit, dim=0), torch.arange(len(logit))"""
    logb, idx_b = torch.log(logit[logit > 0]), torch.arange(len(logit))
    loga, idx_a = torch.log(y[y > 0]), torch.where(y)[0]

    idx_a, idx_b = idx_a.cpu().numpy(), idx_b.cpu().numpy()
    M1 = C[idx_a[:, np.newaxis], idx_b]
    # M2 = C[idx_b[:, np.newaxis], idx_b]
    # M3 = C[idx_a[:, np.newaxis], idx_a]

    M1 = torch.as_tensor(M1, device=y.device, dtype=y.dtype) 
    # M1 = torch.tensor(M1, device=y.device, dtype=y.dtype)
    # M2 = torch.tensor(M2, device=y.device, dtype=y.dtype)
    # M3 = torch.tensor(M3, device=y.device, dtype=y.dtype)

    t1 = WasserstainLossLog(loga, logb, M1, ϵ, max_iter)
    # t2 = WasserstainLossLog(logb, logb, M2, ϵ, max_iter)
    # t3 = WasserstainLossLog(loga, loga, M3, ϵ, max_iter)
    # return 2*t1 - t2 #- t3
    return t1

def getSinkhornLoss(C, ϵ, max_iter):
    def SinkhornLoss(Y, logits, **kwargs):
        """
        Only one of kwargs or no kwargs should be provided.
        Args:
            Y (batch_size, m)
            logits (batch_size, m)
        kwargs:
            indices (batch_size, m)
            k (int)
        """
        batch_size, loss = min(logits.shape[0], 1024), 0.0
        indices, k = [None] * batch_size, None
        for i in range(batch_size):
            loss += approxSinkhornLoss(Y[i], logits[i], C, ϵ, max_iter, indice=indices[i], k=k)
        return loss / batch_size
    return SinkhornLoss
