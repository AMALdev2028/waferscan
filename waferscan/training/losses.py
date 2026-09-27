"""
Loss functions.

focal_loss: cross-entropy that down-weights easy examples by (1 - p)^gamma,
    with label smoothing (targets 0.9/0.0125 instead of 1/0) and per-class
    weights. gamma is *scheduled*: start at 0 (plain CE - stable early
    learning), ramp up so later epochs concentrate on the hard wafers
    (Edge-Loc vs Edge-Ring, Loc vs Center...).
class_balanced_weights: Cui et al. 2019 "effective number of samples":
    E_n = (1 - beta^n) / (1 - beta). With WM-811K's 147k 'None' vs 149
    'Near-full', plain 1/n weights explode; 1/E_n saturates sensibly.
evidential_loss: Sensoy et al. 2018. The head outputs *evidence* for each
    class; alpha = evidence + 1 are Dirichlet parameters. Few total evidence
    (small sum(alpha)) = "I don't know" - an uncertainty that does not need
    sampling. The KL term (annealed in) pushes evidence for wrong classes to 0.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def class_balanced_weights(counts, beta: float = 0.999) -> torch.Tensor:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return torch.tensor(w / w.sum() * len(counts), dtype=torch.float32)


def gamma_at(progress: float, start: float = 0.0, end: float = 2.0, ramp: float = 0.5) -> float:
    """Linear ramp from `start` to `end` over the first `ramp` fraction of training."""
    return end if ramp <= 0 else start + (end - start) * min(1.0, progress / ramp)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float,
               weight: torch.Tensor | None = None, smoothing: float = 0.1) -> torch.Tensor:
    c = logits.shape[1]
    logp = F.log_softmax(logits.float(), 1)
    q = torch.full_like(logp, smoothing / c).scatter_(1, target[:, None], 1.0 - smoothing + smoothing / c)
    per = -(q * (1.0 - logp.exp()).pow(gamma) * logp).sum(1)
    if weight is not None:
        w = weight.to(logits.device)[target]
        return (per * w).sum() / w.sum()
    return per.mean()


def dirichlet_from_evidence(evid_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(evid_logits.float()) + 1.0


def evidential_loss(evid_logits: torch.Tensor, target: torch.Tensor, progress: float, anneal: float = 0.5) -> torch.Tensor:
    alpha = dirichlet_from_evidence(evid_logits)
    c = alpha.shape[1]
    y = F.one_hot(target, c).float()
    s = alpha.sum(1, keepdim=True)
    p = alpha / s
    mse = ((y - p) ** 2 + p * (1 - p) / (s + 1)).sum(1)
    a_t = y + (1 - y) * alpha                                   # remove evidence for the true class
    s_t = a_t.sum(1, keepdim=True)
    kl = (torch.lgamma(s_t).squeeze(1) - math.lgamma(c) - torch.lgamma(a_t).sum(1)
          + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(s_t))).sum(1))
    lam = min(1.0, progress / anneal) if anneal > 0 else 1.0
    return (mse + lam * kl).mean()
