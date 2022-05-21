r"""Simple likelihood complex posterior (SLCP) benchmark.

SLCP is a toy simulator where :math:`\theta` parametrizes a 2-d multivariate Gaussian
from which 4 points are independently drawn and stacked as a single observation :math:`x`.
It is a non-trivial parameter inference benchmark that allows to retrieve the
ground-truth posterior through MCMC sampling of its tractable likelihood.

References:
    Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows
    (Papamakarios et al., 2019)
    https://arxiv.org/abs/1805.07226

Shapes:
    theta: :math:`(5,)`.
    x: :math:`(8,)`.
"""

import torch

from torch import Tensor, BoolTensor
from typing import *

from . import Simulator
from ..distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    ReshapeTransform,
    TransformedDistribution,
)
from ..utils import broadcast


LABELS = [f'$\\theta_{{{i + 1}}}$' for i in range(5)]

LOWER, UPPER = torch.tensor([
    [-3., 3.],  # theta_1
    [-3., 3.],  # theta_2
    [-3., 3.],  # theta_3
    [-3., 3.],  # theta_4
    [-3., 3.],  # theta_5
]).t()


class SLCP(Simulator):
    r"""Creates a simple likelihood complex posterior (SLCP) simulator."""

    def likelihood(self, theta: Tensor, eps: float = 1e-8) -> Distribution:
        r"""Returns the likelihood distribution :math:`p(x | \theta)`."""

        # Mean
        mu = theta[..., :2]

        # Covariance
        s1 = theta[..., 2] ** 2 + eps
        s2 = theta[..., 3] ** 2 + eps
        rho = theta[..., 4].tanh()

        cov = torch.stack([
            s1 ** 2, rho * s1 * s2,
            rho * s1 * s2, s2 ** 2,
        ], dim=-1).reshape(theta.shape[:-1] + (2, 2))

        # Repeat 4 times
        mu = mu.unsqueeze(-2).repeat_interleave(4, -2)
        cov = cov.unsqueeze(-3).repeat_interleave(4, -3)

        # Normal distribution
        normal = MultivariateNormal(mu, cov)

        return TransformedDistribution(
            Independent(normal, 1),
            ReshapeTransform((4, 2), (8,)),
        )

    def __call__(self, theta: Tensor) -> Tensor:
        return self.likelihood(theta).sample()

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = broadcast(theta, x, ignore=1)
        return self.likelihood(theta).log_prob(x)
