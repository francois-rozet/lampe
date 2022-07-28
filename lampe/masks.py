r"""Masking helpers."""

import torch
import torch.nn as nn

from torch import Tensor, BoolTensor, Size
from torch.distributions import Distribution, Bernoulli, Independent
from typing import *


def mask2str(b: BoolTensor) -> str:
    r"""Represents a binary mask as a string.

    Arguments:
        b: A binary mask :math:`b`, with shape :math:`(D,)`.

    Example:
        >>> b = torch.tensor([True, True, False, True, False])
        >>> mask2str(b)
        '11010'
    """

    return ''.join('1' if bit else '0' for bit in b)


def str2mask(string: str) -> BoolTensor:
    r"""Parses the string representation of a binary mask into a tensor.

    Arguments:
        string: A binary mask string representation.

    Example:
        >>> str2mask('11010')
        tensor([True, True, False, True, False])
    """

    return torch.tensor([char == '1' for char in string])


class BernoulliMask(Independent):
    r"""Creates a distribution :math:`P(b)` over all binary masks :math:`b` in the
    hypercube :math:`\{0, 1\}^D` such that each bit :math:`b_i` has a probability
    :math:`p_i` of being positive.

    .. math:: P(b) = \prod^D_{i = 1} p_i^{b_i} (1 - p_i)^{1 - b_i}

    Arguments:
        p: The probability vector :math:`p`, with shape :math:`(*, D)`.

    Example:
        >>> d = BernoulliMask(torch.tensor([0.4, 0.1, 0.9]))
        >>> d.sample()
        tensor([True, False, True])
    """

    has_rsample = False

    def __init__(self, p: Tensor):
        super().__init__(Bernoulli(p), 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(D={self.event_shape.numel()})'

    def log_prob(self, b: BoolTensor) -> Tensor:
        return super().log_prob(b.float())

    def sample(self, shape: Size = ()) -> BoolTensor:
        return super().sample(shape).bool()


class SelectionMask(Distribution):
    r"""Creates a mask distribution :math:`P(b)`, uniform over a selection of
    binary masks :math:`\mathcal{B} \subseteq \{0, 1\}^D`.

    .. math:: P(b) = \begin{cases}
            \frac{1}{|\mathcal{B}|} & \text{if } b \in \mathcal{B} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        selection: A binary mask selection :math:`\mathcal{B}`, with shape :math:`(N, D)`.

    Example:
        >>> selection = torch.tensor([
        ...     [True, False, False],
        ...     [False, True, False],
        ...     [False, False, True],
        ... ])
        >>> d = SelectionMask(selection)
        >>> d.sample()
        tensor([False, True, False])
    """

    def __init__(self, selection: BoolTensor):
        super().__init__(event_shape=selection.shape[-1:])

        self.selection = selection

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(D={self.event_shape.numel()})'

    def log_prob(self, b: BoolTensor) -> Tensor:
        match = torch.all(b[..., None, :] == self.selection, dim=-1)
        prob = match.float().mean(dim=-1)
        return prob.log()

    def sample(self, shape: Size = ()) -> BoolTensor:
        index = torch.randint(len(self.selection), shape)
        return self.selection[index]
