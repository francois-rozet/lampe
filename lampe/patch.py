r"""PyTorch monkey patches."""

import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from typing import *


__all__ = []


################
# Distribution #
################

def deepapply(self: Any, f: Callable):  # -> self
    r"""Applies :py:`f` to all tensors referenced in :py:`self`."""

    if torch.is_tensor(self):
        self = f(self)
    elif isinstance(self, nn.Module):
        self = self.apply(f)
    elif isinstance(self, dict):
        for key, value in self.items():
            self[key] = deepapply(value, f)
    elif isinstance(self, list):
        for i, value in enumerate(self):
            self[i] = deepapply(value, f)
    elif isinstance(self, tuple):
        self = tuple(
            deepapply(value, f)
            for value in self
        )
    elif hasattr(self, '__dict__'):
        deepapply(self.__dict__, f)

    return self

def deepto(self: Any, *args, **kwargs):  # -> self
    r"""Moves and/or casts all tensors references in :py:`self`."""

    return deepapply(self, lambda t: t.to(*args, **kwargs))

Distribution.to = deepto
Distribution.arg_constraints = {}
Distribution._validate_args = False


##########
# Module #
##########

def requires_grad_(self: nn.Module, mode: bool = True):  # -> self
    r"""Sets whether autograd should record operations on the module's parameters."""

    for p in self.parameters():
        p.requires_grad_(mode)

    return self

nn.Module.requires_grad_ = requires_grad_


#############
# Optimizer #
#############

def lrs(self: Optimizer) -> Iterable[float]:
    r"""Yields the learning rates of the parameter groups."""

    return (group['lr'] for group in self.param_groups)

def parameters(self: Optimizer) -> Iterable[Tensor]:
    r"""Yields the parameter tensors of the parameter groups."""

    return (p for group in self.param_groups for p in group['params'])

Optimizer.lrs = lrs
Optimizer.parameters = parameters
