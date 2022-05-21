r"""Flows and parametric distributions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from torch import Tensor, Size
from typing import *

from . import MLP, MaskedMLP
from ..distributions import *
from ..utils import broadcast


__all__ = [
    'DistributionModule', 'TransformModule', 'FlowModule',
    'MaskedAutoregressiveTransform', 'MAF',
]


class DistributionModule(nn.Module):
    r"""Abstract distribution module."""

    def forward(y: Tensor = None) -> Distribution:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A distribution :math:`p(x | y)`.
        """

        raise NotImplementedError()


class TransformModule(nn.Module):
    r"""Abstract transform module."""

    def forward(y: Tensor = None) -> Transform:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A transform :math:`y = f(x | y)`.
        """

        raise NotImplementedError()


class FlowModule(DistributionModule):
    r"""Creates a normalizing flow module.

    Arguments:
        transforms: A list of transforms.
        base: A distribution.
    """

    def __init__(
        self,
        transforms: List[TransformModule],
        base: DistributionModule,
    ):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self, y: Tensor = None) -> NormalizingFlow:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A normalizing flow :math:`p(x | y)`.
        """
        return NormalizingFlow(
            [t(y) for t in self.transforms],
            self.base(y) if y is None else self.base(y).expand(y.shape[:-1]),
        )


class Buffer(nn.Module):
    r"""Creates a buffer module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *args,
    ):
        super().__init__()

        self.meta = meta

        for i, arg in enumerate(args):
            self.register_buffer(f'_{i}', arg)

    def __repr__(self) -> str:
        return repr(self.forward())

    def forward(self, y: Tensor = None) -> Any:
        return self.meta(*self._buffers.values())


class Parameters(nn.ParameterList):
    r"""Holds tensors as parameters."""

    def __init__(self, parameters: Iterable[Tensor]):
        super().__init__(list(map(nn.Parameter, parameters)))

    def extra_repr(self) -> str:
        return '\n'.join(f'({i}): {p.shape}' for i, p in enumerate(self))


class Parametrization(nn.Module):
    r"""Creates a parametrization module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *params: Tensor,
        context: int = 0,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta
        self.shapes = [p.shape for p in params]
        self.sizes = [s.numel() for s in self.shapes]

        if context > 0:
            self.params = build(context, sum(self.sizes), **kwargs)
        else:
            self.params = Parameters(params)

    def extra_repr(self) -> str:
        base = self.meta(*map(torch.randn, self.shapes))
        return f'(base): {base}'

    def forward(self, y: Tensor = None) -> Any:
        if isinstance(self.params, Parameters):
            args = self.params
        else:
            args = self.params(y).split(self.sizes, dim=-1)
            args = [a.reshape(a.shape[:-1] + s) for a, s in zip(args, self.shapes)]

        return self.meta(*args)


class MaskedAutoregressiveTransform(TransformModule):
    r"""Creates a masked autoregressive transform.

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The feature ordering. If :py:`None`, use :py:`range(features)` instead.
        passes: The number of passes for the inverse transformation.
        univariate: A univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`lampe.nn.MaskedMLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        order: LongTensor = None,
        passes: int = -1,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: List[Size] = [(), ()],
        **kwargs,
    ):
        super().__init__()

        self.univariate = univariate
        self.shapes = list(map(Size, shapes))
        self.sizes = [s.numel() for s in self.shapes]

        if order is None:
            order = torch.arange(features)

        self.passes = passes if passes > 0 else features
        self.order = torch.div(order, ceil(features / self.passes), rounding_mode='floor')

        in_order = torch.cat((self.order, torch.full((context,), -1)))
        out_order = self.order.tile(sum(self.sizes))

        self.params = MaskedMLP(in_order[..., None] < out_order, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        return '\n'.join([
            f'(base): {base}',
            f'(order): {self.order.tolist()}',
        ])

    def forward(self, y: Tensor = None) -> AutoregressiveTransform:
        def meta(x: Tensor) -> Transform:
            if y is not None:
                x = torch.cat(broadcast(x, y, ignore=1), dim=-1)

            params = self.params(x)
            params = params.reshape(*params.shape[:-1], sum(self.sizes), -1)
            params = params.transpose(-1, -2).contiguous()

            args = params.split(self.sizes, dim=-1)
            args = [a.reshape(a.shape[:-1] + s) for a, s in zip(args, self.shapes)]

            return self.univariate(*args)

        return AutoregressiveTransform(meta, self.passes)


class MAF(FlowModule):
    r"""Creates a masked autoregressive flow (MAF).

    References:
        Masked Autoregressive Flow for Density Estimation
        (Papamakarios et al., 2017)
        https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transforms.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        linear: bool = False,
        **kwargs,
    ):
        increasing = torch.arange(features)
        decreasing = torch.flipud(increasing)

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=decreasing if i % 2 else increasing,
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = Buffer(DiagNormal, torch.zeros(features), torch.ones(features))

        super().__init__(transforms, base)
