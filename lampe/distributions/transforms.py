r"""Parameterizable transformations and bijections."""

import math
import torch
import torch.nn.functional as F

from torch import Tensor, LongTensor
from torch.distributions import Transform, constraints
from typing import *

from ..utils import broadcast


torch.distributions.transforms._InverseTransform.__name__ = 'Inverse'


class PermutationTransform(Transform):
    r"""Creates a transformation that permutes the elements.

    Arguments:
        order: The permuatation order, with shape :math:`(*, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        order: LongTensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.order = order

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.order.tolist()})'

    def _call(self, x: Tensor) -> Tensor:
        return x.gather(-1, self.order.expand(x.shape))

    def _inverse(self, y: Tensor) -> Tensor:
        return y.gather(-1, torch.argsort(self.order, -1).expand(y.shape))

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.new_zeros(x.shape[:-1])


class CosTransform(Transform):
    r"""Creates a transformation :math:`f(x) = -\cos(x)`."""

    domain = constraints.interval(0, math.pi)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CosTransform)

    def _call(self, x: Tensor) -> Tensor:
        return -x.cos()

    def _inverse(self, y: Tensor) -> Tensor:
        return (-y).acos()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.sin().abs().log()


class SinTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \sin(x)`."""

    domain = constraints.interval(-math.pi / 2, math.pi / 2)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SinTransform)

    def _call(self, x: Tensor) -> Tensor:
        return x.sin()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.asin()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.cos().abs().log()


class MonotonicAffineTransform(Transform):
    r"""Creates a transformation :math:`f(x) = x \times \text{softplus}(\alpha) + \beta`.

    Arguments:
        shift: The shift term :math:`\beta`, with shape :math:`(*,)`.
        scale: The unconstrained scale factor :math:`\alpha`, with shape :math:`(*,)`.
        eps: A numerical stability term.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shift = shift
        self.scale = F.softplus(scale) + eps

    def _call(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

    def _inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log(self.scale).expand(x.shape)


class MonotonicRationalQuadraticSplineTransform(Transform):
    r"""Creates a monotonic rational-quadratic spline transformation.

    References:
        Neural Spline Flows (Durkan et al., 2019)
        https://arxiv.org/abs/1906.04032

    Arguments:
        widths: The unconstrained bin widths, with shape :math:`(*, K)`.
        heights: The unconstrained bin heights, with shape :math:`(*, K)`.
        derivatives: The unconstrained knot derivatives, with shape :math:`(*, K - 1)`.
        bound: The spline's (co)domain bound :math:`B`.
        eps: A numerical stability term.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: float = 3,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        widths = F.softmax(widths, dim=-1) + eps
        heights = F.softmax(heights, dim=-1) + eps
        derivatives = F.softplus(derivatives) + eps

        self.horizontal = 2 * bound * torch.cumsum(F.pad(widths, (1, 0), value=-1), dim=-1)
        self.vertical = 2 * bound * torch.cumsum(F.pad(heights, (1, 0), value=-1), dim=-1)
        self.derivatives = F.pad(derivatives, (1, 1), value=1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bins={self.bins})'

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: LongTensor) -> Tuple:
        mask = torch.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = torch.stack((k, k+1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = hs.gather(-1, k0_k1).squeeze(-1)
        y0, y1 = vs.gather(-1, k0_k1).squeeze(-1)
        d0, d1 = ds.gather(-1, k0_k1).squeeze(-1)

        s = (y1 - y0) / (x1 - x0)

        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Tensor, value: Tensor) -> LongTensor:
        return torch.sum(seq < value[..., None], dim=-1) - 1

    def _call(self, x: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        y = (
            y0 +
            (y1 - y0) *
            (s * z**2 + d0 * z * (1 - z)) /
            (s + (d0 + d1 - 2 * s) * z * (1 - z))
        )

        return torch.where(mask, y, x)

    def _inverse(self, y: Tensor) -> Tensor:
        k = self.searchsorted(self.vertical, y)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)

        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - (b**2 - 4 * a * c).sqrt())

        x = x0 + z * (x1 - x0)

        return torch.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z)**2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z))**2
        )

        return torch.log(jacobian) * mask


class MonotonicTransform(Transform):
    r"""Creates a transformation from a monotonic univariate function :math:`f(x)`.

    The inverse function :math:`f^{-1}` is approximated using the bisection method.

    Wikipedia:
        https://wikipedia.org/wiki/Bisection_method

    Arguments:
        f: A monotonic univariate function :math:`f(x)`.
        bound: The domain bound :math:`B`.
        eps: The numerical tolerance for the inverse transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        bound: float = 1e1,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.f = f
        self.bound = bound
        self.eps = eps

    def _call(self, x: Tensor) -> Tensor:
        return self.f(x / torch.sqrt(1 + (x / self.bound)**2))

    def _inverse(self, y: Tensor) -> Tensor:
        a = torch.full_like(y, -self.bound)
        b = torch.full_like(y, self.bound)

        for _ in range(int(math.log2(self.bound / self.eps))):
            c = (a + b) / 2

            mask = self.f(c) < y

            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)

        x = (a + b) / 2

        return x / torch.sqrt(1 - (x / self.bound)**2)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log(torch.autograd.functional.jacobian(
            func=lambda x: self._call(x).sum(),
            inputs=x,
            create_graph=torch.is_grad_enabled(),
        ))


class AutoregressiveTransform(Transform):
    r"""Tranform via an autoregressive mapping.

    Arguments:
        meta: A meta function :math:`g(x) = f`.
        passes: The number of passes for the inverse transformation.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        passes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.meta = meta
        self.passes = passes

    def _call(self, x: Tensor) -> Tensor:
        return self.meta(x)(x)

    def _inverse(self, y: Tensor) -> Tensor:
        x = torch.zeros_like(y)
        for _ in range(self.passes):
            x = self.meta(x).inv(y)

        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.meta(x).log_abs_det_jacobian(x, y).sum(dim=-1)
