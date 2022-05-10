r"""Priors and distributions."""

import math
import torch

from textwrap import indent
from torch import Tensor
from torch.distributions import *
from torch.distributions.constraints import interval
from typing import *


class BoxUniform(Independent):
    r"""Creates a distribution for a multivariate random variable :math:`X`
    distributed uniformly over an hypercube domain. Formally,

    .. math:: l_i \leq X_i < u_i ,

    where :math:`l_i` and :math:`u_i` are respectively the lower and upper bounds
    of the domain in the :math:`i`-th dimension.

    Arguments:
        lower: The lower bounds (inclusive).
        upper: The upper bounds (exclusive).
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = BoxUniform(-torch.ones(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([ 0.1859, -0.9698,  0.0665])
    """

    def __init__(self, lower: Tensor, upper: Tensor, ndims: int = 1):
        super().__init__(Uniform(lower, upper), ndims)

    def __repr__(self) -> str:
        return f'Box{self.base_dist}'


class DiagNormal(Independent):
    r"""Creates a multivariate normal distribution parametrized by the variables
    mean :math:`\mu` and standard deviation :math:`\sigma`, but assumes no
    correlation between the variables.

    Arguments:
        loc: The mean :math:`\mu` of the variables.
        scale: The standard deviation :math:`\sigma` of the variables.
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = DiagNormal(torch.zeros(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([ 0.7304, -0.1976, -1.7591])
    """

    def __init__(self, loc: Tensor, scale: Tensor, ndims: int = 1):
        super().__init__(Normal(loc, scale), ndims)

    def __repr__(self) -> str:
        return f'Diag{self.base_dist}'


class Joint(Distribution):
    r"""Joins independent random variables into a single distribution.

    Arguments:
        marginals: A list of independent distributions. The distributions
            should not be batched.

    Example:
        >>> d = Joint([Uniform(0, 1), Normal(0, 1)])
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([ 0.8969, -2.6717])
    """

    def __init__(self, marginals: List[Distribution]):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([sum(
                dist.event_shape.numel()
                for dist in marginals
            )]),
        )

        self.marginals = marginals

    def __repr__(self) -> str:
        lines = [
            indent(repr(dist), '  ')
            for dist in self.marginals
        ]

        return f'{self.__class__.__name__}(\n' + ',\n'.join(lines) + '\n)'

    def log_prob(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        i, lp = 0, 0

        for dist in self.marginals:
            j = i + dist.event_shape.numel()
            y = x[..., i:j].reshape(shape + dist.event_shape)
            lp = lp + dist.log_prob(y)
            i = j

        return lp

    def sample(self, shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.sample(shape)
            y = y.reshape(shape + (-1,))
            x.append(y)

        return torch.cat(x, dim=-1)


class Sort(Distribution):
    r"""Creates a distribution for a :math:`n`-d random variable :math:`X`, whose elements
    :math:`X_i` are :math:`n` draws from a base distribution :math:`p(Y)`, ordered
    such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            n! \, \prod_{i = 1}^n p(Y = x_i) & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Y)`.
        n: The number of draws :math:`n`.
        descending: Whether the elements are sorted in descending order or not.

    Example:
        >>> d = Sort(Normal(0, 1), 3)
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([-1.4434, -0.3861,  0.2439])
    """

    def __init__(
        self,
        base: Distribution,
        n: int = 2,
        descending: bool = False,
    ):
        assert len(base.event_shape) < 1, "base must be scalar"

        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=torch.Size([n]),
        )

        self.base = base
        self.n = n
        self.descending = descending
        self.log_fact = math.log(math.factorial(n))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.base}, {self.n})'

    def log_prob(self, x: Tensor) -> Tensor:
        if self.descending:
            ordered = x[..., :-1] >= x[..., 1:]
        else:
            ordered = x[..., :-1] <= x[..., 1:]

        ordered = ordered.all(dim=-1)

        return (
            ordered.log() +
            self.log_fact +
            self.base.log_prob(x).sum(dim=-1)
        )

    def sample(self, shape: torch.Size = ()) -> Tensor:
        x = torch.stack([
            self.base.sample(shape)
            for _ in range(self.n)
        ], dim=-1)
        x, _ = torch.sort(x, dim=-1, descending=self.descending)

        return x


class TopK(Sort):
    r"""Creates a distribution for a :math:`k`-d random variable :math:`X`, whose elements
    :math:`X_i` are the top :math:`k` among :math:`n` draws from a base distribution
    :math:`p(Y)`, ordered such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            \frac{n!}{(n - k)!} \, \prod_{i = 1}^k p(Y = x_i)
                \, P(Y \geq x_k)^{n - k} & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Y)`.
        k: The number of selected elements :math:`k`.
        n: The number of draws :math:`n`.
        kwargs: Keyword arguments passed to :class:`Sort`.

    Example:
        >>> d = TopK(Normal(0, 1), 2, 3)
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([-0.2167,  0.6739])
    """

    def __init__(
        self,
        base: Distribution,
        k: int = 1,
        n: int = 2,
        **kwargs,
    ):
        assert 1 <= k < n, "k should be in [1, n)"

        super().__init__(base, n, **kwargs)

        self._event_shape = torch.Size([k])

        self.k = k
        self.log_fact = self.log_fact - math.log(math.factorial(n - k))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.base}, {self.k}, {self.n})'

    def log_prob(self, x: Tensor) -> Tensor:
        cdf = self.base.cdf(x[..., -1])

        if not self.descending:
            cdf = 1 - cdf

        return (
            (self.n - self.k) * cdf.log() +
            super().log_prob(x)
        )

    def sample(self, shape: torch.Size = ()) -> Tensor:
        return super().sample(shape)[..., :self.k]


class Minimum(TopK):
    r"""Creates a distribution for a scalar random variable :math:`X`, which is the
    minimum among :math:`n` draws from a base distribution :math:`p(Y)`.

    .. math:: p(X = x) = n \, p(Y = x) \, P(Y \geq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Y)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Minimum(Normal(0, 1), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(-1.7552)
    """

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, 1, n)

        self._event_shape = torch.Size()

        self.descending = False

    def __repr__(self) -> str:
        return Sort.__repr__(self)

    def log_prob(self, x: Tensor) -> Tensor:
        return super().log_prob(x.unsqueeze(dim=-1))

    def sample(self, shape: torch.Size = ()) -> Tensor:
        return super().sample(shape).squeeze(dim=-1)


class Maximum(Minimum):
    r"""Creates a distribution for a scalar random variable :math:`X`, which is the
    maximum among :math:`n` draws from a base distribution :math:`p(Y)`.

    .. math:: p(X = x) = n \, p(Y = x) \, P(Y \leq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Y)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Maximum(Normal(0, 1), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(1.1644)
    """

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, n)

        self.descending = True


class NoisyUniform(Uniform):
    r"""Creates a distribution for a random variable :math:`X = Y + \varepsilon`, where
    :math:`Y` follows a uniform distribution :math:`\mathcal{U}(l, u)` and
    :math:`\varepsilon` is random noise.

    .. math:: p(X = x) = \frac{1}{u - l}
        \int_{x - u}^{x - l} p(\varepsilon) \operatorname{d}\!\varepsilon

    Arguments:
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).
        noise: The noise distribution :math:`p(\varepsilon)`.
        bins: The number of bins to approximate the integral.

    Example:
        >>> d = NoisyUniform(0, 1, Normal(0, 1))
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(1.2437)
    """

    def __init__(
        self,
        lower: Tensor,
        upper: Tensor,
        noise: Distribution,
        bins: int = 32,
    ):
        super().__init__(lower, upper)

        self.noise = noise
        self.bins = bins

    def log_prob(self, x: Tensor) -> Tensor:
        a, b = x - self.high, x - self.low
        p = self.noise.cdf(b) - self.noise.cdf(a)

        t = torch.linspace(0, 1, self.bins)
        t = t.view((-1,) + (1,) * a.dim()).to(a)

        return torch.where(
            p > 1e-3,
            torch.log(p + 1e-9) - torch.log(b - a),
            torch.logsumexp(
                self.noise.log_prob(torch.lerp(a, b, t)),
                dim=0,
            ) - math.log(self.bins),
        )

    def sample(self, shape: torch.Size = ()) -> Tensor:
        return super().sample(shape) + self.noise.sample(shape)


class TransformedUniform(TransformedDistribution):
    r"""Creates a distribution for a random variable :math:`X`, whose
    transformation :math:`f(X)` is uniformly distributed over the interval
    :math:`[f(l), f(u)]`.

    .. math:: p(X = x) = \frac{1}{f(u) - f(l)}
        \begin{cases}
            f'(x) & \text{if } f(l) \leq f(x) < f(u) \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).
        f: A transformation :math:`f`, monotonically increasing over
            :math:`[l, u]`.

    Example:
        >>> d = TransformedUniform(-1, 1, ExpTransform())
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(0.5594)
    """

    def __init__(self, lower: Tensor, upper: Tensor, f: Transform):
        super().__init__(Uniform(f(lower), f(upper)), [f.inv])


class TruncatedDistribution(Distribution):
    r"""Truncates the base distribution :math:`p(X)` of a random variable :math:`X`
    between a lower bound :math:`l` and an upper bound :math:`u`.

    .. math:: p(X = x | l \leq X < u) = \frac{1}{P(X \leq u) - P(X \leq l)}
        \begin{cases}
            p(X = x) & \text{if } l \leq x < u \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(X)`.
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).

    Example:
        >>> d = TruncatedDistribution(Normal(0, 1), 1, 2)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(1.2573)
    """

    def __init__(
        self,
        base: Distribution,
        lower: Tensor = float('-inf'),
        upper: Tensor = float('+inf'),
    ):
        assert len(base.event_shape) < 1, "base must be scalar"

        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=torch.Size(),
        )

        self.base = base
        self.uniform = Uniform(base.cdf(lower), base.cdf(upper))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.base})'

    def cdf(self, x: Tensor) -> Tensor:
        return self.uniform.cdf(self.base.cdf(x))

    def log_prob(self, x: Tensor) -> Tensor:
        return self.uniform.log_prob(self.base.cdf(x)) + self.base.log_prob(x)

    def sample(self, shape: torch.Size = ()) -> Tensor:
        return self.base.icdf(self.uniform.sample(shape).clamp(1e-6, 1 - 1e-6))


class CosTransform(Transform):
    r"""Transform via the mapping :math:`y = -\cos(x)`."""

    domain = interval(0, math.pi)
    codomain = interval(-1, 1)
    bijective = True

    def _call(self, x: Tensor) -> Tensor:
        return -x.cos()

    def _inverse(self, y: Tensor) -> Tensor:
        return (-y).acos()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.sin().abs().log()


class SinTransform(Transform):
    r"""Transform via the mapping :math:`y = \sin(x)`."""

    domain = interval(-math.pi / 2, math.pi / 2)
    codomain = interval(-1, 1)
    bijective = True

    def _call(self, x: Tensor) -> Tensor:
        return x.sin()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.asin()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.cos().abs().log()
