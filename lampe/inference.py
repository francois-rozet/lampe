r"""Inference components such as estimators, training losses and MCMC samplers.

.. admonition:: TODO

    * Finish documentation (NPE, AMNPE).
    * Find references.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor, Size
from typing import *

from .distributions import Distribution, DiagNormal, AffineTransform
from .nn import Affine, MLP
from .nn.flows import Buffer, FlowModule, MAF
from .utils import broadcast


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) classifier network.

    The principle of neural ratio estimation is to train a classifier network
    :math:`d_\phi(\theta, x)` to discriminate between pairs :math:`(\theta, x)`
    equally sampled from the joint distribution :math:`p(\theta, x)` and the
    product of the marginals :math:`p(\theta)p(x)`. Formally, the optimization
    problem is

    .. math:: \arg \min_\phi
        \mathbb{E}_{p(\theta, x)} \big[ \ell(d_\phi(\theta, x)) \big] +
        \mathbb{E}_{p(\theta)p(x)} \big[ \ell(1 - d_\phi(\theta, x)) \big]

    where :math:`\ell(p) = - \log p` is the negative log-likelihood.
    For this task, the decision function modeling the Bayes optimal classifier is

    .. math:: d(\theta, x)
        = \frac{p(\theta, x)}{p(\theta, x) + p(\theta) p(x)}

    thereby defining the likelihood-to-evidence (LTE) ratio

    .. math:: r(\theta, x)
        = \frac{d(\theta, x)}{1 - d(\theta, x)}
        = \frac{p(\theta, x)}{p(\theta) p(x)}
        = \frac{p(x | \theta)}{p(x)}
        = \frac{p(\theta | x)}{p(\theta)} .

    To prevent numerical stability issues when :math:`d_\phi(\theta, x) \to 0`,
    the neural network returns the logit of the class prediction
    :math:`\text{logit}(d_\phi(\theta, x)) = \log r_\phi(\theta, x)`.

    References:
        Approximating Likelihood Ratios with Calibrated Discriminative Classifiers
        (Cranmer et al., 2015)
        https://arxiv.org/abs/1506.02169

        Likelihood-free MCMC with Amortized Approximate Ratio Estimators
        (Hermans et al., 2019)
        https://arxiv.org/abs/1903.04057

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        moments: The parameters moments :math:`\mu` and :math:`\sigma`. If provided,
            the moments are used to standardize the parameters.
        build: The network constructor (e.g. :class:`nn.MLP` or :class:`nn.ResMLP`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        moments: Tuple[Tensor, Tensor] = None,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        if moments is None:
            self.standardize = nn.Identity()
        else:
            mu, sigma = moments
            self.standardize = Affine(-mu / sigma, 1 / sigma)

        self.net = build(theta_dim + x_dim, 1, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta, x)`, with shape :math:`(*,)`.
        """

        theta = self.standardize(theta)
        theta, x = broadcast(theta, x, ignore=1)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class NRELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a NRE classifier
    :math:`d_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = - \log p` is the negative log-likelihood.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class AMNRE(NRE):
    r"""Creates an arbitrary marginal neural ratio estimation (AMNRE) classifier
    network.

    The principle of AMNRE is to introduce, as input to the classifier, a binary mask
    :math:`b \in \{0, 1\}^D` indicating a subset of parameters :math:`\theta_b =
    (\theta_i: b_i = 1)` of interest. Intuitively, this allows the classifier to
    distinguish subspaces and to learn a different ratio for each of them. Formally,
    the classifer network takes the form :math:`d_\phi(\theta_b, x, b)` and the
    optimization problem becomes

    .. math:: \arg \min_\phi
        \mathbb{E}_{p(\theta, x) P(b)} \big[ \ell(d_\phi(\theta_b, x, b)) \big] +
        \mathbb{E}_{p(\theta)p(x) P(b)} \big[ \ell(1 - d_\phi(\theta_b, x, b)) \big],

    where :math:`P(b)` is a binary mask distribution. In this context, the Bayes
    optimal classifier is

    .. math:: d(\theta_b, x, b)
        = \frac{p(\theta_b, x)}{p(\theta_b, x) + p(\theta_b) p(x)}
        = \frac{r(\theta_b, x)}{1 + r(\theta_b, x)} .

    Therefore, a classifier network trained for AMNRE gives access to an estimator
    :math:`\log r_\phi(\theta_b, x, b)` of all marginal LTE log-ratios
    :math:`\log r(\theta_b, x)`.

    References:
        Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference
        (Rozet et al., 2021)
        https://arxiv.org/abs/2110.00449

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        args: Positional arguments passed to :class:`NRE`.
        kwargs: Keyword arguments passed to :class:`NRE`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim * 2, x_dim, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`, or
                a subset :math:`\theta_b`, with shape :math:`(*, |b|)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta_b, x, b)`, with shape :math:`(*,)`.
        """

        zeros = theta.new_zeros(theta.shape[:-1] + b.shape[-1:])

        if b.dim() == 1 and theta.shape[-1] < b.numel():
            theta = zeros.masked_scatter(b, theta)
        else:
            theta = torch.where(b, theta, zeros)

        theta = self.standardize(theta) * b
        theta, x, b = broadcast(theta, x, b * 2. - 1., ignore=1)

        return self.net(torch.cat((theta, x, b), dim=-1)).squeeze(-1)


class AMNRELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a AMNRE classifier
    :math:`d_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i \odot b_i, x_i, b_i)) +
        \ell(1 - d_\phi(\theta_{i+1} \odot b_i, x_i, b_i))

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
    ):
        super().__init__()

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        b = self.mask_dist.sample(theta.shape[:-1])

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x, b,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class NPE(nn.Module):
    r"""Creates a neural posterior estimation (NPE) normalizing flow.

    TODO

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        moments: The parameters moments :math:`\mu` and :math:`\sigma`. If provided,
            the moments are used to standardize the parameters.
        build: The flow constructor (e.g. :class:`lampe.nn.flows.MAF`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        moments: Tuple[Tensor, Tensor] = None,
        build: Callable[[int, int], FlowModule] = MAF,
        **kwargs,
    ):
        super().__init__()

        self.flow = build(theta_dim, x_dim, **kwargs)

        if moments is not None:
            mu, sigma = moments
            self.flow.transforms.insert(0, Buffer(AffineTransform, -mu / sigma, 1 / sigma))

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.flow(x).log_prob(theta)

    @torch.no_grad()
    def sample(self, x: Tensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            shape: TODO

        Returns:
            The samples :math:`\theta \sim p_\phi(\theta | x)`,
            with shape :math:`(*, S, D)`.
        """

        return self.flow(x).sample(shape)


class NPELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a NPE normalizing flow
    :math:`p_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N -\log p_\phi(\theta_i | x_i) .

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x)

        return -log_p.mean()


class AMNPE(NPE):
    r"""Creates an arbitrary marginal neural posterior estimation (AMNPE)
    normalizing flow.

    TODO

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        args: Positional arguments passed to :class:`NPE`.
        kwargs: Keyword arguments passed to :class:`NPE`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim, x_dim + theta_dim, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x, b)`, with shape :math:`(*,)`.
        """

        theta, x, b = broadcast(theta, x, b * 2. - 1., ignore=1)

        return self.flow(torch.cat((x, b), dim=-1)).log_prob(theta)

    @torch.no_grad()
    def sample(self, x: Tensor, b: BoolTensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(D,)`.
            shape: TODO

        Returns:
            The samples :math:`\theta_b \sim p_\phi(\theta_b | x, b)`,
            with shape :math:`(*, S, D)`.
        """

        x, b = broadcast(x, b * 2. - 1., ignore=1)

        return self.flow(torch.cat((x, b), dim=-1)).sample(shape)[..., b]


class AMNPELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of an AMNPE normalizing flow
    :math:`p_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        -\log p_\phi(\theta_i \odot b_i + \theta_{i + 1} \odot (1 - b_i) | x_i, b_i)

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
    ):
        super().__init__()

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        b = self.mask_dist.sample(theta.shape[:-1])
        theta = torch.where(b, theta, theta_prime)

        log_prob = self.estimator(theta, x, b)

        return -log_prob.mean()


class MetropolisHastings(object):
    r"""Creates a batched Metropolis-Hastings sampler.

    Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) sampling algorithm used to
    sample from intractable distributions :math:`p(x)` whose density is proportial to a
    tracatble function :math:`f(x)`, with :math:`x \in \mathcal{X}`. The algorithm
    consists in repeating the following routine for :math:`t = 1` to :math:`T`, where
    :math:`x_0` is the initial sample and :math:`q(x' | x)` is a pre-defined transition
    distribution.

    1. sample :math:`x' \sim q(x' | x_{t-1})`
    2. :math:`\displaystyle \alpha \gets \frac{f(x')}{f(x_{t-1})} \frac{q(x_{t-1} | x')}{q(x' | x_{t-1})}`
    3. sample :math:`u \sim \mathcal{U}(0, 1)`
    4. :math:`x_t \gets \begin{cases} x' & \text{if } u \leq \alpha \\ x_{t-1} & \text{otherwise} \end{cases}`

    Asymptotically, i.e. when :math:`T \to \infty`, the distribution of samples
    :math:`x_t` is guaranteed to converge towards :math:`p(x)`. In this implementation,
    a Gaussian transition :math:`q(x' | x) = \mathcal{N}(x'; x, \Sigma)` is used, which
    can be modified by subclassing :class:`MetropolisHastings`.

    Wikipedia:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

    Arguments:
        x_0: A batch of initial points :math:`x_0`, with shape :math:`(*, L)`.
        f: A function :math:`f(x)` proportional to a density function :math:`p(x)`.
        log_f: The logarithm :math:`\log f(x)` of a function proportional
            to :math:`p(x)`.
        sigma: The standard deviation of the Gaussian transition.
            Either a scalar or a vector.

    Example:
        >>> x_0 = torch.rand(128, 7)
        >>> log_f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> sampler = MetropolisHastings(x_0, log_f=log_f, sigma=0.5)
        >>> samples = [x for x in sampler(2**8, burn=2**7, step=2**2)]
        >>> samples = torch.stack(samples)
        >>> samples.shape
        torch.Size([32, 128, 7])
    """

    def __init__(
        self,
        x_0: Tensor,
        f: Callable[[Tensor], Tensor] = None,
        log_f: Callable[[Tensor], Tensor] = None,
        sigma: Union[float, Tensor] = 1.,
    ):
        super().__init__()

        self.x_0 = x_0

        assert f is not None or log_f is not None, \
            "Either 'f' or 'log_f' must be provided."

        if f is None:
            self.log_f = log_f
        else:
            self.log_f = lambda x: f(x).log()

        self.q = lambda x: DiagNormal(x, torch.ones_like(x) * sigma)
        self.symmetric = True  # q(x | y) is equal to q(y | x)

    def __iter__(self) -> Iterator[Tensor]:
        x = self.x_0

        # log f(x)
        log_f_x = self.log_f(x)

        while True:
            # y ~ q(y | x)
            y = self.q(x).sample()

            # log f(y)
            log_f_y = self.log_f(y)

            #     f(y)   q(x | y)
            # a = ---- * --------
            #     f(x)   q(y | x)
            log_a = log_f_y - log_f_x

            if not self.symmetric:
                log_a = log_a + self.q(y).log_prob(x) - self.q(x).log_prob(y)

            a = log_a.exp()

            # u in [0; 1]
            u = torch.rand(a.shape).to(a)

            # if u < a, x <- y
            # else x <- x
            mask = u < a

            x = torch.where(mask.unsqueeze(-1), y, x)
            log_f_x = torch.where(mask, log_f_y, log_f_x)

            yield x

    def __call__(self, stop: int, burn: int = 0, step: int = 1) -> Iterable[Tensor]:
        return islice(self, burn, stop, step)
