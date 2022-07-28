r"""Tests for the lampe.distributions module."""

import pytest
import torch

from lampe.distributions import *


def test_distributions():
    ds = [
        NormalizingFlow([ExpTransform()], Gamma(2, 1)),
        Joint(Uniform(0, 1), Normal(0, 1)),
        GeneralizedNormal(2),
        DiagNormal(torch.zeros(2), torch.ones(2)),
        BoxUniform(-torch.ones(2), torch.ones(2)),
        TransformedUniform(ExpTransform(), -1, 1),
        Truncated(Normal(0, 1), 1, 2),
        Sort(Normal(0, 1), 3),
        TopK(Normal(0, 1), 2, 3),
        Minimum(Normal(0, 1), 3),
        Maximum(Normal(0, 1), 3),
    ]

    shape = (2**18,)

    for d in ds:
        assert d.batch_shape == (), d

        # Shapes
        x = d.sample(shape)

        assert x.shape == shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == shape, d

        # Expectation
        lower, upper = x.min(dim=0).values, x.max(dim=0).values
        width = upper - lower

        x = Uniform(lower - width / 2, upper + width / 2).sample(shape)

        p = d.log_prob(x).exp().nanmean() * (2 * width).prod()

        assert (0.9 <= p) and (p <= 1.1), d

        # Expand
        d = d.expand((32,))

        assert d.batch_shape == (32,), d

        x = d.sample()

        assert x.shape == d.batch_shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == d.batch_shape, d


def test_transforms():
    ts = [
        CosTransform(),
        SinTransform(),
        MonotonicAffineTransform(torch.tensor(42.), torch.tensor(-0.69)),
        MonotonicRQSTransform(*map(torch.randn, (8, 8, 7))),
        MonotonicTransform(lambda x: x**3),
    ]

    for t in ts:
        if hasattr(t.domain, 'lower_bound'):
            x = torch.linspace(t.domain.lower_bound, t.domain.upper_bound, 256)
        else:
            x = torch.linspace(-1e1, 1e1, 256)

        y = t(x)

        assert x.shape == y.shape, t

        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-5), t

        # Jacobian
        J = torch.autograd.functional.jacobian(t, x)

        assert (torch.triu(J, diagonal=1) == 0).all()
        assert (torch.tril(J, diagonal=-1) == 0).all()

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(ladj, t.log_abs_det_jacobian(x, y), atol=1e-5), t


def test_PermutationTransform():
    t = PermutationTransform(torch.randperm(8))

    x = torch.randn(256, 8)
    y = t(x)

    assert x.shape == y.shape

    match = x[:, :, None] == y[:, None, :]

    assert (match.sum(dim=-1) == 1).all()
    assert (match.sum(dim=-2) == 1).all()

    z = t.inv(y)

    assert x.shape == z.shape
    assert (x == z).all()
