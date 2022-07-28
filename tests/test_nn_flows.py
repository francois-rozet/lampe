r"""Tests for the lampe.nn.flows module."""

import pytest
import torch

from lampe.nn.flows import *
from torch import randn
from torch.distributions import Transform


def test_MaskedAutoregressiveTransform():
    # Without context
    mat = MaskedAutoregressiveTransform(3)
    t = mat()

    assert isinstance(t, Transform)

    x = randn(3)
    z = t(x)

    assert z.shape == x.shape
    assert z.requires_grad
    assert torch.allclose(t.inv(z), x, atol=1e-5)

    # With context
    mat = MaskedAutoregressiveTransform(3, 5)

    x, y = randn(256, 3), randn(5)
    t = mat(y)
    z = t(x)

    assert z.shape == x.shape
    assert z.requires_grad
    assert torch.allclose(t.inv(z), x, atol=1e-5)

    # Passes

    ## Autoregressive
    t = MaskedAutoregressiveTransform(7)()

    x = randn(7)
    J = torch.autograd.functional.jacobian(t, x)

    assert (torch.triu(J, diagonal=1) == 0).all()

    ## Coupling
    t = MaskedAutoregressiveTransform(7, passes=2)()

    x = randn(7)
    J = torch.autograd.functional.jacobian(t, x)

    assert (torch.triu(J, diagonal=1) == 0).all()
    assert (torch.tril(J[:4, :4], diagonal=-1) == 0).all()
    assert (torch.tril(J[4:, 4:], diagonal=-1) == 0).all()


def test_NeuralAutoregressiveTransform():
    # Without context
    nat = NeuralAutoregressiveTransform(3)
    t = nat()

    assert isinstance(t, Transform)

    x = randn(3)
    z = t(x)

    assert z.shape == x.shape
    assert z.requires_grad
    assert torch.allclose(t.inv(z), x, atol=1e-5)

    # With context
    nat = NeuralAutoregressiveTransform(3, 5)

    x, y = randn(256, 3), randn(5)
    t = nat(y)
    z = t(x)

    assert z.shape == x.shape
    assert z.requires_grad
    assert torch.allclose(t.inv(z), x, atol=1e-5)

    # Passes

    ## Autoregressive
    t = NeuralAutoregressiveTransform(7)()

    x = randn(7)
    J = torch.autograd.functional.jacobian(t, x)

    assert (torch.triu(J, diagonal=1) == 0).all()

    ## Coupling
    t = NeuralAutoregressiveTransform(7, passes=2)()

    x = randn(7)
    J = torch.autograd.functional.jacobian(t, x)

    assert (torch.triu(J, diagonal=1) == 0).all()
    assert (torch.tril(J[:4, :4], diagonal=-1) == 0).all()
    assert (torch.tril(J[4:, 4:], diagonal=-1) == 0).all()


def test_MAF():
    flow = MAF(3, 5)

    x, y = randn(256, 3), randn(5)
    log_p = flow(y).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    z = flow(y).sample((32,))

    assert z.shape == (32, 3)


def test_NSF():
    flow = NSF(3, 5)

    x, y = randn(256, 3), randn(5)
    log_p = flow(y).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    z = flow(y).sample((32,))

    assert z.shape == (32, 3)


def test_NAF():
    flow = NAF(3, 5)

    x, y = randn(256, 3), randn(5)
    log_p = flow(y).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    z = flow(y).sample((32,))

    assert z.shape == (32, 3)
