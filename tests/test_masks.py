r"""Tests for the lampe.masks module."""

import pytest
import torch

from lampe.masks import *


def test_mask2str():
    b = torch.tensor([True, True, False, True, False])

    assert mask2str(b) == '11010'
    assert (str2mask('11010') == b).all()


def test_BernouilliMask():
    p = torch.rand(3)
    d = BernoulliMask(p)

    assert d.event_shape == (3,)

    b = d.sample()

    assert b.shape == (3,)
    assert b.dtype == torch.bool

    b = d.sample((5, 4))

    assert b.shape == (5, 4, 3)

    lp = d.log_prob(b)

    assert lp.shape == (5, 4)


def test_SelectionMask():
    selection = torch.tensor([
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ])
    d = SelectionMask(selection)

    assert d.event_shape == (3,)

    b = d.sample()

    assert b.shape == (3,)
    assert b.dtype == torch.bool

    b = d.sample((5, 4))

    assert b.shape == (5, 4, 3)

    lp = d.log_prob(b)

    assert lp.shape == (5, 4)
