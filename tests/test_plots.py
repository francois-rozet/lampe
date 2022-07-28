r"""Tests for the lampe.plots module."""

import pytest
import numpy as np

from lampe.plots import *


def test_nice_rc():
    assert type(nice_rc()) is dict


def test_corner():
    # Samples
    samples = np.random.randn(2**16, 3)
    labels = ['a', 'b', 'c']
    figure = corner(samples, bins=42, labels=labels, figsize=(4.8, 4.8))

    assert np.asarray(figure.axes).size == 3**2

    # Histograms
    hists = [[None] * 3 for _ in range(3)]

    for i in range(3):
        hists[i][i] = np.random.rand(10 + 2 * i)

        for j in range(i):
            hists[i][j] = np.random.rand(10 + 2 * i, 10 + 2 * j)

    new = corner(hists, bounds=(-np.ones(3), np.ones(3)), figure=figure)

    assert figure is new

    # Mark points
    mark_point(figure, [0.5, 0.3, -0.7], color='black')


def test_rank_ecdf():
    ranks = np.random.rand(1024)**2
    figure = rank_ecdf(ranks)
