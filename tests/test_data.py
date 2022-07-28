r"""Tests for the lampe.data module."""

import h5py
import math
import numpy as np
import pytest
import torch

from lampe.data import *


def test_JointLoader():
    prior = torch.distributions.Normal(torch.zeros(3), torch.ones(3))

    count = 0
    def simulator(theta):
        nonlocal count

        count += 1

        assert torch.is_tensor(theta)
        assert theta.dtype == torch.float

        return torch.repeat_interleave(theta, 2, dim=-1)

    # Non-vectorized
    loader = JointLoader(prior, simulator, batch_size=4, vectorized=False)

    assert isinstance(loader, torch.utils.data.DataLoader)

    it = iter(loader)
    theta1, x1 = next(it)
    theta2, x2 = next(it)

    assert count == 8

    assert torch.is_tensor(theta1) and torch.is_tensor(x1)
    assert theta1.shape == (4, 3) and x1.shape == (4, 6)
    assert theta1.dtype == x1.dtype == torch.float
    assert (theta1 != theta2).all()

    # Vectorized
    loader = JointLoader(prior, simulator, batch_size=5, vectorized=True)

    it = iter(loader)
    theta1, x1 = next(it)
    theta2, x2 = next(it)

    assert count == 10

    assert torch.is_tensor(theta1) and torch.is_tensor(x1)
    assert theta1.shape == (5, 3) and x1.shape == (5, 6)
    assert theta1.dtype == x1.dtype == torch.float
    assert (theta1 != theta2).all()

    # NumPy
    def simulator(theta):
        assert not torch.is_tensor(theta)
        assert theta.dtype == np.double

        return np.repeat(theta, 3, axis=-1)

    loader = JointLoader(prior, simulator, batch_size=6, numpy=True)

    it = iter(loader)
    theta1, x1 = next(it)

    assert torch.is_tensor(theta1) and torch.is_tensor(x1)
    assert theta1.shape == (6, 3) and x1.shape == (6, 9)
    assert theta1.dtype == x1.dtype == torch.float


def test_H5Dataset(tmp_path):
    prior = torch.distributions.Normal(torch.zeros(3), torch.ones(3))
    simulator = lambda theta: torch.repeat_interleave(theta, 2, dim=-1)

    theta = prior.sample((4096,))
    x = simulator(theta)

    # Store
    pairs = list(zip(theta.split(256), x.split(256)))

    H5Dataset.store(pairs, tmp_path / 'data_1.h5', size=4096)
    H5Dataset.store(iter(pairs), tmp_path / 'data_2.h5', size=4096)
    H5Dataset.store(pairs, tmp_path / 'data_3.h5', size=256)

    with pytest.raises(FileExistsError):
        H5Dataset.store(pairs, tmp_path / 'data_1.h5', size=4096)

    # Load
    for file in tmp_path.glob('data_*.h5'):
        dataset = H5Dataset(file)

        assert len(dataset) in {256, 4096}

        ## __getitem__
        for i in map(lambda x: 2**x, range(int(math.log2(len(dataset))))):
            theta_i, x_i = dataset[i]

            assert theta_i.shape == (3,) and x_i.shape == (6,)
            assert (theta_i == theta[i]).all()
            assert (x_i == x[i]).all()

        ## __item__
        for i, (theta_i, x_i) in enumerate(dataset):
            assert (theta_i == theta[i]).all()
            assert (x_i == x[i]).all()

        assert i == len(dataset) - 1

    ## Shuffle
    dataset = H5Dataset(tmp_path / 'data_1.h5', batch_size=256, shuffle=True)

    it = iter(dataset)
    theta1, x1 = next(it)
    theta2, x2 = next(it)

    assert theta1.shape == (256, 3) and x1.shape == (256, 6)
    assert theta2.shape == (256, 3) and x2.shape == (256, 6)

    theta3, x3 = torch.cat((theta1, theta2)), torch.cat((x1, x2))

    match = (theta3[:, None, :] == theta).all(dim=-1)

    assert (match.sum(dim=0) <= 1).all()
    assert (match.sum(dim=-1) == 1).all()

    _, index = torch.nonzero(match, as_tuple=True)

    assert (x3 == x[index]).all()
