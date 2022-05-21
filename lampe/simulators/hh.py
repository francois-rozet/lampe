r"""Hodgkin-Huxley (HH) benchmark.

HH is a widespread non-linear mechanistic model of neural dynamics.

References:
    A quantitative description of membrane current and its application to conduction and excitation in nerve
    (Hodgkin et al., 1952)
    https://link.springer.com/article/10.1007/BF02459568

    Training deep neural density estimators to identify mechanistic models of neural dynamics
    (Gonçalves et al., 2020)
    https://elifesciences.org/articles/56261

Shapes:
    theta: :math:`(8,)`.
    x: :math:`(7,)`.
"""

import numpy as np
import torch

from numpy import ndarray as Array
from torch import Tensor, BoolTensor
from typing import *

from . import Simulator


LABELS = [
    f'${l}$' for l in [
        r'g_{\mathrm{Na}}', r'g_{\mathrm{K}}', r'g_{\mathrm{M}}', 'g_l',
        r'\tau_{\max}', 'V_t', r'\sigma', 'E_l',
    ]
]

LOWER, UPPER = torch.tensor([
    [0.5, 80.],     # g_Na [mS/cm^2]
    [1e-4, 15.],    # g_K [mS/cm^2]
    [1e-4, .6],     # g_M [mS/cm^2]
    [1e-4, .6],     # g_l [mS/cm^2]
    [50., 3000.],   # tau_max [ms]
    [-90., -40.],   # V_t [mV]
    [1e-4, .15],    # sigma [uA/cm^2]
    [-100., -35.],  # E_l [mV]
]).t()


class HH(Simulator):
    r"""Creates an Hodgkin-Huxley (HH) simulator.

    Arguments:
        summary: Whether voltage traces are converted to summary statistics or not.
        seed: A random number generator seed.
        kwargs: Simulator settings and constants (e.g. duration, inital voltage, ...).
    """

    def __init__(self, summary: bool = True, seed: int = None, **kwargs):
        super().__init__()

        # Constants
        default = {
            'duration': 80.,  # s
            'time_step': 0.02,  # s
            'padding': 10.,  # s
            'initial_voltage': -70., # mV
            'current': 5e-4 / (np.pi * 7e-3 ** 2),  # uA / cm^2
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        # Summary statistics
        self.summary = summary

        # RNG
        self.rng = np.random.default_rng(seed)

    def __call__(self, theta: Array) -> Array:
        x = voltage_trace(theta, self.constants, self.rng)

        if self.summary:
            x = summarize(x, self.constants)

        return x


def voltage_trace(
    theta: Array,
    constants: Dict[str, float],
    rng: np.random.Generator,
) -> Array:
    r"""Simulates an Hodgkin-Huxley voltage trace.

    References:
        https://github.com/mackelab/sbi/blob/main/examples/HH_helper_functions.py
    """

    # Parameters
    T = constants['duration']
    dt = constants['time_step']
    pad = constants['padding']
    V_0 = constants['initial_voltage']
    I = constants['current']

    theta = np.expand_dims(theta, axis=0)
    g_Na, g_K, g_M, g_leak, tau_max, V_t, sigma, E_leak = [
        theta[..., i] for i in range(8)
    ]

    C = 1.  # uF/cm^2
    E_Na = 53.  # mV
    E_K = -107.  # mV

    # Kinetics
    exp = np.exp
    efun = lambda x: np.where(
        np.abs(x) < 1e-4,
        1 - x / 2,
        x / (exp(x) - 1)
    )

    alpha_n = lambda x: 0.032 * efun(-0.2 * (x - 15)) / 0.2
    beta_n = lambda x: 0.5 * exp(-(x - 10) / 40)
    tau_n = lambda x: 1 / (alpha_n(x) + beta_n(x))
    n_inf = lambda x: alpha_n(x) / (alpha_n(x) + beta_n(x))

    alpha_m = lambda x: 0.32 * efun(-0.25 * (x - 13)) / 0.25
    beta_m = lambda x: 0.28 * efun(0.2 * (x - 40)) / 0.2
    tau_m = lambda x: 1 / (alpha_m(x) + beta_m(x))
    m_inf = lambda x: alpha_m(x) / (alpha_m(x) + beta_m(x))

    alpha_h = lambda x: 0.128 * exp(-(x - 17) / 18)
    beta_h = lambda x: 4 / (1 + exp(-0.2 * (x - 40)))
    tau_h = lambda x: 1 / (alpha_h(x) + beta_h(x))
    h_inf = lambda x: alpha_h(x) / (alpha_h(x) + beta_h(x))

    tau_p = lambda x: tau_max / (3.3 * exp(0.05 * (x + 35)) + exp(-0.05 * (x + 35)))
    p_inf = lambda x: 1 / (1 + exp(-0.1 * (x + 35)))

    # Iterations
    voltages = []

    V = np.full_like(V_t, V_0)
    V_rel = V - V_t

    n = n_inf(V_rel)
    m = m_inf(V_rel)
    h = h_inf(V_rel)
    p = p_inf(V)

    for t in np.arange(0, T, dt):
        tau_V = C / (
            g_Na * m**3 * h
            + g_K * n**4
            + g_M * p
            + g_leak
        )

        V_inf = tau_V * (
            E_Na * g_Na * m**3 * h
            + E_K * g_K * n**4
            + E_K * g_M * p
            + E_leak * g_leak
            + I * (pad <= t < T - pad)
            + sigma * rng.standard_normal(V.shape) / dt**0.5
        ) / C

        V = V_inf + (V - V_inf) * exp(-dt / tau_V)
        V_rel = V - V_t

        n = n_inf(V_rel) + (n - n_inf(V_rel)) * exp(-dt / tau_n(V_rel))
        m = m_inf(V_rel) + (m - m_inf(V_rel)) * exp(-dt / tau_m(V_rel))
        h = h_inf(V_rel) + (h - h_inf(V_rel)) * exp(-dt / tau_h(V_rel))
        p = p_inf(V) + (p - p_inf(V)) * exp(-dt / tau_p(V))

        voltages.append(V)

    return np.stack(voltages, axis=-1).squeeze(axis=0)


def summarize(x: Array, constants: Dict[str, float]) -> Array:
    r"""Returns summary statistics of a voltage trace."""

    # Constants
    T = constants['duration']
    dt = constants['time_step']
    pad = constants['padding']

    t = np.arange(0, T, dt)

    # Number of spikes
    spikes = np.maximum(x, -10)
    spikes = np.diff(np.sign(np.diff(spikes)))
    spikes = np.sum(spikes < 0, axis=-1)

    # Resting moments
    rest = x[..., (pad / 2 <= t) * (t < pad)]
    rest_mean = np.mean(rest, axis=-1)
    rest_std = np.std(rest, axis=-1)

    # Moments
    x = x[..., (pad <= t) * (t < T - pad)]
    x_mean = np.mean(x, axis=-1)
    x_std = np.std(x, axis=-1)

    z = (x - x_mean[..., None]) / x_std[..., None]

    x_skew = np.mean(z**3, axis=-1)
    x_kurtosis = np.mean(z**4, axis=-1)

    return np.stack([
        spikes,
        rest_mean, rest_std,
        x_mean, x_std, x_skew, x_kurtosis,
    ], axis=-1)
