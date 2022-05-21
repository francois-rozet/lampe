r"""Gravitational waves (GW) benchmark.

The GW simulator computes the gravitational waves emitted by precessing quasi-circular
binary black hole (BBH) systems, and project them onto LIGO detectors (H1 and L1).
It assumes stationary Gaussian noise with respect to the detectors' noise spectral
densities, estimated from 1024 seconds of detector data prior to GW150914. The
waveforms are compressed to a reduced-order basis corresponding to the first 128
components of a singular value decomposition (SVD).

References:
    Observation of Gravitational Waves from a Binary Black Hole Merger
    (Abbott et al., 2016)
    https://arxiv.org/abs/1602.03837

    Complete parameter inference for GW150914 using deep learning
    (Green et al., 2021)
    https://arxiv.org/abs/2008.03312

Shapes:
    theta: :math:`(15,)`.
    x: :math:`(2, 256)`.
"""

import numpy as np
import os
import torch

try:
    os.environ['GWPY_RCPARAMS'] = '0'

    from gwpy.timeseries import TimeSeries
    from lal import MSUN_SI
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from pycbc.catalog import Merger
    from pycbc.detector import Detector
    from pycbc.psd import welch
    from pycbc.waveform import get_fd_waveform
except Exception as e:
    print(f"ImportWarning: {e}. 'GW' requires")
    print("  pip install gwpy pycbc")

from numpy import ndarray as Array
from torch import Tensor, BoolTensor
from tqdm import tqdm
from typing import *

from . import Simulator
from ..distributions import (
    Distribution,
    Joint,
    Uniform,
    Sort,
    Maximum,
    Minimum,
    TransformedUniform,
    CosTransform,
    SinTransform,
)
from ..utils import cache, vectorize


LABELS = [
    f'${l}$' for l in [
        'm_1', 'm_2', r'\phi_c', 't_c', 'd_L',
        'a_1', 'a_2', r'\theta_1', r'\theta_2', r'\phi_{12}', r'\phi_{JL}',
        r'\theta_{JN}', r'\psi', r'\alpha', r'\delta',
    ]
]

LOWER, UPPER = torch.tensor([
    [10., 80.],               # primary mass [solar masses]
    [10., 80.],               # secondary mass [solar masses]
    [0., 2 * np.pi],          # coalesence phase [rad]
    [-0.1, 0.1],              # coalescence time [s]
    [100., 1000.],            # luminosity distance [megaparsec]
    [0., 0.88],               # a_1 [/]
    [0., 0.88],               # a_2 [/]
    [0., np.pi],              # tilt_1 [rad]
    [0., np.pi],              # tilt_2 [rad]
    [0., 2 * np.pi],          # phi_12 [rad]
    [0., 2 * np.pi],          # phi_jl [rad]
    [0., np.pi],              # theta_jn [rad]
    [0., np.pi],              # polarization [rad]
    [0., 2 * np.pi],          # right ascension [rad]
    [-np.pi / 2, np.pi / 2],  # declination [rad]
]).t()


def build_prior(b: BoolTensor = None) -> Distribution:
    r"""Returns a prior distribution :math:`p(\theta)` for BBH systems.

    Arguments:
        b: An optional binary mask :math:`b`, with shape :math:`(D,)`.
    """

    if b is None:
        b = [True] * 15

    marginals = []

    if b[0] or b[1]:
        base = Uniform(LOWER[0], UPPER[0])

        if b[0] and b[1]:
            law = Sort(base, n=2, descending=True)
        elif b[0]:
            law = Maximum(base, n=2)
        elif b[1]:
            law = Minimum(base, n=2)

        marginals.append(law)

    for i in range(2, len(b)):
        if b[i]:
            if i in [7, 8, 11]:  # [tilt_1, tilt_2, theta_jn]
                m = TransformedUniform(CosTransform(), LOWER[i], UPPER[i])
            elif i == 14:  # declination
                m = TransformedUniform(SinTransform(), LOWER[i], UPPER[i])
            else:
                m = Uniform(LOWER[i], UPPER[i])

            marginals.append(m)

    return Joint(marginals)


class GW(Simulator):
    r"""Creates a gravitational waves (GW) simulator.

    Arguments:
        reduced_basis: Whether waveform are compressed to a reduced basis or not.
        noisy: Whether noise is added to waveforms or not.
        seed: A random number generator seed.
        kwargs: Simulator settings and constants (e.g. event, approximant, ...).
    """

    def __init__(
        self,
        reduced_basis: bool = True,
        noisy: bool = True,
        seed: int = None,
        **kwargs,
    ):
        super().__init__()

        # Constants
        default = {
            'event': 'GW150914',
            'detectors': ('H1', 'L1'),
            'approximant': 'IMRPhenomPv2',
            'duration': 4.,  # s
            'buffer': 2.,  # s
            'segment': 1024.,  # s
            'sample_rate': 1024.,  # Hz
            'f_ref': 40.,  # Hz
            'f_lower': 20.,  # Hz
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        self.nsd = event_nsd(**self.constants)
        self.nsd = crop_dft(self.nsd, **self.constants)

        # Reduced SVD basis
        if reduced_basis:
            self.basis = svd_basis(**self.constants)
        else:
            self.basis = None

        # RNG
        self.noisy = noisy
        self.rng = np.random.default_rng(seed)

    def __call__(self, theta: Array) -> Array:
        x = gravitational_waveform(theta, **self.constants)
        x = self.process(x)

        if self.noisy:
            x = x + self.rng.standard_normal(x.shape)

        return x

    def process(self, x: Array) -> Array:
        r"""Processes waveforms into network-friendly inputs."""

        x = crop_dft(x, **self.constants)
        x = x / self.nsd

        if self.basis is not None:
            x = x @ self.basis

        return x.view(np.float64)


@cache
def ligo_detector(name: str):
    r"""Fetches LIGO detector."""

    return Detector(name)


@cache
def event_gps(event: str = 'GW150914') -> float:
    r"""Fetches event's GPS time."""

    return Merger(event).data['GPS']


@cache
def tukey_window(
    duration: int,  # s
    sample_rate: float,  # Hz
    roll_off: float = 0.4,  # s
) -> Array:
    r"""Returns a tukey window.

    References:
        https://en.wikipedia.org/wiki/Window_function
    """

    from scipy.signal import tukey

    length = int(duration * sample_rate)
    alpha = 2 * roll_off / duration

    return tukey(length, alpha)


@cache(persist=True)
def event_nsd(
    event: str,
    detectors: Tuple[str, ...],
    duration: float,  # s
    segment: float,  # s
    **absorb,
) -> Array:
    r"""Fetches event's noise spectral density (NSD).

    Wikipedia:
        https://en.wikipedia.org/wiki/Noise_spectral_density

    References:
        https://github.com/gwastro/pycbc/blob/master/pycbc/noise/gaussian.py#L35
    """

    time = event_gps(event) - duration

    nsds = []

    for det in detectors:
        strain = TimeSeries.fetch_open_data(det, time - segment, time, cache=True).to_pycbc(copy=False)

        win = tukey_window(duration, strain.sample_rate)
        win_factor = np.sum(win ** 2) / len(win)
        psd = welch(strain, len(win), len(win), window=win, avg_method='median') * win_factor
        nsd = 0.5 * np.sqrt(psd.data / psd.delta_f)

        nsds.append(nsd)

    return np.stack(nsds)


@cache(persist=True)
def event_dft(
    event: str,
    detectors: Tuple[str, ...],
    duration: float,  # s
    buffer: float,  # s
    **absorb,
) -> Array:
    r"""Fetches event's discrete fourier transform (DFT)."""

    time = event_gps(event) + buffer

    dfts = []

    for det in detectors:
        strain = TimeSeries.fetch_open_data(det, time - duration, time, cache=True).to_pycbc(copy=False)

        win = tukey_window(duration, strain.sample_rate)
        dft = (strain * win).to_frequencyseries().cyclic_time_shift(buffer)

        dfts.append(dft.data)

    return np.stack(dfts)


@vectorize(otypes=[float] * 7)
def lal_spins(*args) -> Tuple[float, ...]:
    r"""Converts LALInference geometric parameters to LALSimulation spins."""

    return tuple(SimInspiralTransformPrecessingNewInitialConditions(*args))


@vectorize(otypes=[Array, Array])
def plus_cross(**kwargs) -> Tuple[Array, Array]:
    r"""Simulates frequency-domain plus and cross polarizations of gravitational wave."""

    hp, hc = get_fd_waveform(**kwargs)
    return hp.numpy(), hc.numpy()


def gravitational_waveform(
    theta: Array,
    event: str,
    detectors: Tuple[str, ...],
    approximant: str,
    duration: float,  # s
    sample_rate: float,  # Hz
    f_ref: float,  # Hz
    f_lower: float,  # Hz
    **absorb,
) -> Array:
    r"""Simulates a frequency-domain gravitational wave projected onto LIGO detectors.

    References:
        http://pycbc.org/pycbc/latest/html/waveform.html
        http://pycbc.org/pycbc/latest/html/detector.html
    """

    shape = theta.shape[:-1]
    theta = theta.reshape(-1, theta.shape[-1])

    # Parameters
    m_1, m_2, phi_c, t_c, d_L, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, psi, alpha, delta = [
        theta[..., i] for i in range(15)
    ]

    iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = lal_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
        m_1 * MSUN_SI, m_2 * MSUN_SI, f_ref, phi_c
    )

    # Gravitational wave
    hp, hc = plus_cross(
        ## Constants
        approximant=approximant,
        delta_f=1/duration,
        f_ref=f_ref,
        f_lower=f_lower,
        f_final=sample_rate/2,
        ## Variables
        mass1=m_1,
        mass2=m_2,
        coa_phase=phi_c,
        distance=d_L,
        inclination=iota,
        spin1x=spin1x,
        spin2x=spin2x,
        spin1y=spin1y,
        spin2y=spin2y,
        spin1z=spin1z,
        spin2z=spin2z,
    )

    hp, hc = np.stack(hp), np.stack(hc)

    # Projection on detectors
    time = event_gps(event)
    length = int(duration * sample_rate / 2) + 1
    angular_speeds = -1j * 2 * np.pi * np.arange(length) / duration

    strains = []

    for name in detectors:
        det = ligo_detector(name)

        ## Noiseless strain
        fp, fc = det.antenna_pattern(alpha, delta, psi, time)
        s = fp[..., None] * hp + fc[..., None] * hc

        ## Cyclic time shift
        dt = det.time_delay_from_earth_center(alpha, delta, time) + t_c
        s = s * np.exp(dt[..., None] * angular_speeds)

        strains.append(s)

    strains = np.stack(strains, axis=-2)
    strains = strains.reshape(shape + strains.shape[1:])

    return strains


def crop_dft(
    dft: Array,
    duration: float,  # s
    sample_rate: float,  # Hz
    f_lower: float,  # Hz
    **absorb,
) -> Array:
    r"""Crops low and high frequencies of discrete fourier transform (DFT)."""

    return dft[..., int(duration * f_lower):int(duration * sample_rate / 2) + 1]


@cache(persist=True)
def svd_basis(
    n_components: int = 2**7,  # 128
    n_samples: int = 2**15,  # 32768
    batch_size: int = 2**10,  # 1024
    seed: int = 0,
    **kwargs,
) -> Array:
    r"""Builds Singular Value Decompostition (SVD) basis."""

    prior = gw_prior()
    simulator = GW(reduced_basis=False, noisy=False, **kwargs)

    print("Generating samples...")

    xs = []

    for _ in tqdm(range(n_samples // batch_size), unit='sample', unit_scale=batch_size):
        theta = prior.sample((batch_size,))
        theta[..., 4] = LOWER[4]  # fixed luminosity distance
        theta = theta.numpy().astype(np.float64)

        xs.append(simulator(theta))

    x = np.stack(xs).view(np.complex128)
    x = x.reshape(-1, x.shape[-1])

    print("Computing SVD basis...")

    try:
        from sklearn.utils.extmath import randomized_svd

        _, _, Vh = randomized_svd(
            x,
            n_components=n_components,
            n_oversamples=n_components,
            random_state=seed,
        )
    except ImportError as e:
        _, _, Vh = np.linalg.svd(x, full_matrices=False)
        Vh = Vh[:n_components]

    V = Vh.T.conj()

    print("Done!")

    return V
