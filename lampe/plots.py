r"""Plotting helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from numpy import ndarray as Array
from typing import *


__all__ = ['nice_rc', 'corner', 'rank_ecdf']


def nice_rc(latex: bool = False) -> Dict[str, Any]:
    r"""Returns a dictionary of runtime configuration (rc) settings for nicer
    :mod:`matplotlib` plots. The settings include 12pt font size, higher DPI,
    tight layout, transparent background, etc.

    Arguments:
        latex: Whether to use LaTeX typesetting or not.

    Example:
        >>> plt.rcParams.update(nice_rc())
    """

    rc = {
        'axes.axisbelow': True,
        'axes.linewidth': .8,
        'figure.autolayout': True,
        'figure.dpi': 150,
        'figure.figsize': (6.4, 4.8),
        'font.size': 12.,
        'legend.fontsize': 'x-small',
        'lines.linewidth': 1.,
        'lines.markersize': 3.,
        'savefig.bbox': 'tight',
        'savefig.transparent': True,
        'xtick.labelsize': 'x-small',
        'xtick.major.width': .8,
        'ytick.labelsize': 'x-small',
        'ytick.major.width': .8,
    }

    if mpl.checkdep_usetex(latex):
        rc.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern'],
            'text.usetex': True,
        })

    return rc


class LinearAlphaColormap(mpl.colors.LinearSegmentedColormap):
    r"""Linear segmented transparency colormap.

    Arguments:
        color: A color.
        levels: A sequence of levels dividing the domain into segments.
        alpha: The transparancy range.
        name: A name for the colormap.
    """

    def __new__(
        self,
        color: Union[str, tuple],
        levels: Array = None,
        alpha: Tuple[float, float] = (0., 1.),
        name: str = None,
    ):
        if name is None:
            if type(color) is str:
                name = f'alpha_{color}'
            else:
                name = f'alpha_{hash(color)}'

        if levels is None:
            levels = [0., 1.]

        levels = np.asarray(levels)
        levels = (levels - levels.min()) / (levels.max() - levels.min())
        alpha = np.linspace(*alpha, len(levels))

        rgb = mpl.colors.to_rgb(color)

        colors = sorted([
            (l, rgb + (a,))
            for l, a in zip(levels, alpha)
        ])

        return mpl.colors.LinearSegmentedColormap.from_list(
            name=name,
            colors=colors,
        )


def gaussian_blur(img: Array, sigma: float = 1.) -> Array:
    r"""Applies a Gaussian blur to an image.

    Arguments:
        img: An image array.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The blurred image.

    Example:
        >>> img = np.random.rand(128, 128)
        >>> gaussian_blur(img, sigma=2.)
        array([...])
    """

    size = 2 * int(3 * sigma) + 1

    k = np.arange(size) - size / 2
    k = np.exp(-k ** 2 / (2 * sigma ** 2))
    k = k / np.sum(k)

    smooth = lambda x: np.convolve(x, k, mode='same')

    for i in range(len(img.shape)):
        img = np.apply_along_axis(smooth, i, img)

    return img


def credible_levels(hist: Array, creds: Array) -> Array:
    r"""Returns the levels of credibility region contours.

    Arguments:
        hist: An histogram.
        creds: The region credibilities.
    """

    x = np.sort(hist, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, creds * cdf[-1])

    return x[idx]


def corner(
    data: Array,
    bins: Union[int, List[int]] = 100,
    bounds: Tuple[Array, Array] = None,
    creds: Array = [.6827, .9545, .9973],
    color: Union[str, tuple] = None,
    alpha: Tuple[float, float] = (0., .5),
    legend: str = None,
    labels: List[str] = None,
    markers: List[Array] = [],
    smooth: float = 0,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""Displays each 1 or 2-d projection of multi-dimensional data, as a triangular
    matrix of histograms, known as corner plot. For 2-d histograms, highest density
    credibility regions are delimited.

    Arguments:
        data: Multi-dimensional data, either as a table or as a matrix of histograms.
        bins: The number(s) of bins per dimension.
        bounds: A tuple of lower and upper domain bounds. If :py:`None`, inferred from data.
        creds: The region credibilities (in :math:`[0, 1]`) to delimit.
        color: A color for histograms.
        alpha: A transparency range.
        legend: A legend.
        labels: The dimension labels.
        markers: A list of points to mark on the histograms.
        smooth: The standard deviation of the smoothing kernels.
        figure: A corner plot over which to draw the new one.
        kwargs: Keyword arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The figure instance for the corner plot.

    Example:
        >>> data = np.random.randn(2**16, 3)
        >>> labels = [r'$\alpha$', r'$\beta$', r'$\gamma$']
        >>> corner(data, bins=42, labels=labels, figsize=(4.8, 4.8))

    .. image:: ../static/images/corner.png
        :align: center
        :width: 600
    """

    # Histograms
    data = np.asarray(data)

    if np.isscalar(data[0, 0]):
        D = data.shape[-1]
        data = data.reshape(-1, D)

        if type(bins) is int:
            bins = [bins] * D

        if bounds is None:
            lower, upper = data.min(axis=0), data.max(axis=0)
        else:
            lower, upper = map(np.asarray, bounds)

        bins = [
            np.histogram_bin_edges(data, bins[i], range=(lower[i], upper[i]))
            for i in range(D)
        ]

        hists = np.ndarray((D, D), dtype=object)

        for i in range(D):
            for j in range(i + 1):
                if i == j:
                    hist, _ = np.histogram(
                        data[..., i],
                        bins=bins[i],
                        density=True,
                    )
                else:
                    hist, _, _ = np.histogram2d(
                        data[..., i], data[..., j],
                        bins=(bins[i], bins[j]),
                        density=True,
                    )

                hists[i, j] = hist
    else:
        D = len(data)

        if bounds is None:
            lower, upper = np.zeros(D), np.ones(D)
        else:
            lower, upper = map(np.asarray, bounds)

        bins = [None] * D
        for i in range(D):
            if data[i, i] is not None:
                bins[i] = np.linspace(lower[i], upper[i], len(data[i, i]) + 1)

        hists = data

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (6.4, 6.4))

        figure, axes = plt.subplots(
            D, D,
            squeeze=False,
            sharex='col',
            gridspec_kw={'wspace': 0., 'hspace': 0.},
            **kwargs,
        )
        new = True
    else:
        axes = np.asarray(figure.axes).reshape(D, D)
        new = False

    # Legend

    ## Color
    lines = axes[0, -1].plot([], [], color=color, label=legend)
    color = lines[-1].get_color()

    handles, texts = axes[0, -1].get_legend_handles_labels()

    ## Quantiles
    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)

    cmap = LinearAlphaColormap('black', levels=creds, alpha=alpha)

    levels = (creds - creds.min()) / (creds.max() - creds.min())
    levels = (levels[:-1] + levels[1:]) / 2

    for c, l in zip(creds[:-1], levels):
        handles.append(mpl.patches.Patch(color=cmap(l), linewidth=0))
        texts.append(r'${:.1f}\,\%$'.format(c * 100))

    ## Update
    if not new:
        figure.legends.clear()

    figure.legend(handles, texts, loc='upper right', bbox_to_anchor=(0.975, 0.975), frameon=False)

    # Plot
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            hist = hists[i, j]

            if j > i:
                ax.axis('off')
                continue

            if hist is None:
                continue

            if smooth > 0:
                hist = gaussian_blur(hist, smooth)

            ## Draw
            x, y = bins[j], bins[i]
            x = (x[1:] + x[:-1]) / 2
            y = (y[1:] + y[:-1]) / 2

            if i == j:
                ax.plot(x, hist, color=color)

                _, top = ax.get_ylim()
                bottom = 0.
                top = max(top, hist.max() * 1.0625)

                ax.set_xlim(left=bins[i][0], right=bins[i][-1])
                ax.set_ylim(bottom=bottom, top=top)
            else:
                levels = np.unique(credible_levels(hist, creds))

                cf = ax.contourf(
                    x, y, hist,
                    levels=levels,
                    cmap=LinearAlphaColormap(color, levels, alpha=alpha),
                )
                ax.contour(cf, colors=color)

                if j > 0:
                    ax.sharey(axes[i, j - 1])
                else:
                    ax.set_ylim(bottom=bins[i][0], top=bins[i][-1])

            ## Markers
            for marker in map(np.asarray, markers):
                ax.axvline(
                    marker[j],
                    color='k',
                    linestyle='--',
                    zorder=420,
                )

                if i != j:
                    ax.axhline(
                        marker[i],
                        color='k',
                        linestyle='--',
                        zorder=420,
                    )

                    ax.plot(
                        marker[j], marker[i],
                        color='k',
                        marker='s',
                        zorder=420,
                    )

            ## Ticks
            if i == D - 1:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45.,
                    horizontalalignment='right',
                    rotation_mode='anchor',
                )
            else:
                ax.xaxis.set_ticks_position('none')

            if i == j:
                ax.set_yticks([])
            elif j == 0:
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
            else:
                ax.yaxis.set_ticks_position('none')

            ## Labels
            if labels is not None:
                if i == D - 1:
                    ax.set_xlabel(labels[j])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])

            ax.label_outer()

    figure.align_labels()

    return figure


def rank_ecdf(
    ranks: Array,
    color: Union[str, tuple] = None,
    legend: str = None,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""Draws the empirical cumulative distribution function (ECDF) of a rank
    statistic :math:`r \in [0, 1]`.

    Arguments:
        ranks: Samples of the rank statistic.
        color: A color.
        legend: A legend.
        figure: A ECDF plot over which to draw the new one.
        kwargs: Keyword arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The figure instance for the ECDF plot.

    Example:
        >>> ranks = np.random.rand(2**12)**2
        >>> rank_ecdf(ranks)

    .. image:: ../static/images/rank_ecdf.png
        :align: center
        :width: 400
    """

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (3.2, 3.2))

        figure, ax = plt.subplots(**kwargs)
        new = True
    else:
        ax = figure.axes.squeeze()
        new = False

    # ECDF
    ranks = np.sort(np.asarray(ranks))
    ranks = np.hstack([0, ranks, 1])
    ecdf = np.linspace(0, 1, len(ranks))

    # Plot
    if new:
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')

    ax.plot(ranks, ecdf, color=color, label=legend)

    ax.grid()
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\mathrm{ECDF}(r)$')

    if legend is not None:
        ax.legend(loc='upper left')

    return figure
