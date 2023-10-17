import itertools
import typing
from typing import Callable, Collection, Optional
from pandera.typing import Series as SeriesT
from pandera.typing import DataFrame as DataFrameT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import logging

from jttools.statistics import OLS

DFOrSeries = pd.DataFrame | pd.Series


def minminmaxmax(x, y):
    nn = min([min(x), min(y)])
    xx = max([max(x), max(y)])
    return nn, xx

def hxbin(x, y, ax=None, **kwarghole):
    if ax is None:
        plt.figure(figsize=(4, 4))
    else:
        plt.sca(ax)
    plt.hexbin(x, y, gridsize=40, bins='log')



def plt_labels(xlab='', ylab='', title='', ax=None):
    """Add axes labels and figure title to the current matplotlib Figure"""
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)


def hixbin(x, y, ax=None, **hxbn_kwargs):
    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
    kwargs = dict({'bins': 'log', 'gridsize': 40}, **hxbn_kwargs)
    ax.hexbin(x, y, **kwargs)


def prep_data_for_upsetplot(bool_table:pd.DataFrame):
    """UpSetPlot (for displaying intersection sizes) requires data to be
    formated in a very specific way. This does that.
    Expects a table of bools with columns being the samples"""

    bool_counts = {}

    # get table with each combination of bools, and their occurance
    for row in bool_table.values:
        # row = ''.join([str(int(b)) for b in row])
        row = tuple(row)
        try:
            bool_counts[row] += 1
        except KeyError:
            bool_counts[row] = 1

    b = (True, False)
    multindx = pd.MultiIndex.from_product([b for _ in bool_table.columns],
                                          names=bool_table.columns)

    # duplicates the data in 2 dimensions, and the sample names are on the wrong dimension
    usp_data = pd.DataFrame(bool_counts, index=multindx)
    # so transpose a row
    usp_data.iloc[:, 0] = usp_data.iloc[0, :]
    # and select that column
    usp_data = usp_data.iloc[:, 0]

    return usp_data


def do_plot_regression(x, y, ax, colour='black'):
    if all([type(xy) is pd.Series for xy in (x, y)]):
        if x.isna().any() or y.isna().any():
            raise RuntimeError('x and/or y contain NANs.')

    ols = OLS(y, sm.add_constant(x)).fit()
    xs = [x.min() * 0.8, x.max() * 0.8]
    ys = ols.predict(sm.add_constant(xs))

    ax.plot(xs, ys, color=colour, )


def get_density(x, y):
    """Get kernal density estimate for each (x, y) point.

    If pd.Series are used, return value is Series."""
    isSeries = type(x) is pd.Series
    if isSeries:
        assert x.index.equals(y.index)

    # do KDE
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    density = kernel(values)

    if isSeries:
        density = pd.Series(density, index=x.index)

    return density


def plot_density_scatter(
        x: pd.Series, y: pd.Series, ax=None, log_d=True,
        return_d=False,
        **scatterkwargs):
    """Plot scatterplot with colors representing density of points"""
    if ax is None:
        ax = plt.gca()

    # if x was not series this will fail
    d = get_density(x, y).sort_values()
    if log_d:
        d = d.apply(np.log)

    x, y = [xy.reindex(d.index) for xy in (x, y)]

    kw = dict(s=4, cmap='magma', c=d)
    kw.update(scatterkwargs)
    ax.scatter(x, y, **kw)
    if return_d:
        return ax, d
    return ax

def get_palette(color_factor:pd.Series, palettes:tuple=('deep', 'pastel')) -> pd.Series:
    """Get custom colour palette for use with Seaborn plots.

    Returns series with same index as color_factor and values of a colour
    for each unique value in color_factor

    Cycles through palettes in palettes"""
    clr_cycle = itertools.cycle(
        itertools.chain(
            *[sns.color_palette(p) for p in palettes]
    ))

    fctrs = color_factor.unique()
    cdict = dict(zip(fctrs, clr_cycle))

    return color_factor.map(cdict)


def factor_color_map(factors: DFOrSeries, palletes=('deep', 'pastel', 'Set2')):
    """Get a DF|Series of colors for each unique value, per column, in `factors`.

    Pass sample details DF, output can be passed to sns.clustermap"""

    if type(factors) is not pd.DataFrame:
        factordf = pd.DataFrame(factors)
    else:
        factordf = factors
    factordf = factordf.copy()

    if type(palletes) is str:
        palletes = (palletes,)

    palletes = itertools.cycle(palletes)

    for col in factordf:
        clrs = sns.color_palette(next(palletes))
        fctrs = factordf[col].unique()
        if len(fctrs) > len(clrs):
            #raise
            logging.warning(f'Too many factors in {col} for colours available, recyling colours.')
        cdict = dict(zip(fctrs, itertools.cycle(clrs), ))
        factordf[col] = factordf[col].map(cdict)
    return factordf

def resize_line(x0:float, x1:float, y0:float, y1:float, proportion) \
        -> tuple[float, float, float, float]:
    """Reduce or increase the size of a line. Can be used for axes
    limits as well."""
    # Calculate the midpoint of the line
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2

    # Calculate the direction vector
    dir_vector_x = (x1 - x0) / 2
    dir_vector_y = (y1 - y0) / 2

    # Multiply direction vector by half of the desired length
    dir_vector_x *= proportion
    dir_vector_y *= proportion

    # Calculate new endpoints based on the midpoint and modified direction vector
    new_x0 = mid_x - dir_vector_x
    new_x1 = mid_x + dir_vector_x
    new_y0 = mid_y - dir_vector_y
    new_y1 = mid_y + dir_vector_y

    return new_x0, new_x1, new_y0, new_y1


def plt_diagonal(ax=None, shrink=0.2, x_shift=0, y_shift=0, **plotkw):
    """Plot a line across most of x=y.
    Args:
        ax: pass specific ax to draw on it
        shrink: move ends of line away from corners, proportion, range 0-1.
        x_shift: shift the line on x axis
        y_shift: shift the line on y axis
        plotkw: passed to plt.plot
    returns plt.Axes
    """
    if ax is None:
        ax = plt.gca()

    # calculate shrink extent
    mn, mx = minminmaxmax(ax.get_xlim(), ax.get_ylim())
    extent = mx - mn
    delta = extent * (shrink/2)

    # xy values
    xs, ys = [
        (
            mn + delta + xy_off,
            mx - delta + xy_off
        )
        for xy_off in (x_shift, y_shift)
    ]

    kwargs = dict(ls='--', color='k') | plotkw

    ax.plot(xs, ys, **kwargs)
    return ax


def plt_line(x=None, y=None, ax=None, **plotkwargs):
    """Plot horizontal or vertical line across the axis length.

    kwargs passed to plt.plot.

    Axes do not get expanded by lines, so lines may be outside the drawn area.
    """
    if (x is None) and (y is None):
        raise ValueError("Set a value for x and/or y")

    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    plotkwargs = dict(ls='dashed', color='k', lw=0.7) | plotkwargs

    if x is not None:
        ax.plot([x, x], [ylims[0], ylims[1]], **plotkwargs)

    if y is not None:
        ax.plot([xlims[0], xlims[1]], [y, y], **plotkwargs)

    ax.set_ylim(ylims)
    ax.set_xlim(xlims)


def plt_expand_axlims(xamount: float = 0.1, yamount: Optional[float] = None, ax=None) -> None:
    if yamount is None:
        yamount = xamount

    if ax is None:
        ax = plt.gca()

    for getlim, setlim, amount in [
        (ax.get_xlim, ax.set_xlim, xamount),
        (ax.get_ylim, ax.set_ylim, yamount)
    ]:
        mn, mx = getlim()

        d = (mx - mn) * amount / 2
        setlim(mn - d, mx + d)


def plot_qq(values: np.ndarray,
            distribution=stats.norm,
            scale_values: Callable = lambda v: v / np.std(v),
            ax: plt.Axes = None
    ) -> plt.Axes:
    """Quantile-quantile plot.

    Args:
        values: An array of values, e.g. residuals
        distribution: Distribution to draw quantiles from. Requires ppf method. You
            can initiate scipy.stats distributions objects here if you want specific
            params.
        scale_values: If a function, apply to values before generating the quantiles.
            Defaults to norm by std. Otherwise use raw values.
        ax: pyplot Axes to use
    """
    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()

    n = values.shape[0]

    perc = np.linspace(0, 100, n)

    if isinstance(scale_values, Callable):
        values = scale_values(values)

    values = np.sort(values)

    # Calculate the quantiles for a standard normal distribution
    even = distribution.ppf(np.linspace(1 / n, 1 - 1 / n, n))

    normal_quantiles = np.percentile(even, perc)

    ax.scatter(normal_quantiles, values, )
    plt_diagonal(color='red', ax=ax)
    plt_labels('Theoretical quantiles', 'Sorted values', 'Quantile-quantile')
    return ax


def plt_annotate_adjust(
        x: SeriesT[float],
        y: SeriesT[float],
        text: Collection[str],
        arrowstyle: Optional[str] = '-',
        arrowcolor: str = 'k',
        textkwargs: Optional[dict] = None,
        adjustkwargs: Optional[dict] = None,
        ax: Optional[plt.Axes] = None,
):
    """Annotate points in x & y with text, and adjust their positions to
    prevent overlaps.

    If values of text are indicies of x & y, text will be used to get the
    positions. Otherwise; x, y & text are zipped and iterated through.

    textkwargs passed to plt.text.
    adjustkwargs to adjustText.adjust_text

    Set arrowstyle to None to remove arrow."""
    from adjustText import adjust_text
    if ax is None:
        ax = plt.gca()

    default_textkwargs = {
        'fontsize':'x-small',
        'bbox':{'facecolor':'white',
                'boxstyle':'square,pad=0',
                'alpha':.6}
    }
    if textkwargs is None:
        textkwargs = default_textkwargs
    else:
        textkwargs = default_textkwargs|textkwargs

    if adjustkwargs is None:
        adjustkwargs = {}

    if arrowstyle is not None:
        # adjustkwargs values take precident
        adjustkwargs = dict(arrowstyle=arrowstyle, color=arrowcolor) | adjustkwargs

    if isinstance(x, pd.Series) and all([ss in x.index for ss in text]):
        annotes = [ax.text(x[s], y[s], s, **textkwargs) for s in text]
    else:
        assert len(x) == len(y) == len(text)
        annotes = [ax.text(xx, yy, s, **textkwargs) for (xx, yy, s) in zip(x, y, text)]

    adjust_text(annotes, x, y, **adjustkwargs)

    return annotes

def plt_match_xylims():
    axmin, axmax = minminmaxmax(plt.xlim(), plt.ylim())
    plt.xlim(axmin, axmax)
    plt.ylim(axmin, axmax)
