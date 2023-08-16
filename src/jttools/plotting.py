import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import logging

OLS = sm.regression.linear_model.OLS
DFOrSeries = pd.DataFrame | pd.Series


def minminmaxmax(x, y):
    nn = min([min(x), min(y)])
    xx = max([max(x), max(y)])
    return (nn, xx)



def plt_match_xylims():
    axmin, axmax = minminmaxmax(plt.xlim(), plt.ylim())
    plt.xlim(axmin, axmax)
    plt.ylim(axmin, axmax)

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


def prep_data_for_upsetplot(bool_table):
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
