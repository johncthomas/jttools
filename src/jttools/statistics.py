import typing
from typing import Tuple, Collection, Mapping

import attrs
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

from scipy.stats import t as t_dist

from sklearn.preprocessing import StandardScaler

from jttools.data_wrangling import AttrMapAC

from pandas.api.types import is_numeric_dtype

log2p1 = lambda x: np.log2(x + 1)
neglog10 = lambda p: -np.log10(p)

OLS = sm.regression.linear_model.OLS

# Could  make this a function that imports and returns so we don't always start an R instance,
#   but it's probably fine.
from jttools.r_utils import rpy2_converter, ro, R

def multipletests_fdr(ps, method='fdr_bh', **kwargs):
    """Calls statsmodels.stats.multipletests and returns the corrected
    p-values only, with default of fdr_bh, rather than a FWER method.
    """
    kwargs['method'] = method
    try:
        if ps.isna().any():
            raise RuntimeError('ps contains NAN, this will break it')
    except AttributeError:
        pass
    qs = sm.stats.multipletests(ps, **kwargs)[1]
    if type(ps) == pd.Series:
        return pd.Series(qs, index=ps.index)
    return qs

def normalise_counts_log_zscore(counts_df):
    """Log2, abundance and z-normalised counts."""
    # Library size normalization: divide each column by its sum
    df_normalized = counts_df.div(counts_df.sum())

    # Add a small constant to avoid division by zero, then log2 transform
    df_log2 = np.log2(df_normalized + 1e-5)

    # Standardize the data to have a mean of 0 and a variance of 1
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_log2)
    df_scaled = pd.DataFrame(df_scaled,
                             columns=counts_df.columns,
                             index=counts_df.index)
    return df_scaled


def normalise_abundance(count:pd.DataFrame) -> pd.DataFrame:
    """Make column sums equal to the median of untransformed column sums"""
    return (count / count.sum()) * count.sum().median()

def normalise_zscore(count:pd.DataFrame) -> pd.DataFrame:
    """Make data have mean==0 and std==1"""
    return (count-count.mean())/count.std()

def normalise_median(count:pd.DataFrame, keep_scale=True) -> pd.DataFrame:
    """Divide columns by their median value, and (if keep_scale) multiply
    all columns by the pre-normalisation median of medians"""
    medians = count.median()
    count = count/medians
    if keep_scale:
        count = count*medians.median()
    return count

def apply_log2(count:pd.DataFrame, pseudocount=1):
    return count.apply(lambda n: np.log2(n+pseudocount))


def predict_xy_slope(
        xvals:np.ndarray,
        a:float,
        b:float
) -> tuple[list[float, float], list[float, float]]:
    """Predict Y values for min and max xvals, return values for plotting
    a line of best fit.

    a is intercept, b is slope."""
    f = lambda x: x * b + a
    x0, x1 = min(xvals), max(xvals)
    y0, y1 = f(x0), f(x1)
    return [x0, x1], [y0, y1]

def ci_lower_upper_percentile(a:np.ndarray, which=95, axis=0):
    """Lower and upper percentiles along an array.
    With axis=0, the percentile in each column returned."""
    p = (50 - which / 2, 50 + which / 2)
    return np.nanpercentile(a, p, axis)


@attrs.define(repr=False)
class OrthRegResult:
    coefficients: np.ndarray
    se: np.ndarray
    sigma: np.ndarray
    residuals: np.ndarray
    variance: np.ndarray

    ci: np.ndarray = attrs.field(converter=lambda arr: arr.T)
    model: pd.DataFrame
    n: np.ndarray
    bootstrapped_coeff:np.ndarray = None

    def __repr__(self):
        reprstr = (f"OrthogonalRegression result: \n"
                f"coeff: {self.coefficients}\n"
                f"deming ci:\n"
                f"{self.ci}")
        if self.bootstrapped_coeff is not None:
            med = ci_lower_upper_percentile(self.bootstrapped_coeff, 50)[0]
            ci = ci_lower_upper_percentile(self.bootstrapped_coeff, 95)
            reprstr += f"\nBS median={med}"
            reprstr += (f"\nBS 95%:\n"
                        f"{ci}")
        return reprstr

    # def get_bs_param(self, percentile):
    #     # this does the same thing as ci_get_line whatever
    #     if percentile == 50:
    #         p = percentile
    #     else:
    #         p = [(percentile/2), (100-(percentile/2))]
    #     return np.nanpercentile(self.bootstrapped_coeff, p, 0)



    def get_bs_ci_lines(self, ci=(50, 95), xi=100) -> dict[str, np.ndarray, np.ndarray]:
        """Get y values of confidence intervals for plotting, from bootstrapped
        coefficients.

        X values are results['x']. Results keyed by 'Upper/Lower {p}%' or 'Median'.
        Values in order of given 'Lower {ci[i]}%', 'Upper {ci[i]}%, then i+1.
        """
        linx = np.linspace(
            self.model.x.min(), self.model.x.max(), xi
        )
        dub_predict = lambda row: row[0] + row[1] * linx
        predicted_lines = np.apply_along_axis(dub_predict, 1,
                                              self.bootstrapped_coeff)

        ci_lines = {'x':linx}
        for p in ci:
            ci_l = ci_lower_upper_percentile(predicted_lines, p)
            if p != 50:
                ci_lines[f'Lower {p}%'] = ci_l[0]
                ci_lines[f'Upper {p}%'] = ci_l[1]
            else:
                ci_lines[f"Median"] = ci_l[0]

        return ci_lines

    def get_jackknife_ci_lines(
            self, xi=100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get lower, upper values of confidence intervals for plotting.

        Uses the CI from calculated SE from jackknife resampling."""
        amed, bmed = self.coefficients

        (amin, amax), (bmin, bmax) = self.ci

        x = self.model.x
        linx = np.linspace(min(x) * 0.8, max(x) * 0.8, xi)

        lines = np.zeros(shape=(4, xi), dtype=float)
        line_coeffs = [(amin, bmed), (amax, bmed), (amed, bmin), (amed, bmax)]
        for i, (a, b) in enumerate(line_coeffs):
            py = OrthogonalRegression.predict_simple(linx, a, b)
            lines[i] = py

        lower = np.min(lines, axis=0)
        upper = np.max(lines, axis=0)

        # this tuple can be passed to plt.fillbetween(*vals), so don't mess with it
        return linx, lower, upper

    def plot(self, ax:plt.Axes=None, scatter=True, ci=95, use_bootstrap=True,
             fill_kwargs:dict=None):
        """If use_bootstrap is True and bootstrapping was done,
        CI lines will come from those coefficients. Otherwise, uses
        the CI from the jackknife method.

        fill_kwargs passed to ax.fill_between"""
        from jttools.plotting import plt, plot_density_scatter

        if ax is None:
            plt.figure(figsize=(4.5, 4.5))
            ax = plt.gca()

        x, y = self.model.x, self.model.y
        if scatter:
            plot_density_scatter(x, y)

        # don't want to expand the graph to fit huge CI
        xylims = ax.get_xlim(), ax.get_ylim()
        if (self.bootstrapped_coeff is None) or (use_bootstrap is False):
            vals = self.get_jackknife_ci_lines()
            ax.fill_between(*vals, alpha=0.3, **fill_kwargs)
        else:
            d = self.get_bs_ci_lines(ci=(ci,))

            linx = d['x']

            if ci != 50:
                low, upp = d[f'Lower {ci}%'], d[f'Upper {ci}%']
                ax.fill_between(linx, low, upp, alpha=0.3, label=f"CI {ci}%")
            else:
                ax.plot(linx, d['Median'], label='BS median')
        ax.set_xlim(*xylims[0])
        ax.set_ylim(*xylims[1])
        return ax


class OrthogonalRegression:
    """Wrapper for R deming package. See
    https://cran.r-project.org/web/packages/deming/deming.pdf

    Loads deming library and defines call_tls in the rpy2 R instance.

    Methods:
        tls_simple: Total least squares regression with 1D x and y. CI
            calculated from stats.
        tls_bootstrap: TLS with 1D x & y. CI calculated with bootstraps.
            Array of
        deming: wrapper for the R deming function, with converter
            context.
    """
    #todo this should be a module, handles the multiple init thing better

    @staticmethod
    def predict_simple(x, a, b):
        """return x * b + a"""
        return x * b + a

    def __init__(self):
        # register function, this is just a basic TLS regression.
        R(
    """
    library(deming)
    call_tls = function(x, y, xstd, ystd) {
        #print(head(as.vector(xstd)))
        df = as.data.frame(cbind(x, y))
        res = deming(
            y ~ x, 
            data=df, 
            xstd=as.vector(xstd), ystd=as.vector(ystd),
            jackknife=TRUE
        )
        return(res)
    }
    
    bootstrap_tls <- function(x, y, nboot=10000) {
        df <- as.data.frame(cbind(x, y))
        
        colnames(df) <- c("x", "y")
        coeff_boot <- matrix(0., nrow=nboot, ncol=2)
        n <- length(x)
        #fixedxsd = rep(sd(x), each=n)
        #fixedysd = rep(sd(y), each=n)
        
        for (rep in 1:nboot) {
            indicies <- sample(1:n, size=n, replace=TRUE)
    
            nd <- df[indicies,]
            
            xsd  <- rep(sd(nd$x), each=n)
            ysd  <- rep(sd(nd$y), each=n)
    
            r  <- deming(
                y ~ x,
                data=nd,
                xstd=xsd,
                ystd=ysd,
                jackknife=FALSE
            )
            coeff_boot[rep,] = r$coefficients
        }
        return(coeff_boot)
    }
    """
        )

        self.result: OrthRegResult|None = None


    def _deming_result(self, result:typing.Mapping) -> OrthRegResult:
        result = dict(result)

        # These attr include rpy objects that can act weird
        #  (call in particular apparently (sometimes) calls deming
        #  with unknown params and crashes when it's viewed).
        del result['call']
        del result['terms']
        self.result = OrthRegResult(**result)
        return self.result

    def deming(self, *args, **kwargs):
        """Directly call R deming function"""
        # todo test this
        with rpy2_converter():
            result = R.deming(*args, **kwargs)
        return self._deming_result(result)

    def tls_simple(self, x, y) -> OrthRegResult:
        """Total least squares regression with 1D x and y,
        using R deming package.

        Residuals must be normally distributed."""
        n = x.shape
        xstd = np.ones(shape=n) * x.std()
        ystd = np.ones(shape=n) * y.std()

        with rpy2_converter():
            result = R.call_tls(x, y, xstd, ystd)

        return self._deming_result(result)

    def tls_bootstrap(self, x, y, nboot=10000) \
            -> OrthRegResult:
        """Total least squares regression with 1D x and y,
        using R deming package. Boostrapped CI from multiple
        calls to deming, and taking 95% of coefficients."""

        with rpy2_converter():
            boot_coef = R.bootstrap_tls(x, y, nboot)
            result = self.tls_simple(x, y)

        result.bootstrapped_coeff = boot_coef

        return result



