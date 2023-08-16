
import statsmodels.api as sm
import numpy as np
import pandas as pd

log2p1 = lambda x: np.log2(x + 1)
neglog10 = lambda p: -np.log10(p)


OLS = sm.regression.linear_model.OLS

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
    if type(ps) == pd.core.series.Series:
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