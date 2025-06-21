import numpy as np
from scipy.stats import rankdata


def _uniform(F_lo, F_hi, alpha):
    rank_lo, rank_hi = rankdata(F_lo, method="max", axis=0), rankdata(
        F_hi, method="min", axis=0
    )
    infZ, supZ = rank_lo.min(axis=1), rank_hi.max(axis=1)
    q_lo, q_hi = np.quantile(infZ, alpha / 2, method="nearest"), np.quantile(
        supZ, 1 - alpha / 2, method="nearest"
    )
    return {"lower": np.sort(F_lo, axis=0)[q_lo], "upper": np.sort(F_hi, axis=0)[q_hi]}


def _student(F_lo, F_hi, alpha, eps, min_val, max_val):
    mean_lo, mean_hi = F_lo.mean(axis=0), F_hi.mean(axis=0)
    std_lo, std_hi = F_lo.std(axis=0) + eps, F_hi.std(axis=0)
    T_lo, T_hi = (F_lo - mean_lo) / std_lo, (F_hi - mean_hi) / std_hi
    infT, supT = T_lo.min(axis=1), T_hi.max(axis=1)
    q_lo, q_hi = np.quantile(infT, alpha / 2), np.quantile(supT, 1 - alpha / 2)
    return {
        "lower": np.clip(mean_lo + q_lo * std_lo, min_val, max_val),
        "upper": np.clip(mean_hi + q_hi * std_hi, min_val, max_val),
    }


def get_bands(F_lo, F_hi, alpha=0.05, eps=1e-8, method="uniform", min_val=0.0, max_val=1.0):
    """
    Compute simultaneous confidence bands for a family of cumulative distribution functions (CDFs).

    Parameters
    ----------
    F_lo : ndarray of shape (n_sim, n_points)
        Matrix of simulated lower bound CDF values at `n_points` evaluation locations, across `n_sim` simulations.

    F_hi : ndarray of shape (n_sim, n_points)
        Matrix of simulated upper bound CDF values at the same `n_points`, across `n_sim` simulations.

    alpha : float, default=0.05
        Significance level for the confidence bands. The returned bands are (1 - alpha) simultaneous confidence bands.

    eps : float, default=1e-8
        Small positive value added to the denominator to stabilize standard deviation estimates in the 'student' method.
        Ignored if `method='uniform'`.

    method : {'uniform', 'student'}, default='uniform'
        Method used to compute the bands:
        - 'uniform': uses rank statistics to construct a uniform confidence envelope.
        - 'student': uses t-statistics (mean ± quantile × std) for marginal band approximation.

    min_val : float, default=0.0
        Minimum value of the functions.

    max_val : float, default=1.0
        Maximum value of the functions.

    Returns
    -------
    bands : dict
        Dictionary with keys:
        - 'lower': array of lower bounds for the confidence band (shape: n_points)
        - 'upper': array of upper bounds for the confidence band (shape: n_points)

    Notes
    -----
    - The 'uniform' method provides valid simultaneous bands by controlling the maximum deviation uniformly.
    - The 'student' method may yield narrower bands but does not guarantee simultaneous coverage.

    Examples
    --------
    >>> bands = get_bands(F_lo, F_hi, alpha=0.05, method='uniform')
    >>> lower, upper = bands['lower'], bands['upper']
    """
    assert method in ("uniform", "student")
    assert alpha >= 0 and alpha <= 1

    if method == "uniform":
        return _uniform(F_lo, F_hi, alpha)

    assert eps > 0

    return _student(F_lo, F_hi, alpha, eps, min_val, max_val)
