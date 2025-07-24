from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata  # type: ignore


def uniform(
    F_lo: NDArray[np.float32 | np.float64],
    F_hi: NDArray[np.float32 | np.float64],
    alpha: float,
) -> dict[str, NDArray[np.float32 | np.float64]]:
    """Implements the Uniform method for uniform control bands.

    Args:
        F_lo (NDArray[np.float32  |  np.float64]): The lower high probability bounds.
        F_hi (NDArray[np.float32  |  np.float64]):  The upper high probability bounds.
        alpha (float): The level of supplementary risk.

    Returns:
        dict[str, NDArray[np.float32 | np.float64]]: The uniform bands for level alpha of supplementary risk.
    """

    rank_lo, rank_hi = rankdata(F_lo, method="max", axis=0), rankdata(
        F_hi, method="min", axis=0
    )
    infZ, supZ = rank_lo.min(axis=1) - 1, rank_hi.max(axis=1) - 1

    q_lo = cast(int, np.quantile(infZ, alpha / 2, method="lower"))
    q_hi = cast(int, np.quantile(supZ, 1 - alpha / 2, method="higher"))

    return {"lower": np.sort(F_lo, axis=0)[q_lo], "upper": np.sort(F_hi, axis=0)[q_hi]}


def student(
    F_lo: NDArray[np.float32 | np.float64],
    F_hi: NDArray[np.float32 | np.float64],
    alpha: float,
    eps: float,
    min_val: float,
    max_val: float,
) -> dict[str, NDArray[np.float32 | np.float64]]:
    """Implements the Student method for uniform control bands.

    Args:
        F_lo (NDArray[np.float32  |  np.float64]): The lower high probability bounds.
        F_hi (NDArray[np.float32  |  np.float64]):  The upper high probability bounds.
        alpha (float): The level of supplementary risk.
        eps (float): The regularization parameter to ensure a well defined division.
        min_val (float): The minimum accepted values of the functions.
        max_val (float): The maximum accepted values of the functions.

    Returns:
        dict[str, NDArray[np.float32 | np.float64]]: The uniform bands for level alpha of supplementary risk.
    """

    mean_lo, mean_hi = F_lo.mean(axis=0), F_hi.mean(axis=0)
    std_lo, std_hi = F_lo.std(axis=0) + eps, F_hi.std(axis=0) + eps

    T_lo, T_hi = (F_lo - mean_lo) / std_lo, (F_hi - mean_hi) / std_hi
    infT, supT = T_lo.min(axis=1), T_hi.max(axis=1)

    q_lo = cast(float, np.quantile(infT, alpha / 2))
    q_hi = cast(float, np.quantile(supT, 1 - alpha / 2))

    return {
        "lower": np.clip(mean_lo + q_lo * std_lo, min_val, max_val),
        "upper": np.clip(mean_hi + q_hi * std_hi, min_val, max_val),
    }
