import numpy as np
from numpy.typing import NDArray

from ._methods import *


def get_bands(
    F_lo: NDArray[np.float32 | np.float64],
    F_hi: NDArray[np.float32 | np.float64] | None = None,
    alpha: float = 0.05,
    *,
    eps: float = 1e-8,
    method: str = "uniform",
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> dict[str, NDArray[np.float32 | np.float64]]:
    """Gets the uniform bands according the specified method.

    Args:
        F_lo (NDArray[np.float32  |  np.float64]): The lower high probability bounds.
        F_hi (NDArray[np.float32  |  np.float64]):  The upper high probability bounds. Defaults to None.
        alpha (float, optional): The level of supplementary risk. Defaults to 0.05.
        eps (float, optional): The regularization parameter to ensure a well defined division.. Defaults to 1e-8.
        method (str, optional): Either "uniform" or "student", the method used to compute uniform bands. Defaults to "uniform".
        min_val (float, optional): The minimum accepted values of the functions.. Defaults to 0.0.
        max_val (float, optional): The maximum accepted values of the functions.. Defaults to 1.0.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict[str, NDArray[np.float32 | np.float64]]: _description_
    """

    F_hi = F_hi if F_hi is not None else F_lo

    if F_lo.shape != F_hi.shape:
        raise ValueError(
            f"F_lo and F_hi shapes must be the same, got {F_lo.shape} != {F_hi.shape}"
        )

    if F_lo.ndim != 2:
        raise ValueError(f"F_lo and F_hi must be matrices, got shape {F_lo.shape}")

    if method not in ("uniform", "student"):
        raise ValueError(f"Method should be in ('uniform', 'student'), got {method}")

    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    if method == "uniform":
        return uniform(F_lo, F_hi, alpha)

    if eps <= 0:
        raise ValueError(
            f"eps must be strictly positive for the student method, got {eps}"
        )

    return student(F_lo, F_hi, alpha, eps, min_val, max_val)
