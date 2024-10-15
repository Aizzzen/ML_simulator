from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def bootstrapping(data, n_bootstraps, quantile):
    bootstrapped_quantiles = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_quantiles.append(np.quantile(sample, quantile))
    return bootstrapped_quantiles


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    bootstrapped_control = bootstrapping(control, n_bootstraps, quantile)
    bootstrapped_experiment = bootstrapping(experiment, n_bootstraps, quantile)
    _, p_value = ttest_ind(bootstrapped_control, bootstrapped_experiment)
    result = p_value < alpha
    return p_value, bool(result)