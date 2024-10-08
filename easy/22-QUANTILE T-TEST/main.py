from typing import List, Tuple
from scipy import stats


def ttest(
    control: List[float], 
    experiment: List[float], 
    alpha: float = 0.05
) -> Tuple[float, bool]:
    t_statistic, p_value = stats.ttest_ind(control, experiment)
    result = p_value < alpha
    return p_value, bool(result)