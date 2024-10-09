from typing import List, Tuple
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    x_control = sum(control) / len(control)
    x_experiment = sum(experiment) / len(experiment)

    sd_for_each_in_control = []
    sd_for_each_in_experiment = []

    for i in control:
        sd_for_each_in_control.append((i - x_control)**2)

    for i in experiment:
        sd_for_each_in_experiment.append((i - x_experiment)**2)

    variance_for_control = sum(sd_for_each_in_control) / (len(control) - 1)
    variance_for_experiment = sum(sd_for_each_in_experiment) / (len(experiment) - 1)

    sd_control = variance_for_control**0.5
    sd_experiment = variance_for_experiment**0.5

    t_value = (x_control - x_experiment) / ((sd_control**2 / len(control)) + (sd_experiment**2 / len(experiment)))**0.5
    df = len(control) + len(experiment) - 2
    
    p_value = stats.t.sf([abs(t_value)], df)[0]
    result = p_value < alpha

    return p_value, result
