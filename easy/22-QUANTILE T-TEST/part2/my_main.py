from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind

# Для реализации бутстреп-теста Стьюдента (t-теста) для сравнения квантилей ошибок прогнозирования, нам нужно выполнить следующие шаги:
# 1. Бутстреп-выборка: Создать бутстреп-выборки для контрольной и экспериментальной групп.
# 2. Вычисление квантили: Вычислить заданный квантиль для каждой бутстреп-выборки.
# 3. T-тест: Провести t-тест на квантилях, полученных из бутстреп-выборок.
# 4. Интерпретация результата: Определить, меньше ли p-значение из t-теста уровня значимости alpha.

# Функция для вычисления квантили из бутстреп-выборок
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
    # Генерация бутстреп-квантилей для контрольной и экспериментальной групп
    bootstrapped_control = bootstrapping(control, n_bootstraps, quantile)
    bootstrapped_experiment = bootstrapping(experiment, n_bootstraps, quantile)
    # Проведение t-теста на бутстреп-квантилях
    _, p_value = ttest_ind(bootstrapped_control, bootstrapped_experiment)
    # Определение, отвергаем ли нулевую гипотезу
    result = p_value < alpha
    return p_value, bool(result)