# -*- coding: utf-8 -*-
"""
Обратная задача для связанной модели.

Назначение:
По желаемому графику давления подобрать такие значения ПЛК,
которые после прохождения через физическую модель дадут
максимально близкий реальный отклик.

В данном файле отсутствует любая логика интерфейса.
Функции работают только с массивами numpy.
"""

import numpy as np
from scripts.coupled_linear_predictor.pressure_predictor_coupled import solve_coupled_system
import model_constants as mc

# =============================================================================
# ПОСТРОЕНИЕ ЖЕЛАЕМОГО ПРОФИЛЯ
# =============================================================================

def build_goal_profile_preview(
    presets: list[float],
    duration: float,
    phases: dict,
    dt: float = 0.02,
):
    """
    Построение непрерывного желаемого профиля давления.

    presets : list          Значения давления в ключевых точках (бар).
    duration : float        Общая длительность впрыска.
    phases : dict           Конфигурация фаз из CONFIG_GENERATE_TARGET_*.
    dt : float              Шаг построения графика.
    
    Returns
    time_grid : ndarray     Временная сетка.
    goal_profile : ndarray  Непрерывный желаемый профиль давления.
    preset_times : ndarray  Время расположения ключевых точек.
    """

    # Формирование временной сетки
    t_start = phases["t_start"]

    time_grid = np.arange(t_start, duration + dt, dt)

    # Время расположения опорных точек
    preset_times = np.linspace(t_start, duration, len(presets))

    # Линейная интерполяция между заданными точками
    goal_profile = np.interp(time_grid, preset_times, presets)

    return (time_grid, goal_profile, preset_times)

# =============================================================================
# ГЕНЕРАЦИЯ СИГНАЛА ДЛЯ ПЛК
# =============================================================================

def generate_target_signal(plc_values: list[int], time_grid: np.ndarray, preset_times: np.ndarray):
    """
    Формирование ступенчатого сигнала задания ПЛК.

    plc_values : list            Дискретные значения ПЛК (0...60).
    time_grid : ndarray          Временная сетка расчета.
    preset_times : ndarray       Время переключения пресетов.

    Returns
    target_signal : ndarray      Ступенчатый сигнал ПЛК.
    """

    target_signal = np.zeros_like(time_grid)

    for i, value in enumerate(plc_values):

        if i == len(plc_values) - 1:
            mask = time_grid >= preset_times[i]
        else:
            mask = (time_grid >= preset_times[i]) & (time_grid < preset_times[i + 1])

        target_signal[mask] = value

    return target_signal


# =============================================================================
# ЦЕЛЕВАЯ ФУНКЦИЯ ОПТИМИЗАЦИИ
# =============================================================================

def objective_loss(
    plc_tank,
    plc_pipe,
    time_grid,
    preset_times,
    goal_tank,
    goal_pipe,
    flat_params_tank,
    flat_params_pipe,
):
    """
    Вычисление ошибки между желаемым и рассчитанным профилем.

    plc_tank : ndarray           Подбираемые значения ПЛК бака.
    plc_pipe : ndarray           Подбираемые значения ПЛК трубы.
    time_grid : ndarray          Временная сетка.
    preset_times : ndarray       Моменты переключения пресетов.
    goal_tank : ndarray          Желаемое давление в баке.
    goal_pipe : ndarray          Желаемое давление в трубе.
    flat_params_* : dict         Параметры физической модели.

    Returns
    loss : float                 Средняя абсолютная ошибка.
    """

    # ПЛК работает только с целыми значениями
    plc_tank = np.round(plc_tank).astype(int)
    plc_pipe = np.round(plc_pipe).astype(int)

    # Формирование ступенчатых сигналов управления
    target_tank = generate_target_signal(plc_tank, time_grid, preset_times)
    target_pipe = generate_target_signal(plc_pipe, time_grid, preset_times)

    # Расчет физической модели
    prediction_tank, prediction_pipe = solve_coupled_system(
        target_tank,
        target_pipe,
        flat_params_tank,
        flat_params_pipe,
    )

    # Ошибка по каждому каналу
    mae_tank = np.mean(np.abs(goal_tank - prediction_tank))
    mae_pipe = np.mean(np.abs(goal_pipe - prediction_pipe))

    return mae_tank + mae_pipe

# =============================================================================
# ПЕРЕВОД ЗНАЧЕНИЙ ПЛК В ПРОЦЕНТЫ
# =============================================================================

def plc_to_percent(plc_values):
    """
    Перевод дискретных значений ПЛК в рекомендуемые проценты.

    plc_values : ndarray         Значения ПЛК (0...60).

    Returns
    percent_values : ndarray     Целые проценты для оператора.
    """

    percent = np.floor(plc_values / mc.PRESSURE_SCALE / mc.PRESSURE_MAX * 100)

    return percent.astype(int)

def run_preset_optimization(
    presets_tank,
    presets_pipe,
    duration,
    phases,
    flat_params_tank,
    flat_params_pipe,
):
    """
    Подбор оптимальных значений ПЛК.

    presets_tank : list          Желаемый профиль бака.
    presets_pipe : list          Желаемый профиль трубы.
    duration : float             Длительность цикла.
    phases : dict                Конфигурация фаз.
    flat_params_* : dict         Параметры физической модели.

    Returns
    result : dict                Результаты оптимизации.
    """

    # Построение желаемых профилей
    time_grid, goal_tank, preset_times = build_goal_profile_preview(
        presets_tank,
        duration,
        phases,
    )

    _, goal_pipe, _ = build_goal_profile_preview(
        presets_pipe,
        duration,
        phases,
    )

    # Начальное приближение
    initial_tank = np.round(goal_tank * PRESSURE_SCALE).astype(int)
    initial_pipe = np.round(goal_pipe * PRESSURE_SCALE).astype(int)

    # Запуск Differential Evolution
    ...
    best_plc_tank = ...
    best_plc_pipe = ...

    # Финальный расчет модели
    target_tank = _generate_target_signal(best_plc_tank, time_grid, preset_times)
    target_pipe = _generate_target_signal(best_plc_pipe, time_grid, preset_times)

    prediction_tank, prediction_pipe = solve_coupled_system(
        target_tank,
        target_pipe,
        flat_params_tank,
        flat_params_pipe,
    )

    return {
        "time_grid": time_grid,

        "goal_tank": goal_tank,
        "goal_pipe": goal_pipe,

        "prediction_tank": prediction_tank,
        "prediction_pipe": prediction_pipe,

        "target_plc_tank": best_plc_tank,
        "target_plc_pipe": best_plc_pipe,

        "target_percent_tank": _plc_to_percent(best_plc_tank),
        "target_percent_pipe": _plc_to_percent(best_plc_pipe),
    }