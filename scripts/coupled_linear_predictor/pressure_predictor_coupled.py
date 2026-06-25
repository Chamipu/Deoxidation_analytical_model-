# -*- coding: utf-8 -*-
"""
Связанная grey-box модель давлений бака и трубы.

Слои:
  - apply_coupled_model  — оркестрация (pandas, запись результатов)
  - solve_coupled_system — численное ядро для одного цикла (numpy)
  - вспомогательные функции физики и метрик
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "ChannelParams",
    "PredictorConfig",
    "apply_coupled_model",
    "solve_coupled_system",
    "solve_first_order_step",
    "calculate_static_precharge",
    "get_delayed_value",
    "calculate_absolute_errors",
    "calculate_interval_mae",
]

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Шаг времени по умолчанию, используемый при интегрировании модели.
DEFAULT_DT: float = 0.2
# Максимально допустимый шаг времени между соседними отсчётами.
MAX_VALID_DT: float = 1.0
# Время начала воздействия впрыска или запуска модели.
INJECTION_START_TIME: float = 0.0

# Диапазон времени для расчёта средней абсолютной ошибки.
MAE_INTERVAL: tuple[float, float] = (3.0, 7.0)

# Коэффициенты линейной зависимости статического преднадува от целевого значения.
PRECHARGE_SLOPE: float = 0.095
PRECHARGE_INTERCEPT: float = -0.215

# Запас по истории при поиске задержанного значения сигнала.
DELAY_LOOKBACK_PADDING: int = 5

# Префиксы колонок с фактическими измерениями давления для бака и трубы.
ACTUAL_TANK_COLUMN_PREFIX = r"IBA_DB\PT1009"
ACTUAL_PIPE_COLUMN_PREFIX = r"IBA_DB\PT1014"

# Обязательные параметры, которые должны присутствовать в словаре модели.
_REQUIRED_PARAM_KEYS = ("dead_time", "damping", "k_gain", "b_gain")


# ---------------------------------------------------------------------------
# Конфигурация и параметры
# ---------------------------------------------------------------------------

# Набор физико-математических параметров для одного канала модели.
@dataclass(frozen=True, slots=True)
class ChannelParams:
    """Параметры одного канала (бак или труба)."""

    dead_time: float
    damping: float
    k_gain: float
    b_gain: float
    sigma_out: float = 0.0
    sigma_in: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> ChannelParams:
        missing = [key for key in _REQUIRED_PARAM_KEYS if key not in data]
        if missing:
            raise KeyError(f"Отсутствуют обязательные параметры модели: {missing}")
        damping = float(data["damping"])
        if damping <= 0.0:
            raise ValueError(f"damping должен быть > 0, получено {damping}")
        return cls(
            dead_time=float(data["dead_time"]),
            damping=damping,
            k_gain=float(data["k_gain"]),
            b_gain=float(data["b_gain"]),
            sigma_out=float(data.get("sigma_out", 0.0)),
            sigma_in=float(data.get("sigma_in", 0.0)),
        )


# Описание того, какие колонки DataFrame используются для прогноза и метрик.
@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Колонки DataFrame для одного канала прогноза."""

    col_time: str
    col_cycle: str
    col_target: str
    col_result: str
    col_MAE: str
    col_actual: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredictorConfig:
        return cls(
            col_time=data["col_time"],
            col_cycle=data["col_cycle"],
            col_target=data["col_target"],
            col_result=data["col_result"],
            col_MAE=data["col_MAE"],
            col_actual=data.get("col_actual"),
        )


# ---------------------------------------------------------------------------
# Публичный API (pandas)
# ---------------------------------------------------------------------------

# Применяет связанную модель к всем циклам в логах и сохраняет прогноз и ошибки.
def apply_coupled_model(
    df_logs: pd.DataFrame,
    df_registry: pd.DataFrame,
    config_tank: Mapping[str, Any],
    config_pipe: Mapping[str, Any],
    flat_params_tank: Mapping[str, float],
    flat_params_pipe: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Прогноз давлений бака и трубы для всех циклов в df_logs.

    Записывает колонки col_result и col_MAE в df_logs и df_registry.
    Состояние модели сбрасывается на границе каждого cycle_id.
    """
    tank_cfg = PredictorConfig.from_dict(config_tank)
    pipe_cfg = PredictorConfig.from_dict(config_pipe)
    params_tank = ChannelParams.from_mapping(flat_params_tank)
    params_pipe = ChannelParams.from_mapping(flat_params_pipe)

    _validate_shared_columns(tank_cfg, pipe_cfg)

    tank_actual_col = _resolve_actual_column(df_logs, tank_cfg, ACTUAL_TANK_COLUMN_PREFIX)
    pipe_actual_col = _resolve_actual_column(df_logs, pipe_cfg, ACTUAL_PIPE_COLUMN_PREFIX)

    registry_tank_mae: dict[Any, float] = {}
    registry_pipe_mae: dict[Any, float] = {}

    for cycle_id, df_cycle in df_logs.groupby(tank_cfg.col_cycle, sort=False):
        times = df_cycle[tank_cfg.col_time].to_numpy(dtype=float)
        target_tank = df_cycle[tank_cfg.col_target].to_numpy(dtype=float)
        target_pipe = df_cycle[pipe_cfg.col_target].to_numpy(dtype=float)

        p_tank, p_pipe = solve_coupled_system(
            times, target_tank, target_pipe, params_tank, params_pipe,
        )

        index = df_cycle.index
        df_logs.loc[index, tank_cfg.col_result] = p_tank
        df_logs.loc[index, pipe_cfg.col_result] = p_pipe

        if tank_actual_col is not None:
            actual_tank = df_cycle[tank_actual_col].to_numpy(dtype=float)
            df_logs.loc[index, tank_cfg.col_MAE] = calculate_absolute_errors(actual_tank, p_tank)
            registry_tank_mae[cycle_id] = calculate_interval_mae(
                times, actual_tank, p_tank, *MAE_INTERVAL,
            )

        if pipe_actual_col is not None:
            actual_pipe = df_cycle[pipe_actual_col].to_numpy(dtype=float)
            df_logs.loc[index, pipe_cfg.col_MAE] = calculate_absolute_errors(actual_pipe, p_pipe)
            registry_pipe_mae[cycle_id] = calculate_interval_mae(
                times, actual_pipe, p_pipe, *MAE_INTERVAL,
            )

    if tank_actual_col is not None:
        df_registry[tank_cfg.col_MAE] = df_registry[tank_cfg.col_cycle].map(registry_tank_mae)

    if pipe_actual_col is not None:
        df_registry[pipe_cfg.col_MAE] = df_registry[pipe_cfg.col_cycle].map(registry_pipe_mae)

    return df_logs, df_registry


# ---------------------------------------------------------------------------
# Численное ядро (numpy)
# ---------------------------------------------------------------------------

# Решает связанную систему для одного цикла с учётом задержек и взаимосвязи двух каналов.
def solve_coupled_system(
    times: np.ndarray,
    target_tank: np.ndarray,
    target_pipe: np.ndarray,
    params_tank: ChannelParams | Mapping[str, float],
    params_pipe: ChannelParams | Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Решатель связанной системы давлений для одного цикла.

    Фаза t < 0: статический преднадув бака, труба = 0.
    Фаза t >= 0: интегрирование связанных уравнений первого порядка (Эйлер).
    """
    if isinstance(params_tank, Mapping):
        params_tank = ChannelParams.from_mapping(params_tank)
    if isinstance(params_pipe, Mapping):
        params_pipe = ChannelParams.from_mapping(params_pipe)

    n_steps = len(times)
    if n_steps == 0:
        return np.array([]), np.array([])

    if len(target_tank) != n_steps or len(target_pipe) != n_steps:
        raise ValueError("times, target_tank и target_pipe должны иметь одинаковую длину")

    p_tank = np.zeros(n_steps, dtype=float)
    p_pipe = np.zeros(n_steps, dtype=float)

    idx_t0 = int(np.searchsorted(times, INJECTION_START_TIME, side="left"))

    for i in range(idx_t0):
        p_tank[i] = calculate_static_precharge(target_tank[i])
        p_pipe[i] = 0.0

    if idx_t0 >= n_steps:
        return p_tank, p_pipe

    p_tank_state = p_tank[idx_t0 - 1] if idx_t0 > 0 else 0.0
    p_pipe_state = 0.0

    for i in range(idx_t0, n_steps):
        dt = _step_dt(times, i)

        delayed_tank = get_delayed_value(
            times, target_tank, times[i] - params_tank.dead_time, i, dt,
        )
        delayed_pipe = get_delayed_value(
            times, target_pipe, times[i] - params_pipe.dead_time, i, dt,
        )

        delta_p = p_tank_state - p_pipe_state
        coupling_tank = -params_tank.sigma_out * delta_p
        coupling_pipe = params_pipe.sigma_in * delta_p

        p_tank_state = max(
            0.0,
            solve_first_order_step(
                p_tank_state, delayed_tank, dt,
                params_tank.damping, params_tank.k_gain, params_tank.b_gain,
                coupling_tank,
            ),
        )
        p_pipe_state = max(
            0.0,
            solve_first_order_step(
                p_pipe_state, delayed_pipe, dt,
                params_pipe.damping, params_pipe.k_gain, params_pipe.b_gain,
                coupling_pipe,
            ),
        )

        p_tank[i] = p_tank_state
        p_pipe[i] = p_pipe_state

    return p_tank, p_pipe

# Выполняет один шаг интегрирования первого порядка для заданного канала.
def solve_first_order_step(
    p_prev: float,
    target_p: float,
    dt: float,
    damping: float,
    k_gain: float,
    b_gain: float,
    coupling_term: float = 0.0,
) -> float:
    """
    Один шаг модели первого порядка: τ·dP/dt + P = K(P)·S + coupling.

    dP/dt = (K(P)·S − P) / τ + coupling,  K(P) = k_gain·P + b_gain.
    """
    if damping <= 0.0:
        raise ValueError(f"damping должен быть > 0, получено {damping}")
    if dt <= 0.0:
        return p_prev

    k_effective = k_gain * p_prev + b_gain
    dp_dt = (k_effective * target_p - p_prev) / damping + coupling_term
    return p_prev + dp_dt * dt

# Возвращает начальное статическое давление до момента запуска модели.
def calculate_static_precharge(
    target: float,
    slope: float = PRECHARGE_SLOPE,
    intercept: float = PRECHARGE_INTERCEPT,
) -> float:
    """Статическое давление преднадува до t = 0."""
    if target <= 0.0:
        return 0.0
    return max(0.0, slope * target + intercept)

# Находит значение целевого сигнала с учётом запаздывания и линейной интерполяции.
def get_delayed_value(
    times: np.ndarray,
    targets: np.ndarray,
    query_time: float,
    current_idx: int,
    dt: float = DEFAULT_DT,
) -> float:
    """
    Линейная интерполяция уставки в момент query_time (учёт dead time).

    Окно интерполяции масштабируется по dead_time и шагу dt.
    """
    if query_time <= times[0]:
        return float(targets[0])

    lookback = max(DELAY_LOOKBACK_PADDING, int(np.ceil(abs(times[current_idx] - query_time) / max(dt, 1e-9))) + 1)
    start_idx = max(0, current_idx - lookback)

    return float(np.interp(
        query_time,
        times[start_idx: current_idx + 1],
        targets[start_idx: current_idx + 1],
    ))


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

# Вычисляет покомпонентную абсолютную ошибку между фактом и прогнозом.
def calculate_absolute_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Построчная абсолютная ошибка |actual − predicted|."""
    return np.abs(actual - predicted)

# Считает среднюю абсолютную ошибку на заданном интервале времени.
def calculate_interval_mae(
    times: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    t_min: float = MAE_INTERVAL[0],
    t_max: float = MAE_INTERVAL[1],
) -> float:
    """Средняя абсолютная ошибка на интервале [t_min, t_max]."""
    mask = (times >= t_min) & (times <= t_max)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

# Проверяет, что бак и труба используют одинаковые временные и цикловые колонки.
def _validate_shared_columns(tank: PredictorConfig, pipe: PredictorConfig) -> None:
    if tank.col_cycle != pipe.col_cycle:
        raise ValueError(
            f"col_cycle должен совпадать для бака и трубы: "
            f"{tank.col_cycle!r} != {pipe.col_cycle!r}"
        )
    if tank.col_time != pipe.col_time:
        raise ValueError(
            f"col_time должен совпадать для бака и трубы: "
            f"{tank.col_time!r} != {pipe.col_time!r}"
        )

# Ищет фактическую колонку с измерениями, если в конфиге она не задана явно.
def _resolve_actual_column(
    df: pd.DataFrame,
    config: PredictorConfig,
    fallback_prefix: str,
) -> str | None:
    if config.col_actual and config.col_actual in df.columns:
        return config.col_actual
    return next((col for col in df.columns if col.startswith(fallback_prefix)), None)

# Возвращает шаг интегрирования для текущей точки, защищая сетку от артефактов.
def _step_dt(times: np.ndarray, index: int) -> float:
    """Шаг интегрирования для точки index; защита от некорректной сетки."""
    if index <= 0:
        return DEFAULT_DT

    dt = float(times[index] - times[index - 1])
    if dt <= 0.0 or dt > MAX_VALID_DT:
        return DEFAULT_DT
    return dt
