# pressure_predictor_coupled.py
import numpy as np
import pandas as pd

# =============================================================================
# БЛОК 1: ДИСКУРС С PANDAS (БЕЗ ДУБЛИРОВАНИЯ)
# =============================================================================

def _apply_coupled_model(df_logs, df_registry, CONFIG_TANK, CONFIG_PIPE, flat_params_tank, flat_params_pipe):
    """
    Оркестратор: только запускает расчет давлений. Ошибки рассчитываются на выходе.
    """
    p_tank_results = np.zeros(len(df_logs))
    p_pipe_results = np.zeros(len(df_logs))
    
    cycle_col = CONFIG_TANK['col_cycle']
    # Раскладываем датафрейм на отдельные наборы по цикл айди
    grouped = df_logs.groupby(cycle_col)
    
    # Создаем словарь соответствия: ID цикла -> длительность (duration)
    duration_map = df_registry.set_index(cycle_col)['duration'].dropna().to_dict()

    # 1. Расчет физики
    # Цикл итерируется по этим стопкам
    for cycle_id, df_cycle in grouped:
        # Извлечение оригинальных индексов строк по cycle_id для сопоставления результатов в массивы
        indices = df_cycle.index.values
        
        cycle_duration = duration_map[cycle_id]

        # передаем чисто массивы за счет .values
        p_tank_pred, p_pipe_pred = solve_coupled_system(
            times=df_cycle[CONFIG_TANK['col_time']].values,
            target_tank=df_cycle[CONFIG_TANK['col_target']].values,
            target_pipe=df_cycle[CONFIG_PIPE['col_target']].values,
            flat_params_tank=flat_params_tank,
            flat_params_pipe=flat_params_pipe,
        )
        
        p_tank_results[indices] = p_tank_pred
        p_pipe_results[indices] = p_pipe_pred

    # 3. Запись результатов и расчет всех ошибок (в один проход для каждого канала)
    _finalize_channel_data(df_logs, df_registry, CONFIG_TANK, p_tank_results, duration_map)
    _finalize_channel_data(df_logs, df_registry, CONFIG_PIPE, p_pipe_results, duration_map)

    return df_logs, df_registry   

def _finalize_channel_data(df_logs, df_registry, CONFIG, predicted_results, duration_map):
    col_cycle, col_time = CONFIG['col_cycle'], CONFIG['col_time']
    
    # 1. Записываем прогноз и вычисляем построчную ошибку
    df_logs[CONFIG['col_result']] = predicted_results
    df_logs[CONFIG['col_MAE']] = (df_logs[CONFIG['col_actual']] - predicted_results).abs()
    
    # 2. Фильтруем интервал [0.0, duration] на лету через .map()
    active_logs = df_logs[(df_logs[col_time] >= 0.0) & (df_logs[col_time] <= df_logs[col_cycle].map(duration_map))]
    
    # 3. Группируем и переносим среднюю ошибку цикла в реестр
    cycle_mae_series = active_logs.groupby(col_cycle)[CONFIG['col_MAE']].mean()
    df_registry[CONFIG['col_MAE']] = df_registry[col_cycle].map(cycle_mae_series)


# =============================================================================
# БЛОК 2: МАТЕМАТИЧЕСКОЕ ЯДРО (ТОЛЬКО СКВОЗНЫЕ ПАРАМЕТРЫ И ВЕКТОРЫ)
# =============================================================================

def solve_coupled_system(times, target_tank, target_pipe, flat_params_tank, flat_params_pipe):
    n_steps = len(times)
    # заготовка массива под прогнозные значения
    p_tank_pred = np.zeros(n_steps)
    p_pipe_pred = np.zeros(n_steps)
    
    # индекс нулевого момента времени - 100
    idx_t0 = np.searchsorted(times, 0.0)
    
    # заполняем график давления до 0с заполняя его статическим преднадувом, рассчитанным по линейной зависимости
    for i in range(idx_t0):
        p_tank_pred[i] = _calculate_static_precharge(target_tank[i])
        p_pipe_pred[i] = _calculate_static_precharge(target_pipe[i])

    p_tank_pred[idx_t0] = _calculate_static_precharge(target_tank[idx_t0])
    p_pipe_pred[idx_t0] = 0

    p_tank_current = p_tank_pred[idx_t0]
    p_pipe_current = p_pipe_pred[idx_t0]

    dt = 0.2

    for i in range(idx_t0+1, n_steps):

        target_t_delayed = _get_delayed_value(times, target_tank, times[i] - flat_params_tank['dead_time'], i)
        target_p_delayed = _get_delayed_value(times, target_pipe, times[i] - flat_params_pipe['dead_time'], i)

        delta_p = p_tank_current - p_pipe_current

        coupling_term_tank = - flat_params_tank['sigma_out'] * delta_p
        coupling_term_pipe = + flat_params_pipe['sigma_in'] * delta_p

        # Мы сознательно сохраняем параллельную структуру для наглядности ОДУ
        p_tank_next = _solve_first_order_step(
            p_prev=p_tank_current,
            target_p=target_t_delayed,
            dt=dt,
            damping=flat_params_tank['damping'],
            k_gain=flat_params_tank['k_gain'],
            b_gain=flat_params_tank['b_gain'],
            coupling_term=coupling_term_tank
        )

        p_pipe_next = _solve_first_order_step(
            p_prev=p_pipe_current,
            target_p=target_p_delayed,
            dt=dt,
            damping=flat_params_pipe['damping'],
            k_gain=flat_params_pipe['k_gain'],
            b_gain=flat_params_pipe['b_gain'],
            coupling_term=coupling_term_pipe
        )

        p_tank_current = max(0.0, p_tank_next)
        p_pipe_current = max(0.0, p_pipe_next)

        p_tank_pred[i] = p_tank_current
        p_pipe_pred[i] = p_pipe_current

    return p_tank_pred, p_pipe_pred


# =============================================================================
# БЛОК 3: ФИЗИКА И МАТЕМАТИКА (ПОЛНОСТЬЮ ОБОСОБЛЕНА)
# =============================================================================

def _solve_first_order_step(p_prev, target_p, dt, damping, k_gain, b_gain, coupling_term):
    k_gain_base = k_gain * p_prev + b_gain
    dp_dt = (1.0 / damping) * (k_gain_base * target_p - p_prev) + coupling_term
    return p_prev + dp_dt * dt


def _calculate_static_precharge(target, slope=0.1, intercept=-0.23):
    """
    Расчет установившегося давления преднадува по линейной зависимости.
    Построено по точкам: (Сигнал 3.0 -> 0.07 бар) и (Сигнал 4.0 -> 0.17 бар).
    """
    if target <= 0.0:
        return 0.0
    p_static = slope * target + intercept

    # print(f'заданное значение{target} начальное {p_static}')

    return max(0.0, p_static)


def _get_delayed_value(times, targets, target_time, current_idx):
    if target_time <= times[0]:
        return targets[0]
    start_idx = max(0, current_idx - 20)
    end_idx = current_idx + 1
    return np.interp(target_time, times[start_idx:end_idx], targets[start_idx:end_idx])


def _calculate_absolute_errors(actual_array, predicted_array):
    return np.abs(actual_array - predicted_array)


def _calculate_interval_mae(times, actual_array, predicted_array, t_min=3.0, t_max=7.0):
    mask = (times >= t_min) & (times <= t_max)
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(actual_array[mask] - predicted_array[mask]))