# scripts/pressure_predictor.py
import numpy as np
import pandas as pd

# =============================================================================
# 1. ПОИСК ЦИКЛА
# =============================================================================

# def _filter_mask(identifier, df):
#     """Находит строки цикла по тегу или времени старта."""
#     try:
#         t_val = pd.to_datetime(identifier)
#         return (df['cycle_start_time'] == t_val)
#     except (ValueError, TypeError):
#         # Если колонки нет в таблице — возвращаем маску из False
#         if 'case_tag' not in df.columns: return pd.Series(False, index=df.index)
#         return (df['case_tag'] == str(identifier))

def _filter_mask(identifier, df):
    """
    Ищет идентификатор в переданном датафрейме.
    """
    # Если ищем по времени
    if 'cycle_start_time' in df.columns:
        try:
            t_val = pd.to_datetime(identifier)
            mask = (df['cycle_start_time'] == t_val)
            if mask.any(): return mask
        except: pass

    # Если ищем по тегу
    if 'case_tag' in df.columns:
        mask = (df['case_tag'] == str(identifier))
        if mask.any(): return mask
        
    # Если ищем напрямую по ID
    if 'cycle_id' in df.columns:
        mask = (df['cycle_id'] == str(identifier))
        if mask.any(): return mask

    return pd.Series(False, index=df.index)

# =============================================================================
# 2. ФИЗИЧЕСКОЕ ЯДРО (Без изменений, чистая математика)
# =============================================================================

def solve_tank_step(target_signal, state, dt, flat_params):
    if 'v_pos' not in state: state['v_pos'] = 0.0

    target_p = flat_params["T_k_gain"] * target_signal
    gap = abs(target_p - state['P'])
    
    current_sharp = min(
        flat_params["T_sharp_base"] + (gap * flat_params["T_sharp_sens"]), 
        flat_params["T_sharp_max"]
    )

    dv = (target_signal - state['v_pos']) / (flat_params["T_valve_time"] + 1e-6)
    state['v_pos'] += dv * dt
    
    K, T, Z = flat_params["T_k_gain"], flat_params["T_time_const"], flat_params["T_damping"]
    d2P = (K * state['v_pos'] - 2 * Z * T * state['dP'] - state['P']) / (T**2)
    
    state['dP'] += d2P * dt
    state['P']  += state['dP'] * dt
    
    p_result = state['P'] + (current_sharp * T * state['dP'])
    return max(0.0, p_result)

# =============================================================================
# 3. ГЛАВНАЯ ФУНКЦИЯ ПРОГНОЗА (get_cycle_model)
# =============================================================================

def get_cycle_model(df_logs, df_registry, identifier, signal_col, flat_params, EXTRACTION_SETTINGS):
    """
    1. Ищет identifier в Реестре (df_registry)
    2. Получает оттуда уникальный cycle_id
    3. По этому ID находит все данные в Логах (df_logs)
    """
    # [Шаг 0] Инициализация колонок (чтобы не были пустыми)
    for col in ['p_tank_theory', 'p_residual_error']:
        if col not in df_logs.columns: df_logs[col] = np.nan
    if 'p_cycle_mae' not in df_registry.columns: df_registry['p_cycle_mae'] = np.nan

    # [Шаг 1] Ищем "зацепку" в Реестре
    mask_reg = _filter_mask(identifier, df_registry)
    
    if not mask_reg.any():
        print(f"!!! ОШИБКА: Цикл '{identifier}' не найден в Реестре (df_registry).")
        return df_logs, df_registry

    # [Шаг 2] Берем уникальный cycle_id из Реестра
    c_id = df_registry.loc[mask_reg, 'cycle_id'].values[0]

    # [Шаг 3] Теперь идем в Логи и берем всё, что относится к этому cycle_id
    # Здесь поиск идет только по одной колонке 'cycle_id', которая точно есть
    mask_logs = df_logs['cycle_id'] == c_id
    idx_logs = df_logs[mask_logs].sort_values('t_relative').index
    
    if len(idx_logs) == 0:
        print(f"!!! ОШИБКА: Данные для ID '{c_id}' не найдены в Логах (df_logs).")
        return df_logs, df_registry

    # --- ДАЛЕЕ РАСЧЕТ БЕЗ ИЗМЕНЕНИЙ ---
    cycle_data = df_logs.loc[idx_logs]
    dt = cycle_data['t_relative'].diff().mean()
    if pd.isna(dt) or dt <= 0: dt = 0.01

    shift_steps = int(flat_params["T_dead_time"] / dt)
    shifted_signal = cycle_data[signal_col].shift(shift_steps, fill_value=0.0).values
    
    state = {'P': 0.0, 'dP': 0.0, 'v_pos': 0.0}
    p_theory_results = []

    for val in shifted_signal:
        p_step = solve_tank_step(val, state, dt, flat_params)
        p_theory_results.append(p_step)

    df_logs.loc[idx_logs, 'p_tank_theory'] = p_theory_results

    # Расчет ошибки и запись MAE в реестр
    df_logs, mae_val = _calculate_cycle_error(df_logs, identifier, EXTRACTION_SETTINGS)
    df_registry.loc[mask_reg, 'p_cycle_mae'] = mae_val
    
    return df_logs, df_registry

# =============================================================================
# 4. РАСЧЕТ ОШИБКИ (Внутренняя)
# =============================================================================

def _calculate_cycle_error(df_logs, identifier, EXTRACTION_SETTINGS):
    """
    Считает MAE на интервале работы клапана порошка.
    Использует имена датчиков из config.py.
    """
    # Имена из конфига

    actual_p_col = EXTRACTION_SETTINGS["actual_p_col"]      # Опорное давление
    trigger_open  = EXTRACTION_SETTINGS["trigger_open"]     # Клапан порошка ОТКР
    trigger_close = EXTRACTION_SETTINGS["trigger_close"]    # Клапан порошка ЗАКР

    mask = _filter_mask(identifier, df_logs)
    cycle_data = df_logs[mask].sort_values('t_relative')

    # 1. Остаточная ошибка (Residual) для каждой точки
    df_logs.loc[mask, 'p_residual_error'] = (
        df_logs.loc[mask, actual_p_col] - df_logs.loc[mask, 'p_tank_theory']
    )

    # 2. Поиск интервала продувки для MAE
    open_idx = cycle_data[cycle_data[trigger_open] > 0.5].index
    close_idx = cycle_data[cycle_data[trigger_close] > 0.5].index
    
    if len(open_idx) > 0 and len(close_idx) > 0:
        start_mae = open_idx[0]
        end_mae = close_idx[close_idx > start_mae][0] if any(close_idx > start_mae) else cycle_data.index[-1]
        # Считаем средний модуль ошибки
        mae_value = df_logs.loc[start_mae:end_mae, 'p_residual_error'].abs().mean()
    else:
        # Если клапан порошка не дергался — считаем по всему циклу
        mae_value = df_logs.loc[mask, 'p_residual_error'].abs().mean()

    return df_logs, mae_value