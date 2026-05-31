import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =============================================================================
# 1. СИНТЕЗ УПРАВЛЯЮЩЕГО СИГНАЛА
# =============================================================================

def get_target_value(t, duration, presets):
    """
    Рассчитывает уставку давления для момента времени t.
    presets: список/массив из 10 значений [p1, ..., p10]

    t - текущее время (сек)
    duration - длительность в секундах
    presets - массив давлений в барах
    """    
    # 0. До "накачки" (ранее -15с)
    if t < -15:
        return 0.0
          
    # 1. Период до начала впрыска (уже надуто до p1)
    if t < -0.3:
        return presets[0]

    if -0.3 <= t < 0:
        return 0
        
    # 2. Период активного впрыска (интерполяция)
    if t <= duration:
        # Локальное время внутри цикла (от 0 до duration)
        local_t = t
        # Равномерно распределяем 10 точек по времени duration
        x_phases = np.linspace(0, duration, 10)
        # Линейно интерполируем значение
        return np.interp(local_t, x_phases, presets)
    
    # 3. После окончания цикла - сброс
    return 0.0

# =============================================================================
# 2. ФИЗИЧЕСКОЕ ЯДРО (Универсальное)
# =============================================================================

def solve_dynamics(target_p, state, dt, par):
    # 1. Позиция клапана
    dv = (target_p - state['v_pos']) / (par["valve_time"] + 1e-6)
    state['v_pos'] += dv * dt
    
    # 2. Динамика давления
    # par["time_const"] и par["damping"] теперь реально отвечают за наклон
    d2P = (state['v_pos'] - 2 * par["damping"] * par["time_const"] * state['dP'] - state['P']) / (par["time_const"]**2)
    
    state['dP'] += d2P * dt
    state['P']  += state['dP'] * dt
    
    # 3. Математическая "резкость" (тоже на чистых разницах)
    gap = abs(target_p - state['P'])
    current_sharp = min(par["sharp_base"] + (gap * par["sharp_sens"]), par["sharp_max"])
    
    # Обновляем внутреннее состояние P для следующего шага
    # (Не возвращаем p_result, а обновляем его внутри state, 
    # чтобы d2P на следующем шаге dt видел корректную инерцию)
    state['P_with_sharp'] = state['P'] + (current_sharp * par["time_const"] * state['dP'])

# =============================================================================
# 3. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def get_cycle_model(df_logs, df_registry, df_ref_setpoint, identifier, par):
    """
    Полностью переписанная логика на основе настроек оператора.
    """
    
    # [Шаг 1] Поиск Dia/L в Registry
    reg_row = df_registry[df_registry['cycle_id'] == identifier]
    if reg_row.empty:
        print(f"ID {identifier} не найден в реестре")
        return df_logs, df_registry
    
    dia = int(reg_row['Dia_Shell'].values[0])
    length = int(reg_row['L_Shell'].values[0])
    
    # [Шаг 2] Поиск пресетов в Settings
    # Ищем строку, где совпадают параметры гильзы
    set_row = df_ref_setpoint[(df_ref_setpoint['Dia_Shell'] == dia) & (df_ref_setpoint['L_Shell'] == length)]
    if set_row.empty:
        print(f"Настройки для Dia {dia} L {length} не найдены")
        return df_logs, df_registry
    
    set_row = set_row.iloc[0]
    
    # Собираем 10 точек и время цикла
    # presets_tank = [float(set_row[f'p_tank_preset_{i}']) * 6 / 100 for i in range(1, 11)]
    presets_tank = [float(set_row[f'p_tank_preset_{i}']) * 0.02646 for i in range(1, 11)]
    # presets_pipe = [set_row[f'p_pipe_preset_{i}'] for i in range(1, 11)] # на будущее
    duration = float(set_row['Time_cycle']) / 1000.0

    # [Шаг 3] Подготовка данных в Логах
    mask_logs = df_logs['cycle_id'] == identifier
    idx_logs = df_logs[mask_logs].sort_values('t_relative').index
    
    if len(idx_logs) == 0: return df_logs, df_registry

    # Начальное состояние: перед 0 сек бак уже "надут" до p1
    p1 = 0
    state_tank = {'P': p1, 'dP': 0.0, 'v_pos': p1}
    
    theory_results = []

    # [Шаг 4] Итерационный расчет по существующей сетке t_relative
    t_values = df_logs.loc[idx_logs, 't_relative'].values
    
    for i in range(len(t_values)):
        t_curr = t_values[i]
        t_prev = t_values[i-1] if i > 0 else t_curr - 0.2
        dt = t_curr - t_prev
        if dt <= 0: dt = 0.2
        
        # Получаем уставку для текущего момента времени
        target_p = get_target_value(t_curr - par['dead_time'], duration, presets_tank)
        # print(f"t={t_curr:.2f}s, target_p={target_p:.2f} bar, dt={dt:.2f}s")
        # Для точности: прогоняем внутренний цикл (5 шагов внутри одного dt лога)
        # Чтобы физика не "сломалась" на dt=0.2
        sub_steps = 5
        sub_dt = dt / sub_steps
        # p_step = state_tank['P']
        
        for _ in range(5):
            # Передаем "чистый" target_p
            solve_dynamics(target_p, state_tank, sub_dt, par)

        p_final_raw = state_tank['P']
        theory_results.append(p_final_raw * par["k_gain"])

    # Запись результатов
    df_logs.loc[idx_logs, 'p_tank_theory'] = theory_results
    df_logs.loc[idx_logs, 'p_tank_target'] = [get_target_value(t, duration, presets_tank) for t in t_values]
    print(df_logs.loc[idx_logs, ['t_relative', 'p_tank_target', 'p_tank_theory', 'cycle_id']].head(10))
    return df_logs, df_registry