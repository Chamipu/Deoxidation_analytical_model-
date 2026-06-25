import numpy as np
import pandas as pd
from scripts.independent_linear_predictor import pressure_predictor_lite as prm
from scripts import data_manager as dm
from scripts.independent_linear_predictor import invers_optimizator as iopt

SIMULATION_DT = 0.2
TIME_GRID_START = -20.0
TIME_GRID_END = 20.0

# =============================================================================
# ПУБЛИЧНЫЙ API (GUI, ноутбуки)
# =============================================================================

def make_time_grid(dt=SIMULATION_DT):
    """Единая временная сетка для синтетических расчётов."""
    return np.arange(TIME_GRID_START, TIME_GRID_END + dt, dt)


def build_goal_profile_preview(presets, injection_duration, phases):
    """
    Строит желаемый сглаженный профиль без оптимизации.
    Возвращает time_grid, goal_pressure_raw и координаты опорных точек по времени.
    """
    time_grid, goal_pressure_raw, df_registry = _init_synthetic_profile(
        presets, injection_duration, phases
    )
    return time_grid, goal_pressure_raw, df_registry['goal_time'].values


def run_preset_optimization(presets, injection_duration, flat_params, phases):
    """
    Полный пайплайн подбора уставок ПЛК: цель → инерция бака → DE-оптимизация → прогноз.
    Возвращает словарь с массивами для отрисовки и списком best_presets (% пульта).
    """
    time_grid, goal_pressure_raw, df_registry = _init_synthetic_profile(
        presets, injection_duration, phases
    )
    goal_pressure = _simulate_tank_dynamics(
        time_grid, goal_pressure_raw, injection_duration, flat_params
    )
    best_presets = iopt.optimize_presets(
        time_grid,
        goal_pressure,
        phases,
        flat_params,
        eval_window=(0.0, injection_duration),
        injection_duration=injection_duration,
    )
    target_profile = _build_plc_target_profile(
        time_grid, best_presets, injection_duration, phases
    )
    prediction = prm.analitic_model(time_grid, target_profile, flat_params)

    return {
        "time_grid": time_grid,
        "goal_raw": goal_pressure_raw,
        "goal_pressure": goal_pressure,
        "best_presets": best_presets,
        "target": target_profile,
        "prediction": prediction,
        "preset_times": df_registry['goal_time'].values,
    }


def generate_base_logs_df(presets, injection_duration, CONFIG_GENERATE_TARGET, flat_params):
    """
    Главный оркестратор: генерирует синтетическую среду, рассчитывает переходные процессы,
    подбирает настройки управления (пресеты) и формирует финальный отчет.
    """
    phases = CONFIG_GENERATE_TARGET['phases']
    result = run_preset_optimization(presets, injection_duration, flat_params, phases)

    df_logs = _assemble_final_dataframe(
        result["time_grid"],
        result["goal_raw"],
        result["goal_pressure"],
        injection_duration,
        CONFIG_GENERATE_TARGET,
    )
    df_registry = pd.DataFrame({
        'goal_time': result["preset_times"],
        'goal_pressure_presets': presets,
    })
    return df_logs, df_registry, result["best_presets"]

# =============================================================================
# БЛОК1: ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОЙ СРЕДЫ И РАСЧЕТ ЦЕЛЕВОГО ГРАФИКА
# =============================================================================

# ШАГ 1: Инициализация временной шкалы и идеального задания
# =============================================================================
def _init_synthetic_profile(presets, injection_duration, phases):
    """
    Генерирует исходную временную сетку, идеальную уставку и реестр для графиков.

    Принимает:
        presets (list): Набор целевых давлений для активного цикла.
        injection_duration (float): Длительность активной фазы.
        phases (dict): Временные границы фаз (pre_charge, dip_start, t_start).

    Возвращает:
        time_grid (np.ndarray): Временная сетка от -20 до +20 сек.
        goal_pressure_raw (np.ndarray): Идеальный целевой график.
        df_registry (pd.DataFrame): Таблица меток для графиков.
    """
    time_grid = make_time_grid()
    
    # Расчет уставки по новой универсальной функции
    goal_pressure_raw = dm.generate_step_profile(
        t_array=time_grid,
        presets=presets,
        duration=injection_duration,
        phases=phases,
        active_phase_calculator=dm.calculate_smooth_strategy
    )

    x_active_phases = np.linspace(phases['t_start'], phases['t_start'] + injection_duration, len(presets))
    df_registry = pd.DataFrame({
        'goal_time': x_active_phases,
        'goal_pressure_presets': presets,
    })

    return time_grid, goal_pressure_raw, df_registry

# ШАГ 2: Симуляция динамики бака (инерция преднадува и провала)
# =============================================================================

def _simulate_tank_dynamics(time_grid, goal_pressure_raw, injection_duration, flat_params):
    """
    Моделирует физический отклик давления в баке с учетом инерции.

    Принимает:
        time_grid (np.ndarray): Временная сетка.
        goal_pressure_raw (np.ndarray): Целевая уставка.
        injection_duration (float): Длительность активной фазы.
        flat_params (dict): Физические коэффициенты модели (damping, k_gain, b_gain).

    Возвращает:
        goal_pressure (np.ndarray): Сглаженный физический график давления.
    """
    dt = SIMULATION_DT
    goal_pressure = np.zeros_like(time_grid)
    damping = flat_params['damping']
    k_gain, b_gain = 0.1, 1  # Базовые коэффициенты ПЛК
    p_current = 0.0

    for idx, t in enumerate(time_grid):
        if t < 0 or t > injection_duration:
            p_current = prm.predict_pressure(
                p_prev=p_current,
                target_p=goal_pressure_raw[idx],
                dt=dt,
                damping=damping,
                k_gain=k_gain,
                b_gain=b_gain
            )
        else:
            p_current = goal_pressure_raw[idx]
        goal_pressure[idx] = p_current

    return goal_pressure

def _build_plc_target_profile(time_grid, best_presets_percent, injection_duration, phases):
    """Переводит проценты пульта в уставки ПЛК и строит ступенчатый профиль."""
    goal_set_plc = np.floor(np.array(best_presets_percent) * 0.6)
    return dm.generate_step_profile(
        time_grid,
        goal_set_plc,
        injection_duration,
        phases,
        dm.calculate_step_strategy,
    )

# Шаг 3: Оптимизация ПЛК и построение прогноза для активной фазы (greedy, legacy)
# =============================================================================

def _optimize_and_predict_active_phase(time_grid, goal_pressure, injection_duration, flat_params):
    """
    Запускает оптимизатор подбора пресетов для активной фазы и строит прогноз давления.

    Принимает:
        time_grid (np.ndarray): Полная временная сетка.
        goal_pressure (np.ndarray): Рассчитанное физическое давление в баке.
        injection_duration (float): Длительность активной фазы.
        flat_params (dict): Физические коэффициенты модели.

    Возвращает:
        goal_preset (list): 10 подобранных оптимальных процентов для пульта ПЛК.
        p_predicted_active (np.ndarray): Прогноз давления активной фазы по этим пресетам.
    """
    # Выделяем активную фазу времени и давления
    mask_active = (time_grid >= 0.0) & (time_grid <= injection_duration)
    time_active = time_grid[mask_active]
    goal_active = goal_pressure[mask_active]

    # Точка старта (физическое давление при t=0.0)
    idx_t_zero = np.searchsorted(time_grid, 0.0)
    p_start_physical = goal_pressure[idx_t_zero]

    # Подбор пресетов оптимизатором и расчет прогноза
    goal_preset = find_presets_greedy(time_active, goal_active, p_start_physical, flat_params)
    # p_predicted_active = predict_active_phase(time_active, goal_preset, p_start_physical, flat_params)

    return goal_preset

# Шаг 4: Сборка результатов в финальный DataFrame логов
# =============================================================================

def _assemble_final_dataframe(time_grid, goal_pressure_raw, goal_pressure, injection_duration, CONFIG_GENERATE_TARGET):
    """
    Упаковывает все рассчитанные кривые и служебные метаданные в итоговый датафрейм логов.

    Принимает:
        time_grid, goal_pressure_raw, goal_pressure (np.ndarray): Базовые кривые.
        p_predicted_active (np.ndarray): Прогноз активной фазы.
        injection_duration (float): Длительность активной фазы.
        CONFIG_GENERATE_TARGET (dict): Конфигурация колонок.

    Возвращает:
        df_logs (pd.DataFrame): Итоговая таблица логов.
    """
    col_time = CONFIG_GENERATE_TARGET['columns']['col_time']
    
    df_logs = pd.DataFrame({
        col_time: time_grid,
        'goal_pressure_raw': goal_pressure_raw,
        'goal_pressure': goal_pressure
    })
    
    # Записываем прогноз по маске активной фазы
    mask_active = (time_grid >= 0.0) & (time_grid <= injection_duration)
    df_logs['predicted_pressure'] = 0.0
    # df_logs.loc[mask_active, 'predicted_pressure'] = p_predicted_active
    
    df_logs[CONFIG_GENERATE_TARGET['columns']['col_cycle']] = "CALCULATED_CYCLE"
    
    return df_logs

# =============================================================================
# БЛОК2: ИТТЕРАЦИОННЫЙ ПОДБОР НАСТРОЕК ПУЛЬТА ДЛЯ АКТИВНОЙ ФАЗЫ
# =============================================================================

def find_presets_greedy(time_grid, goal_pressure, p_start_physical, flat_params):
    """
    Пошагово подбирает 10 настроек пульта управления (0-100%) по желаемому графику.

    Принимает:
        time_grid (np.ndarray): Временная сетка активной фазы (шаг 0.2 с).
        goal_pressure (np.ndarray): Желаемый график давления на этой сетке.
        p_start_physical (float): Начальное давление в системе перед стартом.
        flat_params (dict): Физические коэффициенты модели.

    Возвращает:
        presets (list): Список из 10 целых чисел (оптимальные проценты для пульта).
    """
    presets = []
    # p_current = p_start_physical
    p_current = 0
    
    # 1. Разбиваем индексы временного массива на 10 равных частей (фаз)
    phase_slices = np.array_split(np.arange(len(time_grid)), 10)
    
    # 2. Последовательно ищем лучший процент для каждого из 10 шагов
    for phase_idx, indices in enumerate(phase_slices):
        best_percent, p_current, min_error = _find_best_percent_for_phase(
            p_current, indices, goal_pressure, flat_params
        )
        
        print(f"Фаза {phase_idx+1}: Лучший процент = {best_percent}%, "
              f"Ошибка = {min_error:.4f}, Давление в конце фазы = {p_current:.4f} бар")
        
        presets.append(best_percent)
        
    return presets

def _evaluate_candidate_percent(percent, p_start, indices, goal_pressure, flat_params):
    """
    Симулирует физику процесса для ОДНОГО конкретного процента на текущем шаге 
    и рассчитывает среднюю ошибку симуляции.

    Принимает:
        percent (int): Тестируемый процент пульта (0-100).
        p_start (float): Стартовое давление в начале этой фазы.
        indices (np.ndarray): Индексы точек времени текущей фазы.
        goal_pressure (np.ndarray): Желаемый график давления.
        flat_params (dict): Физические коэффициенты модели.

    Возвращает:
        mean_error (float): Средняя ошибка симуляции при данном проценте.
        p_sim (float): Конечное давление в баке в конце фазы.
    """
    if indices[0] < 1: dt = 5 
    else: dt = 0.2
    # dt = 0.2
    damping = flat_params['damping']
    k_gain = flat_params['k_gain']
    b_gain = flat_params['b_gain']

    # Переводим процент в реальное округленное давление на ПЛК (в барах)
    p_target = np.floor(percent * 6/10)
    
    p_sim = p_start
    phase_errors = []
    
    for idx in indices:
        p_sim = prm.predict_pressure(
            p_prev=p_sim,
            target_p=p_target,
            dt=dt,
            damping=damping,
            k_gain=k_gain,
            b_gain=b_gain
        )
        phase_errors.append(abs(p_sim - goal_pressure[idx]))
        
    return np.mean(phase_errors), p_sim


def _find_best_percent_for_phase(p_start, indices, goal_pressure, flat_params):
    """
    Перебирает все возможные управляющие сигналы (от 0% до 100%) для одной фазы 
    и находит наилучший вариант с минимальной ошибкой.

    Принимает:
        p_start (float): Стартовое давление в начале фазы.
        indices (np.ndarray): Индексы точек времени текущей фазы.
        goal_pressure (np.ndarray): Желаемый график давления.
        flat_params (dict): Физические коэффициенты модели.

    Возвращает:
        best_percent (int): Оптимальный процент для этой фазы.
        best_p_end (float): Конечное давление при лучшей настройке.
        min_error (float): Ошибка симуляции для лучшего процента.
    """
    best_percent = 0
    min_error = float('inf')
    best_p_end = p_start

    # Перебираем варианты на пульте
    for percent in range(0, 101):
        mean_error, p_end = _evaluate_candidate_percent(
            percent, p_start, indices, goal_pressure, flat_params
        )
        
        if mean_error < min_error:
            min_error = mean_error
            best_percent = percent
            best_p_end = p_end

    return best_percent, best_p_end, min_error


# =============================================================================
# БЛОК3: РАСЧЕТ ФИНАЛЬНОГО ГРАФИКА ПО ПОДОБРАННЫМ ЗНАЧЕНИЯМ 
# =============================================================================

def predict_active_phase(time_active, presets, p_start, model_params):
    """
    Рассчитывает финальный график давления для активной фазы по найденным уставкам.
    """
    p_predicted = np.zeros_like(time_active)
    p_current = p_start
    dt = 0.2
    
    damping = model_params['damping']
    k_gain_base = model_params['k_gain']
    b_gain = model_params['b_gain']
    
    # Разбиваем временную сетку на 10 фаз
    phase_slices = np.array_split(np.arange(len(time_active)), 10)
    
    for phase_idx in range(10):
        indices = phase_slices[phase_idx]
        # Переводим уставку этой фазы в бары
        percent = presets[phase_idx]
        p_target = np.floor(percent * 6/10)
        
        # Симулируем этот отрезок времени
        for idx in indices:

            p_current = prm.predict_pressure(
                p_prev=p_current,
                target_p=p_target,
                dt=dt,
                damping=damping,
                k_gain=k_gain_base,
                b_gain=b_gain
            )
            p_predicted[idx] = p_current

    return p_predicted