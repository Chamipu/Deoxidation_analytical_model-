# pressure_predictor_lite.py 
import numpy as np

# def apply_analytic_model(df_logs, df_registry, CONFIG_PREDICTOR, flat_params):
#     # 1. Извлекаем данные
#     times = df_logs[CONFIG_PREDICTOR['col_time']].values
#     targets = df_logs[CONFIG_PREDICTOR['col_target']].values
    
#     predictions = []
#     p_current = 0.0
#     dt = 0.2 
    
#     for i in range(len(times)):

#         # k_gain = (flat_params['k_gain']*p_current)+flat_params['b_gain']
#         target = _get_delayed_target(i, times, targets, flat_params['dead_time'])
#         # ШАГ 4: Пересчитываем физику
#         p_current = predict_pressure(
#             p_current, 
#             target, # Подаем реальную уставку (без задержки) для проверки потенциала модели
#             dt, 
#             flat_params['damping'], 
#             flat_params['k_gain'], 
#             flat_params['b_gain'], 
#         )
#         predictions.append(p_current)
        
#         dt = times[i] - times[i-1]

#     df_logs[CONFIG_PREDICTOR['col_result']] = predictions

#     df_logs, df_registry = _calculate_and_save_errors(df_logs, df_registry, CONFIG_PREDICTOR)
#     return df_logs, df_registry

def apply_analytic_model(df_logs, df_registry, CONFIG_PREDICTOR, flat_params):
    # 1. Извлекаем данные
    time_grid = df_logs[CONFIG_PREDICTOR['col_time']].values
    target = df_logs[CONFIG_PREDICTOR['col_target']].values
    
    predictions = analitic_model(time_grid, target, flat_params)
    df_logs[CONFIG_PREDICTOR['col_result']] = predictions

    df_logs, df_registry = _calculate_and_save_errors(df_logs, df_registry, CONFIG_PREDICTOR)
    return df_logs, df_registry

def analitic_model (time_grid, target, flat_params):
    predictions = []
    p_current = 0.0
    dt = 0.2 
    
    for i in range(len(time_grid)):

        # 1. Записываем результат в НОВУЮ переменную delayed_target
        delayed_target = _get_delayed_target(i, time_grid, target, flat_params['dead_time'], dt)
        
        # 2. Подаем delayed_target в расчет физики
        p_current = predict_pressure(
            p_current, 
            delayed_target, # <--- ПЕРЕДАЕМ НОВУЮ ПЕРЕМЕННУЮ
            dt, 
            flat_params['damping'], 
            flat_params['k_gain'], 
            flat_params['b_gain'], 
        )

        predictions.append(p_current)
        
        dt = time_grid[i] - time_grid[i-1]

    return predictions

def _get_delayed_target(i, times, targets, dead_time, dt):
    # Берем "окно" в 10-20 строк назад от текущего индекса i
    # Этого гарантированно хватит для покрытия любой разумной задержки
    steps_needed = int(np.ceil(dead_time / dt)) + 5

    start = max(0, i - steps_needed)
    end = i + 1
    
    # Интерполируем только внутри текущего окна (конкретного цикла)
    return np.interp(
        times[i] - dead_time, 
        times[start:end], 
        targets[start:end]
    )

def predict_pressure(p_prev, target_p, dt, damping, k_gain, b_gain):
    """

    """
    if dt >= 20.0: return 0
    
    # 3. Формула дифура первого порядка:
    # P_new = P_prev + (Изменение_давления_за_шаг)
    # Изменение = ( (Цель * Усиление) - Текущее_давление ) * (Время / Инерция)

    k_gain_base = k_gain*p_prev+b_gain
    p_new = p_prev + (dt / damping) * (k_gain_base * target_p - p_prev)
            
    return p_new

def _calculate_and_save_errors(df_logs, df_registry, CONFIG_PREDICTOR):
    """
    Рассчитывает ошибку для каждого цикла и записывает результат в таблицу регистров.
    """
    # 1. Считаем абсолютную разницу для каждой строки в логах
    # (Текущая ошибка в конкретный момент времени)

    df_logs[CONFIG_PREDICTOR['col_MAE']] = (df_logs["IBA_DB\PT1009 Актуальное давление в баке P1 (бар)_median"] - df_logs[CONFIG_PREDICTOR['col_result']]).abs()

    # 2. Группируем по cycle_id, но только те строки, где время от 0 до 6 секунд
    # Фильтр накладывается перед groupby
    cycle_errors = df_logs[
        (df_logs[CONFIG_PREDICTOR['col_time']] >= 3) & (df_logs[CONFIG_PREDICTOR['col_time']] <= 7)
    ].groupby(CONFIG_PREDICTOR['col_cycle'])[CONFIG_PREDICTOR['col_MAE']].mean()

    # 3. Переносим значения из полученной серии в df_registry
    # map сопоставит cycle_id в регистре с ключами в cycle_errors
    df_registry[CONFIG_PREDICTOR['col_MAE']] = df_registry[CONFIG_PREDICTOR['col_cycle']].map(cycle_errors)

    return df_logs, df_registry