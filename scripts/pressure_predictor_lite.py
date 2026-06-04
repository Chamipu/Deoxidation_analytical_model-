
import numpy as np

def apply_analytic_model(df_logs, df_registry, CONFIG_PREDICTOR, flat_params):
    # 1. Извлекаем данные
    times = df_logs[CONFIG_PREDICTOR['col_time']].values
    targets = df_logs[CONFIG_PREDICTOR['col_target']].values
    
    predictions = []
    p_current = 0.0
    dt = 0.2 
    
    for i in range(len(times)):

        k_gain = (flat_params['k_gain']*p_current)+flat_params['b_gain']
        target = _get_delayed_target(i, times, targets, flat_params['dead_time'])
        # ШАГ 4: Пересчитываем физику
        p_current = _predict_pressure(
            p_current, 
            target, # Подаем реальную уставку (без задержки) для проверки потенциала модели
            dt, 
            flat_params['damping'], 
            k_gain
        )
        predictions.append(p_current)
        
        dt = times[i] - times[i-1]

    df_logs[CONFIG_PREDICTOR['col_result']] = predictions

    df_logs, df_registry = _calculate_and_save_errors(df_logs, df_registry, CONFIG_PREDICTOR)
    return df_logs, df_registry

def _get_delayed_target(i, times, targets, dead_time):
    # Берем "окно" в 10-20 строк назад от текущего индекса i
    # Этого гарантированно хватит для покрытия любой разумной задержки
    start = max(0, i - 20)
    end = i + 1
    
    # Интерполируем только внутри текущего окна (конкретного цикла)
    return np.interp(
        times[i] - dead_time, 
        times[start:end], 
        targets[start:end]
    )

def _predict_pressure(p_prev, target_p, dt, damping, k_gain):
    """

    """
    if dt >= 1.0: return 0
    
    # 3. Формула дифура первого порядка:
    # P_new = P_prev + (Изменение_давления_за_шаг)
    # Изменение = ( (Цель * Усиление) - Текущее_давление ) * (Время / Инерция)
    p_new = p_prev + (dt / damping) * (k_gain * target_p - p_prev)
            
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