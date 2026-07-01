import copy
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from scripts.data_manager import get_flat_params
# Импортируем готовые функции из физико-математического ядра
from scripts.coupled_linear_predictor.pressure_predictor_coupled import (
    solve_coupled_system,
    calculate_interval_mae,
)

# =============================================================================
# ДЕТАЛЬНЫЕ НАСТРОЙКИ ГЛОБАЛЬНОГО ПОИСКА
# =============================================================================
# Временной интервал для оценки ошибки MAE
OPTIMIZATION_T_MIN = 0.0
OPTIMIZATION_T_MAX = 7.0

# Настройки глобального алгоритма Дифференциальной Эволюции
# DE_SETTINGS = {
#     'strategy': 'best1bin',      # Стратегия выбора кандидатов и скрещивания
#     'maxiter': 300,              # Максимальное количество поколений (итераций поиска)
#     'popsize':  50,               # Множитель размера популяции (размер = popsize * число параметров)
#     'tol': 0.001,                 # Критерий сходимости (остановка при малом изменении ошибки)
#     'mutation': (0.5, 1.0),      # Масштаб мутации (широта охвата пространства поиска) 0.5-1.0
#     'recombination': 0.7,        # Вероятность скрещивания параметров (кроссовер) 0.7
#     'polish': True,              # Локальный спуск в конце для сверхточной доводки параметров
#     'disp': False                # Отключение стандартных системных принтов scipy
# }

DE_SETTINGS = {
    'strategy': 'best1bin',      # Стратегия выбора кандидатов и скрещивания
    'maxiter': 300,              # Максимальное количество поколений (итераций поиска)
    'popsize':  100,               # Множитель размера популяции (размер = popsize * число параметров)
    'tol': 0.0001,                 # Критерий сходимости (остановка при малом изменении ошибки)
    'mutation': (0.5, 1.5),      # Масштаб мутации (широта охвата пространства поиска) 0.5-1.0
    'recombination': 0.8,        # Вероятность скрещивания параметров (кроссовер) 0.7
    'polish': True,              # Локальный спуск в конце для сверхточной доводки параметров
    'disp': False                # Отключение стандартных системных принтов scipy
}
# =============================================================================
# КЛАСС ЛОГИРОВАНИЯ ХОДА ОПТИМИЗАЦИИ
# =============================================================================
class OptimizationTracker:
    """
    Отслеживает лучшие значения целевой функции на шагах глобального поиска.
    """
    def __init__(self):
        self.iteration = 0
        self.best_loss = 9999.0

    def register_loss(self, loss):
        """Фиксирует новое рекордное значение ошибки."""
        self.best_loss = loss

    def callback(self, xk, convergence=None):
        """Вызывается алгоритмом после завершения каждого поколения популяции."""
        self.iteration += 1
        print(f"  Поколение {self.iteration:02d} | Текущая лучшая ошибка MAE: {self.best_loss:.5f} бар")


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ И СТРУКТУР
# =============================================================================

# def get_flat_params(config_params):
#     """Вытаскивает только 'value' из структуры конфига."""
#     flat_params = {}
#     for k, v in config_params.items():
#         if isinstance(v, dict) and 'value' in v:
#             flat_params[k] = v['value']
#         else:
#             flat_params[k] = v
#     return flat_params


def _extract_cycles_data(df_logs, CONFIG_TANK, CONFIG_PIPE):
    """
    Извлекает данные из DataFrame и группирует их по циклам (только массивы NumPy).
    Это избавляет расчет от накладных расходов Pandas во время подбора.
    """
    cycle_col = CONFIG_TANK['col_cycle']
    grouped = df_logs.groupby(cycle_col)
    
    cycles_data = []
    for _, df_cycle in grouped:
        cycles_data.append({
            'times': df_cycle[CONFIG_TANK['col_time']].values,
            'target_tank': df_cycle[CONFIG_TANK['col_target']].values,
            'target_pipe': df_cycle[CONFIG_PIPE['col_target']].values,
            'actual_tank': df_cycle[CONFIG_TANK['col_actual']].values,
            'actual_pipe': df_cycle[CONFIG_PIPE['col_actual']].values,
        })
    return cycles_data


def _prepare_optimization_vectors(config_tank, config_pipe, param_keys):
    """
    Формирует начальный вектор параметров x0, границы (bounds) и карту соответствия
    для активных параметров оптимизации на основе JSON структуры.
    """
    x0 = []
    bounds = []
    active_params_map = []  # Список кортежей: ('tank' или 'pipe', имя_параметра)

    if param_keys is None:
        param_keys_tank = [k for k, v in config_tank.items() if isinstance(v, dict) and 'value' in v]
        param_keys_pipe = [k for k, v in config_pipe.items() if isinstance(v, dict) and 'value' in v]
    else:
        param_keys_tank = [k for k in param_keys if k in config_tank]
        param_keys_pipe = [k for k in param_keys if k in config_pipe]

    for k in param_keys_tank:
        cfg = config_tank[k]
        x0.append(cfg['value'])
        bounds.append((cfg['min'], cfg['max']))
        active_params_map.append(('tank', k))

    for k in param_keys_pipe:
        cfg = config_pipe[k]
        x0.append(cfg['value'])
        bounds.append((cfg['min'], cfg['max']))
        active_params_map.append(('pipe', k))

    return np.array(x0), bounds, active_params_map


def _update_config_params(config_tank, config_pipe, best_x, active_params_map):
    """
    Создает копии исходных словарей конфигурации с обновленными значениями 'value'.
    """
    updated_tank = copy.deepcopy(config_tank)
    updated_pipe = copy.deepcopy(config_pipe)
    
    for val, (target, key) in zip(best_x, active_params_map):
        if target == 'tank':
            updated_tank[key]['value'] = val
        elif target == 'pipe':
            updated_pipe[key]['value'] = val
            
    return updated_tank, updated_pipe


# =============================================================================
# МАТЕМАТИЧЕСКАЯ ФУНКЦИЯ ПОТЕРЬ
# =============================================================================

def _objective_loss(x, active_params_map, flat_params_tank, flat_params_pipe, cycles_data, tracker=None):
    """
    Целевая функция для минимизации суммарной ошибки MAE бака и трубы по всем циклам.
    """
    curr_tank = flat_params_tank.copy()
    curr_pipe = flat_params_pipe.copy()
    
    # Обновляем значения только оптимизируемых в данном запуске параметров
    for val, (target, key) in zip(x, active_params_map):
        if target == 'tank':
            curr_tank[key] = val
        elif target == 'pipe':
            curr_pipe[key] = val
            
    total_mae = 0.0
    valid_cycles_count = 0
    
    for cycle in cycles_data:
        # Прямой расчет системы на NumPy
        p_tank_pred, p_pipe_pred = solve_coupled_system(
            times=cycle['times'],
            target_tank=cycle['target_tank'],
            target_pipe=cycle['target_pipe'],
            flat_params_tank=curr_tank,
            flat_params_pipe=curr_pipe
        )
        
        # Оценка MAE на заданном интервале
        mae_tank = calculate_interval_mae(
            cycle['times'], cycle['actual_tank'], p_tank_pred, 
            t_min=OPTIMIZATION_T_MIN, t_max=OPTIMIZATION_T_MAX
        )
        mae_pipe = calculate_interval_mae(
            cycle['times'], cycle['actual_pipe'], p_pipe_pred, 
            t_min=OPTIMIZATION_T_MIN, t_max=OPTIMIZATION_T_MAX
        )
        
        if not np.isnan(mae_tank) and not np.isnan(mae_pipe):
            total_mae += (mae_tank + mae_pipe)
            valid_cycles_count += 1
            
    loss = total_mae / valid_cycles_count if valid_cycles_count > 0 else 9999.0
    
    # Регистрируем ошибку в трекере, если она оказалась лучше предыдущих попыток
    if tracker is not None and loss < tracker.best_loss:
        tracker.register_loss(loss)
        
    return loss


# =============================================================================
# ГЛАВНЫЙ ОРКЕСТРАТОР ОПТИМИЗАЦИИ
# =============================================================================

def run_universal_optimizer(df_logs, df_registry, cnfg_params_tank, cnfg_params_pipe, 
                            CONFIG_PREDICTOR_TANK, CONFIG_PREDICTOR_PIPE, param_keys=None):
    """
    Глобальный оптимизатор настроек физической модели на базе Differential Evolution.
    """
    # 1. Извлекаем плоские параметры по умолчанию
    flat_params_tank = get_flat_params(cnfg_params_tank)
    flat_params_pipe = get_flat_params(cnfg_params_pipe)
    
    # 2. Подготавливаем начальный вектор и границы изменений параметров
    x0, bounds, active_params_map = _prepare_optimization_vectors(
        cnfg_params_tank, cnfg_params_pipe, param_keys
    )
    
    if len(bounds) == 0:
        print("Предупреждение: Список параметров для оптимизации пуст.")
        return df_logs, df_registry, cnfg_params_tank, cnfg_params_pipe
        
    # 3. Кешируем данные циклов в виде NumPy массивов
    cycles_data = _extract_cycles_data(df_logs, CONFIG_PREDICTOR_TANK, CONFIG_PREDICTOR_PIPE)
    
    # Инициализация трекера логирования
    tracker = OptimizationTracker()
    
    # Оценка начальной ошибки до оптимизации
    initial_loss = _objective_loss(x0, active_params_map, flat_params_tank, flat_params_pipe, cycles_data)
    tracker.register_loss(initial_loss)
    
    # Вывод информации о начале процесса
    print("=" * 80)
    print(" ЗАПУСК ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ (DIFFERENTIAL EVOLUTION)")
    print("=" * 80)
    print(f"Количество анализируемых циклов: {len(cycles_data)}")
    print(f"Размерность задачи (параметров): {len(bounds)}")
    print(f"Размер популяции (агентов):     {DE_SETTINGS['popsize'] * len(bounds)}")
    print(f"Начальная суммарная ошибка MAE:  {initial_loss:.5f} бар")
    print("Оптимизируемые параметры:")
    for val, (target, key) in zip(x0, active_params_map):
        cfg = cnfg_params_tank[key] if target == 'tank' else cnfg_params_pipe[key]
        print(f"  - [{target}] {key:<12} | Текущее: {val:<6} | Ограничения: [{cfg['min']}, {cfg['max']}]")
    print("-" * 80)
    print("Поиск глобального минимума...")

    # 4. Запуск глобальной эволюционной оптимизации
    res = differential_evolution(
        func=_objective_loss,
        bounds=bounds,
        args=(active_params_map, flat_params_tank, flat_params_pipe, cycles_data, tracker),
        strategy=DE_SETTINGS['strategy'],
        maxiter=DE_SETTINGS['maxiter'],
        popsize=DE_SETTINGS['popsize'],
        tol=DE_SETTINGS['tol'],
        mutation=DE_SETTINGS['mutation'],
        recombination=DE_SETTINGS['recombination'],
        polish=DE_SETTINGS['polish'],
        disp=DE_SETTINGS['disp'],
        callback=tracker.callback
    )
    
    # 5. Сборка обновленных конфигурационных словарей
    best_config_tank, best_config_pipe = _update_config_params(
        cnfg_params_tank, cnfg_params_pipe, res.x, active_params_map
    )
    
    final_loss = res.fun
    improvement_pct = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0
    
    # 6. Подробный финальный отчет в лог
    print("-" * 80)
    print(" РЕЗУЛЬТАТЫ ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ")
    print("-" * 80)
    print(f"Статус завершения:  {res.message}")
    print(f"Всего поколений:    {res.nit}")
    print(f"Вызовов модели:     {res.nfev}")
    print(f"Финальная ошибка:   {final_loss:.5f} бар (Улучшение: {improvement_pct:+.2f}%)")
    print("\nИзменение параметров:")
    for start_val, best_val, (target, key) in zip(x0, res.x, active_params_map):
        delta = best_val - start_val
        print(f"  - [{target}] {key:<12} : {start_val:.4f} -> {best_val:.4f} ({delta:+.4f})")
    print("=" * 80)
   
    return best_config_tank, best_config_pipe