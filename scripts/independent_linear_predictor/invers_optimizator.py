# optimizer.py
import numpy as np
import math
from scipy.optimize import differential_evolution
from scripts import data_manager as dm
from scripts.independent_linear_predictor import pressure_predictor_lite as prm

def optimize_presets(time_grid, goal_pressure, phases, flat_params, eval_window=(0.0, 6.0), injection_duration=6.0):
    """
    Оптимизатор подбирает уставки напрямую в физическом пространстве ПЛК [0..60],
    а в самом конце переводит их в проценты для пульта [0..100].
    Выводит детальный технический лог в процессе подбора.
    """
    
    # Счетчик итераций (поколений алгоритма)
    iteration = 0
    
    # 1. Функция оценки ошибки
    # def loss_function(targets_float):
    #     goal_set = np.round(targets_float).astype(int)
        
    #     target = dm.generate_step_profile(
    #         time_grid, 
    #         goal_set, 
    #         injection_duration, 
    #         phases, 
    #         dm.calculate_step_strategy
    #     )
        
    #     prediction = prm.analitic_model(time_grid, target, flat_params)
        
    #     mask = (time_grid >= eval_window[0]) & (time_grid <= eval_window[1])
    #     error = np.mean((np.array(prediction)[mask] - goal_pressure[mask]) ** 2)
    #     return error

    def loss_function(targets_float):
        goal_set = np.round(targets_float).astype(int)
        
        # 1. Расчет профиля и симуляция (как обычно)
        target = dm.generate_step_profile(time_grid, goal_set, injection_duration, phases, dm.calculate_step_strategy)
        prediction = prm.analitic_model(time_grid, target, flat_params)
        
        # 2. Обычная ошибка отклонения от цели
        mask = (time_grid >= eval_window[0]) & (time_grid <= eval_window[1])
        error = np.mean((np.array(prediction)[mask] - goal_pressure[mask]) ** 2)
        
        # 3. ШТРАФ ЗА ДИНАМИКУ (разность между соседними уставками)
        # np.diff(goal_set) считает: (уставка2 - уставка1), (уставка3 - уставка2) и т.д.
        # Умножаем сумму этих скачков на весовой коэффициент (например, 0.001)
        jump_penalty = np.sum(np.abs(np.diff(goal_set))) * 0.00001
        
        # Итоговая функция потерь — оптимизатор будет искать компромисс 
        # между точностью совпадения и гладкостью уставки
        return error + jump_penalty

    # 2. Функция вывода статуса (вызывается в конце каждого поколения)
    def monitor_status(xk, convergence=None):
        nonlocal iteration
        iteration += 1
        
        # Считаем ошибку для лучшего кандидата в этом поколении
        current_error = loss_function(xk)
        
        # Переводим текущие уставки ПЛК [0..60] в проценты пульта для наглядности в логе
        current_targets = np.round(xk).astype(int)
        current_presets = [int(math.ceil(t * 10 / 6)) for t in current_targets]
        
        print(f"Поколение {iteration:2d} | "
              f"Ошибка (MSE): {current_error:.6f} | "
              f"Лучшие пресеты на шаге: {current_presets}")

    # Сокращенные границы поиска [0..60]
    bounds = [(0, 60)] * 10
    
    print("=" * 85)
    print("СТАРТ ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ (SciPy Differential Evolution)")
    print(f"Диапазон поиска: уставки ПЛК [0..60] | Окно оценки ошибки: {eval_window} сек")
    print("=" * 85)
    
    # Запускаем глобальный поиск с подключенным колбэком monitor_status
    result = differential_evolution(
        loss_function,
        bounds,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=False,
        callback=monitor_status  # <--- Оптимизатор сам вызывает принт на каждом шаге
    )
    
    # Получаем лучшие уставки ПЛК [0..60]
    best_targets = np.round(result.x).astype(int)
    
    # Переводим их в минимальные проценты пульта [0..100]
    best_presets = [int(math.ceil(t * 10 / 6)) for t in best_targets]
    
    print("=" * 85)
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print(f"Минимальная итоговая ошибка (MSE): {result.fun:.6f}")
    print(f"Подобранные пресеты для пульта:   {best_presets}")
    print("=" * 85)
    
    return best_presets