from scipy.optimize import differential_evolution
import time
from scripts import pressure_predictor_lite as prm

def run_universal_optimizer(df_logs, df_registry, config_full, CONFIG_PREDICTOR, param_keys):
    """
    param_keys: список строк, например ["dead_time", "k_gain", "b_gain", "damping"]
    config_full: ваш словарь с min, max, value
    """
    
    # 1. АВТО-СБОР ГРАНИЦ
    # Мы смотрим в конфиг и берем (min, max) только для тех ключей, что передали
    bounds = [(config_full[k]["min"], config_full[k]["max"]) for k in param_keys]
    
    print(f"--- ОПТИМИЗАЦИЯ ПАРАМЕТРОВ: {param_keys} ---")

    # 2. УНИВЕРСАЛЬНАЯ ЦЕЛЕВАЯ ФУНКЦИЯ
    def objective(x):
        # 1. Собираем параметры из вектора x
        current_params = {param_keys[i]: x[i] for i in range(len(param_keys))}
        
        # 2. Вызываем вашу функцию (она сама всё посчитает и запишет в реестр)
        # Мы используем копии .copy(), чтобы оптимизатор не "затирал" основной 
        # датафрейм промежуточными (плохими) результатами в процессе подбора.
        _, temp_registry = prm.apply_analytic_model(
            df_logs.copy(), 
            df_registry.copy(), 
            CONFIG_PREDICTOR, 
            current_params
        )
        
        # 3. Просто берем среднее значение ошибки из колонки, которую создала функция
        score = temp_registry[CONFIG_PREDICTOR['col_MAE']].mean()
        
        return score

    # 3. НАСТРОЙКИ АЛГОРИТМА (подробнее ниже)
    optimizer_settings = {
        'popsize': 15,        # Кол-во "особей" на каждый параметр
        'maxiter': 40,        # Макс кол-во поколений
        'tol': 0.01,          # Относительная точность для остановки
        'mutation': (0.5, 1), # Степень "разброса" при поиске
        'recombination': 0.7  # Вероятность смешивания параметров
    }

    start_time = time.time()

    # 4. ЗАПУСК
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin',
        **optimizer_settings,
        callback=lambda xk, conv: print(f" Шаг: {conv:.1%}, MAE: {objective(xk):.5f}")
    )
    
    total_time = time.time() - start_time
    print_optimization_report(result, param_keys, bounds, total_time)

    # 1. Получаем плоский словарь результатов
    optimized_flat = {param_keys[i]: result.x[i] for i in range(len(param_keys))}
    
    # 2. Создаем полный конфиг со всеми описаниями
    final_config_full = get_optimized_config_full(config_full, optimized_flat)

    # 5. СБОР РЕЗУЛЬТАТОВ
   
    return final_config_full, result

import copy

def get_optimized_config_full(config_full, optimized_flat_params):
    """
    Объединяет результаты оптимизации с исходными метаданными конфига.
    
    :param config_full: Исходный словарь с описаниями (JSON-like)
    :param optimized_flat_params: Словарь {key: new_value}
    :return: Новый словарь в формате исходного JSON
    """
    # 1. Делаем глубокую копию, чтобы не менять оригинал случайно
    updated_config = copy.deepcopy(config_full)
    
    # 2. Обновляем только числовые значения
    for key, new_val in optimized_flat_params.items():
        if key in updated_config:
            updated_config[key]["value"] = float(new_val)
            
    return updated_config

def print_optimization_report(result, param_keys, bounds, duration):
    """
    Печатает детальный разбор результатов оптимизации.
    
    :param result: Объект OptimizeResult из scipy.optimize
    :param param_keys: Список имен параметров (L, K, B, Z...)
    :param bounds: Список кортежей (min, max) для проверки границ
    :param duration: Затраченное время в секундах
    """
    print("\n" + "="*55)
    print("📋 ДЕТАЛЬНЫЙ ОТЧЕТ ОБ ОПТИМИЗАЦИИ")
    print("="*55)

    # 1. Основные сведения
    status_emoji = "✅ УСПЕШНО" if result.success else "⚠️ ЗАВЕРШЕНО ПО ЛИМИТУ"
    print(f"{'Статус оптимизации:':<25} {status_emoji}")
    print(f"{'Причина остановки:':<25} {result.message}")
    print(f"{'Общее время:':<25} {duration:.2f} сек")
    
    print("-" * 55)
    
    # 2. Метрики эффективности
    print(f"{'Лучшее MAE (ошибка):':<25} {result.fun:.6f} бар")
    print(f"{'Кол-во поколений:':<25} {result.nit}")
    print(f"{'Всего расчетов модели:':<25} {result.nfev} раз")
    
    print("-" * 55)
    
    # 3. Анализ найденных параметров
    print(f"{'Параметр':<15} | {'Значение':^10} | {'Границы [min, max]':^20}")
    print("-" * 55)
    
    for i, key in enumerate(param_keys):
        val = result.x[i]
        b_min, b_max = bounds[i]
        
        # Проверка на достижение границ (в пределах 1%)
        margin = (b_max - b_min) * 0.01
        limit_note = ""
        if val <= b_min + margin: limit_note = " 🚩 [MIN!]"
        if val >= b_max - margin: limit_note = " 🚩 [MAX!]"
        
        # Формируем строку: Имя | Значение | [min, max] | Приметка
        bounds_str = f"[{b_min:.3f}, {b_max:.3f}]"
        print(f"{key:<15} | {val:^10.4f} | {bounds_str:^20} {limit_note}")
    
    print("="*55)
    if not result.success:
        print("СОВЕТ: Если MAE высокое или достигнут MAX!, попробуйте расширить\n"
              "границы параметров в JSON или увеличить popsize/maxiter.")
    print("="*55 + "\n")