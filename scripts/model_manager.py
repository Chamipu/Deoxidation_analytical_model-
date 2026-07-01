# -*- coding: utf-8 -*-
"""
Менеджер моделей для управления вызовами независимого и связанного прогнозирования.
"""
# Импорт новой связанной модели
from scripts.coupled_linear_predictor import pressure_predictor_coupled as coprm
# Импорт старой независимой модели
from scripts.independent_linear_predictor import pressure_predictor_lite as prm

def apply_model(df_logs, df_registry, config_tank, config_pipe, params_tank, params_pipe, model="coupled"):
    """
    Интерфейс для запуска расчетов. 
    Параметр model может принимать значения "coupled" или "independent".
    """
    if model == "coupled":
        # Запуск новой связанной модели
        df_logs, df_registry = coprm.apply_coupled_model(
            df_logs,
            df_registry,
            config_tank,
            config_pipe,
            params_tank,
            params_pipe
        )
    else:
        # Запуск старой независимой модели в два прохода
        df_logs, df_registry = prm.apply_analytic_model(
            df_logs, df_registry, config_tank, params_tank
        )
        df_logs, df_registry = prm.apply_analytic_model(
            df_logs, df_registry, config_pipe, params_pipe
        )
        
    return df_logs, df_registry