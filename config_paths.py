# config_paths.py 
# =============================================================================
# КОНФИГУРАЦИЯ ПУТЕЙ И СТРУКТУРЫ ПРОЕКТА
"""
этот файл служит центральным местом для определения всех путей к данным, результатам и конфигурациям в проекте.
Он обеспечивает единообразие и удобство управления ресурсами, а также позволяет легко изменять структуру папок при необходимости. 

Структура папок
D:.
│   .gitignore
│   app_main.py
│   app_main_reference.py
│   config.py
│   main_AnaliticModel.ipynb
│   ТЗ на разработку.txt
│
├───data
│   │   Сборка логов.xlsx														- файл логов для поиска продувок по настройкам
│   │   ТК.20-500.895.1-7.4. ред.0 (настроечные параметры дезоксидации).csv		- используется в программе как заданные настройки для сопоставления статистики
│   │   ТК.20-500.895.1-7.4. ред.0 (настроечные параметры дезоксидации).xlsx	- исходный файл настроек
│   │
│   ├───config
│   │       Default_model_params.json											- стандартные параметры модели которые открываются по умолчанию в интерфейсе
│   │       gui_settings.json													- настройки интерфейса
│   ├───processed
│   │       cycles_avg_logs.csv
│   │       cycles_avg_registry.csv
│   │       cycles_logs.csv
│   │       cycles_registry.csv
│   │
│   ├───raw
│   └───results
│
├───gui
│       gui_components.py
│       gui_plotter.py
│
├───scripts
│       data_manager.py
│       import_data.py
│       optimizator.py
│       plot_manager.py
│       pressure_predictor.py
│       pressure_predictor_lite.py
│       pressure_reference_predictor.py
│    
└───Документация
        PlotDezoxidation.ipynb
        Сборка результатов.drawio
        ТЗ на разработку.txt

""" 
# =============================================================================
from pathlib import Path

# --- 1. КОРНЕВАЯ ДИРЕКТОРИЯ ПРОЕКТА ---
# Определяет местоположение данного файла и считает его корнем проекта
BASE_DIR = Path(__file__).resolve().parent

# --- 2. ВХОДНЫЕ ДАННЫЕ (INPUT) ---
# Место хранения сырых данных, поступающих с производства
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # Сюда кладем исходные .txt логи
REFERENCE_SETPOINT_FILE = DATA_DIR / "ТК.20-500.895.1-7.4. ред.0 (настроечные параметры дезоксидации).csv"

# --- 3. РЕЗУЛЬТАТЫ ОБРАБОТКИ (OUTPUT) ---
# Место хранения очищенных и структурированных данных после экстракции
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAF_RESALTS_DIR = DATA_DIR / "results"  # Сюда сохраняем графики из интерфейса
LOGS_FILE     = PROCESSED_DATA_DIR / "cycles_logs.csv"     # Временные ряды (датчики)
REGISTRY_FILE = PROCESSED_DATA_DIR / "cycles_registry.csv" # Реестр впрысков (метаданные)

LOGS_AVG_FILE     = PROCESSED_DATA_DIR / "cycles_avg_logs.csv"     # Временные ряды (датчики)
REGISTRY_AVG_FILE = PROCESSED_DATA_DIR / "cycles_avg_registry.csv" # Реестр впрысков (метаданные)


# --- 4. НАСТРОЙКИ МОДЕЛИ И КОНФИГУРАЦИЯ (CONFIG) ---
# Параметры физической модели, коэффициенты и оптимизированные значения
CONFIG_DIR = DATA_DIR / "config"
DEFAULT_MODEL_PARAMS_TANK_FILE = CONFIG_DIR / "Default_model_params_TANK.json"      # Коэффициенты K, T, Zeta...
DEFAULT_MODEL_PARAMS_PIPE_FILE = CONFIG_DIR / "Default_model_params_PIPE.json"      # Коэффициенты K, T, Zeta...
VALVE_SETTINGS_FILE = CONFIG_DIR / "valve_reference.csv"  # Таблица настроек ЧМИ

# --- 5. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ (UI) ---
# Все файлы, отвечающие за визуализацию и графическую оболочку
GUI_DIR     = BASE_DIR / "gui"
GUI_CONFIG_FILE   = CONFIG_DIR / "gui_settings.json"
# ASSETS_DIR = GUI_DIR / "assets"         # Иконки, логотипы, стили
# LAYOUT_DIR = GUI_DIR / "layouts"        # Описания окон и графиков


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def init_structure():
    """
    Создает все необходимые папки проекта, если они отсутствуют.
    Вызывается один раз при старте приложения.
    """
    folders = [
        RAW_DATA_DIR, 
        PROCESSED_DATA_DIR, 
        CONFIG_DIR, 
        GUI_DIR, 
        # ASSETS_DIR, 
        # LAYOUT_DIR
    ]
    
    print("--- Проверка структуры проекта ---")
    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"[СОЗДАНО] {folder.relative_to(BASE_DIR)}")
        else:
            print(f"[OK] {folder.relative_to(BASE_DIR)}")
    print("----------------------------------\n")


if __name__ == "__main__":
    # При запуске файла напрямую — просто создаем структуру
    init_structure()