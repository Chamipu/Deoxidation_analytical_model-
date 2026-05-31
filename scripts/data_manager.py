# -*- coding: utf-8 -*-
import pandas as pd
import os
import json
from pathlib import Path
import numpy as np


# =============================================================================
# БЛОК 1: РАБОТА С КОЛОНКАМИ И БАЗОЙ (CSV)
# =============================================================================

def show_columns_list(df):
    """Выводит названия полей. Работает с любым DF (логи или реестр)."""
    if df is None: return
    print("\n--- Список полей ---")
    for col in df.columns:
        print(f'"{col}",')

def load_database(csv_path):
    """
    Универсальная загрузка. Теперь понимает объекты Path из config.py.
    """
    # Превращаем в Path, если пришла строка
    csv_path = Path(csv_path)
    
    if csv_path.exists():
        print(f"Загрузка: {csv_path.name}")
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
        
        # Автоматическое восстановление времени, если колонка есть
        if 'cycle_start_time' in df.columns:
            df['cycle_start_time'] = pd.to_datetime(df['cycle_start_time'])
        return df
    else:
        print(f"Файл не найден: {csv_path}")
        return None

def save_database(df, csv_path):
    """Сохраняет любой DF по указанному пути."""
    if df is not None:
        df.to_csv(csv_path, index=False, sep=';', encoding='utf-8')

def remove_cycles_from_data(df_logs, df_registry, cycle_ids, col_cycle='cycle_id'):
    """
    Удаляет все записи, относящиеся к списку cycle_ids, из логов и реестра.
    cycle_ids: должен быть списком или массивом [id1, id2, ...]
    """
    # Гарантируем, что на входе список, даже если передали одно число
    if not isinstance(cycle_ids, (list, np.ndarray, pd.Series)):
        cycle_ids = [cycle_ids]

    # Удаляем из логов: оставляем те строки, id которых НЕ ВХОДИТ (~) в список cycle_ids
    df_logs = df_logs[~df_logs[col_cycle].isin(cycle_ids)].copy()
    
    # Удаляем из регистра
    df_registry = df_registry[~df_registry[col_cycle].isin(cycle_ids)].copy()

    
    return df_logs, df_registry

# =============================================================================
# БЛОК 2: УПРАВЛЕНИЕ МЕТКАМИ (Работает с Реестром/df_registry)
# =============================================================================

def update_label_by_time(df_registry, start_time_str, label, csv_path=None):
    """
    ВАЖНО: Теперь работает с df_registryistry (где 1 цикл = 1 строка).
    Обновляет тег 'case_tag' для конкретного впрыска.
    """
    if df_registry is None: return None

    try:
        target_time = pd.to_datetime(start_time_str)
    except Exception as e:
        print(f"Ошибка формата времени: {e}")
        return df_registry

    # Инициализация колонки тегов, если её нет в реестре
    if 'case_tag' not in df_registry.columns:
        df_registry['case_tag'] = "Standard"

    # Ищем строку в реестре
    mask = df_registry['cycle_start_time'] == target_time
    
    if mask.any():
        df_registry.loc[mask, 'case_tag'] = label
        if csv_path:
            save_database(df_registry, csv_path)
        print(f"Тег '{label}' присвоен циклу {start_time_str}")
    else:
        print(f"Цикл {start_time_str} не найден в реестре.")
    
    return df_registry

def get_summary(df, only_labeled=False):
    """
    Универсальная сводка для любого типа реестра (фактического или усредненного).
    """
    if df is None or df.empty:
        return None

    # 1. Список приоритетных колонок, которые мы хотим видеть в интерфейсе
    priority_cols = [
        "cycle_id",
        "cycle_start_time",
        "duration",
        "Dia_Shell",
        "L_Shell",
        "Weight_Group",
        "n_samples",  # Добавил сюда, чтобы в усредненном было видно кол-во циклов
        "case_tag"
    ]

    # 2. Отбираем только те колонки из списка, которые реально существуют в df
    existing_cols = [c for c in priority_cols if c in df.columns]
    
    # Создаем копию с нужными колонками
    summary = df[existing_cols].copy()

    # 3. Логика сортировки и обработки дубликатов
    if "cycle_start_time" in summary.columns:
        # Для ОБЫЧНОГО реестра
        summary = summary.drop_duplicates(subset=['cycle_start_time'])
        summary = summary.sort_values('cycle_start_time', ascending=False) # Свежие сверху
    else:
        # Для УСРЕДНЕННОГО реестра (сортируем по ID)
        summary = summary.sort_values('cycle_id')

    # 4. Фильтрация по меткам (только если есть колонка case_tag)
    if only_labeled and "case_tag" in summary.columns:
        summary = summary[summary['case_tag'] != "Standard"]

    return summary.reset_index(drop=True)

# =============================================================================
# БЛОК: ГЕНЕРАЦИЯ ЗАДАННОГО ЗНАЧЕНИЯ (УСТАВКИ) НА ОСНОВЕ МЕТАДАННЫХ И НАСТРОЕК
# =============================================================================

# def _generate_step_profile(t_array, presets, duration, phase_cfg):
#     """Рассчитывает уставку для массива времени согласно этапам с интерполяцией между циклами."""

#     target = np.zeros_like(t_array)
#     # Этап 1: Надувка бака (до провала)
#     mask_prep = (t_array >= phase_cfg['t_pre_charge']) & (t_array < phase_cfg['t_dip_start'])
#     target[mask_prep] = presets[0]
    
#     # Этап 2: Провал (между t_dip_start и t_start)
#     # Оставляем нулем (np.zeros_like уже сделал это)
    
#     # Этап 3: Активный цикл (интерполяция 10 точек)
#     mask_cycle = (t_array >= phase_cfg['t_start']) & (t_array <= duration)
#     x_phases = np.linspace(0, duration, 10)
#     target[mask_cycle] = np.interp(t_array[mask_cycle], x_phases, presets)

#     return target

def _generate_step_profile(t_array, presets, duration, phase_cfg):
    """Рассчитывает уставку в виде ступенчатого графика."""

    target = np.zeros_like(t_array)
    
    # Этап 1: Надувка бака (постоянное значение)
    mask_prep = (t_array >= phase_cfg['t_pre_charge']) & (t_array < phase_cfg['t_dip_start'])
    target[mask_prep] = presets[0]
    
    # Этап 2: Провал (нули) - пропускаем
    
    # Этап 3: Активный цикл (СТУПЕНЬКИ)
    mask_cycle = (t_array >= phase_cfg['t_start']) & (t_array <= duration)
    
    if np.any(mask_cycle):
        # 1. Создаем границы 10 временных интервалов (как и раньше)
        x_phases = np.linspace(0, duration, 10)
        
        # 2. Для каждого момента t находим, в какой интервал (индекс) он попал
        # np.digitize возвращает индекс интервала от 1 до 10
        # Вычитаем 1, чтобы получить индекс массива presets от 0 до 9
        local_t = t_array[mask_cycle] - phase_cfg['t_start']
        indices = np.digitize(local_t, x_phases) - 1
        
        # 3. Ограничиваем индекс сверху (на случай, если t ровно равно duration)
        indices = np.clip(indices, 0, len(presets) - 1)
        
        # 4. Присваиваем значения из пресетов по индексам
        target[mask_cycle] = np.array(presets)[indices]

    return target

def _generate_rounded_target(df_logs, CONFIG_ADD_SETTINGS):

    col_target = CONFIG_ADD_SETTINGS['columns']['col_target_tank']
    col_target_rounded = CONFIG_ADD_SETTINGS['columns']['col_target_tank_rounded']

    df_logs[col_target_rounded] = (np.floor(df_logs[col_target] * 0.6))

    return df_logs

def apply_target_logic(df_logs, df_registry, df_settings, CONFIG_ADD_SETTINGS):
    # Инициализация колонок
    cols = CONFIG_ADD_SETTINGS['columns']
    df_logs[cols['col_target_tank']] = 0.0
    df_logs[cols['col_target_pipe']] = 0.0

    unique_cycles = df_logs[cols['col_cycle']].unique()

    for cid in unique_cycles:
        # 1. Получаем метаданные (Dia, L)
        reg_row = df_registry[df_registry['cycle_id'] == cid]
        if reg_row.empty: 
            print(f"[DEBUG]: Цикл {cid} не найден в реестре")
            continue
        
        # 2. Получаем настройки пресетов
        dia, length = reg_row.iloc[0][['Dia_Shell', 'L_Shell']]
        settings = df_settings[(df_settings['Dia_Shell'] == dia) & 
                               (df_settings['L_Shell'] == length)]
        if settings.empty: 
            print(f"[DEBUG]: Настройки для цикла {cid} не найдены")
            continue
        set_row = settings.iloc[0]

        # 3. Подготавливаем данные для ядра
        pts_tank = [set_row[f'p_tank_preset_{i}'] for i in range(1, 11)]
        pts_pipe = [set_row[f'p_pipe_preset_{i}'] for i in range(1, 11)]
        duration = float(set_row['Time_cycle']) / 1000.0

        # Выделяем временную сетку конкретного цикла
        mask = df_logs[cols['col_cycle']] == cid
        t_values = df_logs.loc[mask, cols['col_time']].values

        # 4. Вызываем ядро для бака и для трубы
        df_logs.loc[mask, cols['col_target_tank']] = _generate_step_profile(t_values, pts_tank, duration, CONFIG_ADD_SETTINGS['phases_tank'])
        df_logs.loc[mask, cols['col_target_pipe']] = _generate_step_profile(t_values, pts_pipe, duration, CONFIG_ADD_SETTINGS['phases_pipe'])

        df_logs = _generate_rounded_target(df_logs, CONFIG_ADD_SETTINGS)

    return df_logs

# =============================================================================
# БЛОК 3: РАБОТА С КОНФИГУРАЦИЯМИ (JSON/TXT)
# =============================================================================

def save_config(config_dict, file_path):
    """
    Сохраняет настройки. file_path берется из config.py
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        print(f"Конфигурация сохранена: {file_path.name}")
    except Exception as e:
        print(f"Ошибка сохранения JSON: {e}")

def load_config(file_path):
    """
    Загружает настройки и возвращает (полный_конфиг, плоские_параметры).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Файл настроек не найден: {file_path}")
        return None, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_params = json.load(f)
        
        # Получаем плоский словарь значений для физической модели
        flat_params = get_flat_params(config_params)
        return config_params, flat_params
    except Exception as e:
        print(f"Ошибка чтения JSON: {e}")
        return None, None

def get_flat_params(config_dict):
    """Вытаскивает только 'value' из структуры конфига."""
    # Обработка случая, если конфиг уже плоский или имеет структуру {key: {value: X}}
    flat_params = {}
    for k, v in config_dict.items():
        if isinstance(v, dict) and 'value' in v:
            flat_params[k] = v['value']
        else:
            flat_params[k] = v
    return flat_params

# =============================================================================
# БЛОК 4: РАБОТА С УСРЕДНЕНИЕМ ПРОФИЛЕЙ (Группировка по метаданным)
# =============================================================================

def _generate_cycle_id(row, classification_tasks):
    """Приватный помощник: формирует строку ID (бизнес-логика именования)"""
    parts = [f"{t.get('prefix', '')}{row[t['target_col']]}" for t in classification_tasks]
    return "AVG_" + "_".join(parts)

def _build_agg_rules(columns, detailed_cols):
    """Приватный помощник: формирует словарь правил агрегации"""
    return {
        col: (['mean', 'median', 'std'] if col in detailed_cols else 'mean') 
        for col in columns
    }

def get_averaged_profiles(df_logs, df_registry, classification_tasks, detailed_cols):
    """
    ГЛАВНАЯ ФУНКЦИЯ: Оркестратор процесса усреднения.
    """
    if df_logs is None or df_registry is None: return None, None

    group_cols = [t["target_col"] for t in classification_tasks]

    # --- 1. Агрегация Реестра ---
    reg_numeric = df_registry.select_dtypes(include=[np.number]).columns.difference(group_cols)
    df_avg_reg = df_registry.groupby(group_cols)[reg_numeric].mean().reset_index()
    
    # Добавляем количество образцов и ID
    df_avg_reg['n_samples'] = df_registry.groupby(group_cols)['cycle_id'].count().values
    df_avg_reg['cycle_id'] = df_avg_reg.apply(_generate_cycle_id, axis=1, args=(classification_tasks,))

    # --- 2. Агрегация Логов ---
    meta = df_registry[['cycle_id'] + group_cols]
    temp_logs = pd.merge(df_logs, meta, on='cycle_id')

    log_numeric = temp_logs.select_dtypes(include=[np.number]).columns.difference(group_cols + ['t_relative'])
    agg_rules = _build_agg_rules(log_numeric, detailed_cols)

    df_agg_logs = temp_logs.groupby(group_cols + ['t_relative']).agg(agg_rules).reset_index()

    # --- 3. Форматирование имен ---
    df_agg_logs.columns = [f"{c[0]}_{c[1]}" if c[1] not in ('mean', '') else c[0] 
                          for c in df_agg_logs.columns.to_flat_index()]
    df_agg_logs['cycle_id'] = df_agg_logs.apply(_generate_cycle_id, axis=1, args=(classification_tasks,))

    return df_agg_logs.sort_values(['cycle_id', 't_relative']), df_avg_reg

# =============================================================================
# БЛОК 4: ИНТЕРФЕЙСНЫЕ НАСТРОЙКИ (UI)
# =============================================================================

def load_ui_config(file_path):
    """Загружает настройки UI (цвета, лимиты осей и т.д.)"""
    file_path = Path(file_path)
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}
    return {}

def save_ui_config(ui_dict, file_path):
    file_path = Path(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ui_dict, f, ensure_ascii=False, indent=4)