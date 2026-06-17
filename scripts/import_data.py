# scripts/import_data.py
import pandas as pd
import glob
import os
from datetime import timedelta

# =============================================================================
# БЛОК 1: ИМПОРТ ДАННЫХ ПРОДУВОК
# =============================================================================

def _load_raw_data(file_path, time_col_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        columns = [c.strip().strip('[]') for c in all_lines[1].strip().split(';')]
    
    df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=7, names=columns, low_memory=False)
    raw_time_col = [c for c in df.columns if 'time' in c.lower()][0]
    df[time_col_name] = pd.to_datetime(df[raw_time_col], dayfirst=True)
    return df.drop(columns=[raw_time_col])

def _extract_cycles(df, filename, settings, ts_cols, reg_cols):
    """
    settings: словарь с триггерами
    ts_cols, reg_cols: списки колонок
    """
    detected_ts, detected_reg = [], []
    t_open = settings['trigger_open']
    t_close = settings['trigger_close']
    time_col = settings['time_col']

    starts = df.index[(df[t_open].shift(1) == 0) & (df[t_open] == 1)].tolist()
    
    for start_idx in starts:
        t_start = df.loc[start_idx, time_col]
        future_data = df.loc[start_idx : start_idx + 2000]
        closes = future_data.index[(future_data[t_close].shift(1) == 0) & (future_data[t_close] == 1)].tolist()
        
        if closes:
            duration = (df.loc[closes[0], time_col] - t_start).total_seconds()
            if 0 < duration <= settings["MAX_ALLOWED_DURATION"]:
                c_id = f"{filename}_{t_start.strftime('%Y%m%d_%H%M%S')}"
                
                # Реестр
                reg_entry = df.loc[start_idx, reg_cols].to_dict()
                
                # --- ИЗМЕНЕНИЕ: Сдвиг на 20 строк назад (4 секунды при шаге 0.2с) ---
                target_col = "LD31W.VALVE 1007 - клапан бака, бар"
                if target_col in reg_entry:
                    # Ограничиваем снизу нулем, чтобы не выйти за пределы начала датафрейма
                    idx_4s = max(0, start_idx - 20)
                    reg_entry[target_col] = df.loc[idx_4s, target_col]
                # ------------------------------------------------------------------
                
                reg_entry.update({"cycle_id": c_id, "cycle_start_time": t_start, "duration": duration})
                detected_reg.append(reg_entry)
                
                # Логи
                mask = (df[time_col] >= t_start - timedelta(seconds=settings['t_before'])) & \
                       (df[time_col] <= t_start + timedelta(seconds=settings['t_after']))
                
                cycle_slice = df.loc[mask].copy()
                cycle_slice['t_relative'] = (cycle_slice[time_col] - t_start).dt.total_seconds()
                cycle_slice['cycle_id'] = c_id
                
                cols_to_keep = ["cycle_id", "t_relative"] + [c for c in ts_cols if c in cycle_slice.columns]
                detected_ts.append(cycle_slice[cols_to_keep])
                
    return detected_ts, detected_reg

# def _extract_cycles(df, filename, settings, ts_cols, reg_cols):
#     """
#     settings: словарь с триггерами
#     ts_cols, reg_cols: списки колонок
#     """
#     detected_ts, detected_reg = [], []
#     t_open = settings['trigger_open']
#     t_close = settings['trigger_close']
#     time_col = settings['time_col']

#     starts = df.index[(df[t_open].shift(1) == 0) & (df[t_open] == 1)].tolist()
    
#     for start_idx in starts:
#         t_start = df.loc[start_idx, time_col]
#         future_data = df.loc[start_idx : start_idx + 2000]
#         closes = future_data.index[(future_data[t_close].shift(1) == 0) & (future_data[t_close] == 1)].tolist()
        
#         if closes:
#             duration = (df.loc[closes[0], time_col] - t_start).total_seconds()
#             if 0 < duration <= settings["MAX_ALLOWED_DURATION"]:
#                 c_id = f"{filename}_{t_start.strftime('%Y%m%d_%H%M%S')}"
                
#                 # Реестр
#                 reg_entry = df.loc[start_idx, reg_cols].to_dict()
#                 reg_entry.update({"cycle_id": c_id, "cycle_start_time": t_start, "duration": duration})
#                 detected_reg.append(reg_entry)
                
#                 # Логи
#                 mask = (df[time_col] >= t_start - timedelta(seconds=settings['t_before'])) & \
#                        (df[time_col] <= t_start + timedelta(seconds=settings['t_after']))
                
#                 cycle_slice = df.loc[mask].copy()
#                 cycle_slice['t_relative'] = (cycle_slice[time_col] - t_start).dt.total_seconds()
#                 cycle_slice['cycle_id'] = c_id
                
#                 cols_to_keep = ["cycle_id", "t_relative"] + [c for c in ts_cols if c in cycle_slice.columns]
#                 detected_ts.append(cycle_slice[cols_to_keep])
                
#     return detected_ts, detected_reg

def run_extraction(raw_dir, logs_file, reg_file, settings, col_config):
    """
    Выполняет экстракцию данных.
    Возвращает: (df_logs, df_registryistry) или (None, None) при отсутствии данных.
    """
    all_ts, all_reg = [], []
    files = glob.glob(os.path.join(str(raw_dir), "*.txt"))
    
    for f_path in files:
        try:
            raw_df = _load_raw_data(f_path, settings['time_col']) 
            ts, reg = _extract_cycles(raw_df, os.path.basename(f_path), 
                                      settings, col_config['ts_cols'], col_config['reg_cols'])
            all_ts.extend(ts)
            all_reg.extend(reg)
        except Exception as e:
            print(f"Ошибка в файле {f_path}: {e}")

    if all_ts:
        # 1. Формируем итоговые объекты в памяти
        df_logs = pd.concat(all_ts, ignore_index=True)
        df_registry = pd.DataFrame(all_reg)
        
        #Возвращаем масштабированные данные в исходный вид с иба
        df_logs["LD31W.VALVE 1007 - клапан бака, бар"]*=10
        df_logs["LD31W.VALVE 1008 - клапан трубы, бар"]*=10
        df_logs["LD31W.VALVE 1019 - клапан спирали, бар"]*=10
        
        # 2. Сохраняем их на диск (для будущих сессий)
        df_logs.to_csv(logs_file, index=False, sep=';')
        df_registry.to_csv(reg_file, index=False, sep=';')
        
        print(f"Экстракция завершена. Обработано {len(df_registry)} циклов.")
        
        # 3. Возвращаем их пользователю для немедленной работы
        return df_logs, df_registry
    
    return None, None

# =============================================================================
# БЛОК 1: ИМПОРТ ДАННЫХ НАСТРОЕК
# =============================================================================

def classify_registry_generic(df_registryistry, tasks_config):
    if df_registryistry is None or df_registryistry.empty:
        return df_registryistry

    for task in tasks_config:
        src, tgt = task["source_col"], task["target_col"]
        if src not in df_registryistry.columns:
            continue
            
        bins = sorted(task["bins"])

        def determine_category(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return "Unknown"

            # Универсальный поиск: вернет значение бина, если v попал в него,
            # либо метку "_plus" для всех значений выше самого большого порога.
            res = next((b for b in bins if v <= b), f"{bins[-1]}_plus")
            return f"{res}"

        df_registryistry[tgt] = df_registryistry[src].apply(determine_category)
        print(f"[Classify] Создана колонка '{tgt}' на основе '{src}'")

    return df_registryistry

