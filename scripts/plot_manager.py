# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import mplcursors
HAS_CURSORS = True
from pathlib import Path


# =============================================================================
# 1. ВНУТРЕННЯЯ ЛОГИКА ПОИСКА
# =============================================================================

def _get_cycle_ids(df_registry, identifiers):
    ids = identifiers if isinstance(identifiers, list) else [identifiers]
    cols = [c for c in ['cycle_id', 'case_tag', 'cycle_start_time'] if c in df_registry.columns]
    
    # Ищем по всем доступным колонкам
    mask = df_registry['cycle_id'].isin(ids)
    if 'case_tag' in cols: mask |= df_registry['case_tag'].isin(ids)
    if 'cycle_start_time' in cols: mask |= df_registry['cycle_start_time'].isin(pd.to_datetime(ids, errors='coerce'))
    
    return df_registry[mask][cols]

# =============================================================================
# 2. ОСНОВНАЯ ФУНКЦИЯ ОТРИСОВКИ
# =============================================================================
def plot_cycles(df_logs, df_registry, identifiers, sensors, ui_config=None):
    target_cycles = _get_cycle_ids(df_registry, identifiers)
    if target_cycles.empty: return

    # Настройки по умолчанию
    cfg = ui_config or {}
    colors = cfg.get("colors", plt.rcParams['axes.prop_cycle'].by_key()['color'])
    styles = cfg.get("line_styles", ['-', '--', ':', '-.'])

    plt.close('all')
    fig, ax = plt.subplots(figsize=cfg.get("figsize", (10, 6)))
    lines_for_cursor = []

    for i, (_, row) in enumerate(target_cycles.iterrows()):
        # --- ГЕНЕРАЦИЯ УНИВЕРСАЛЬНОЙ МЕТКИ ---
        # Если есть тег — берем его, если нет — берем cycle_id
        tag = row.get('case_tag', row['cycle_id'])
        # Если есть время — форматируем, если нет — пустая строка
        t_val = row.get('cycle_start_time')
        t_str = f" ({pd.to_datetime(t_val).strftime('%H:%M:%S')})" if pd.notnull(t_val) else ""
        label_base = f"{tag}{t_str}"
        
        # Данные
        c_df = df_logs[df_logs['cycle_id'] == row['cycle_id']].sort_values('t_relative')
        if c_df.empty: continue
        
        color = colors[i % len(colors)]

        for j, sensor in enumerate(sensors):
            if sensor not in c_df.columns: continue
            
            line, = ax.plot(c_df['t_relative'], c_df[sensor], 
                            color=color, linestyle=styles[j % len(styles)], 
                            label=f"{label_base} | {sensor}")
            lines_for_cursor.append(line)

    # Оформление
    ax.set_title("Сравнение продувок")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Легенда датчиков (всегда слева)
    s_handles = [Line2D([0], [0], color='gray', linestyle=styles[k % len(styles)], label=s) for k, s in enumerate(sensors)]
    ax.add_artist(ax.legend(handles=s_handles, title="Датчики", loc='upper left'))

    # Легенда случаев (всегда справа)
    # Здесь мы пересобираем те же метки label_base для легенды
    c_handles = []
    for i, (_, row) in enumerate(target_cycles.iterrows()):
        tag = row.get('case_tag', row['cycle_id'])
        t_val = row.get('cycle_start_time')
        t_str = f" ({pd.to_datetime(t_val).strftime('%H:%M:%S')})" if pd.notnull(t_val) else ""
        c_handles.append(Line2D([0], [0], color=colors[i % len(colors)], label=f"{tag}{t_str}"))
    ax.legend(handles=c_handles, title="Продувки", loc='upper right')

    plt.tight_layout()
    # plt.show()

def plot_group_diagnostics(df_logs, df_registry, df_avg_logs, avg_id, sensor, classification_tasks, graf_dir = None, ymax=None, ymin=None):
    """
    Универсальная диагностика: фильтрует данные по всем колонкам из tasks.
    """
    # 1. Получаем список колонок группировки из конфига
    group_cols = [t['target_col'] for t in classification_tasks]

    # 2. Выясняем параметры конкретной группы из усредненного датафрейма
    target_row = df_avg_logs[df_avg_logs['cycle_id'] == avg_id].iloc[0]

    # 3. Находим в реестре все реальные cycle_id, которые совпадают по ВСЕМ признакам
    # Создаем динамическую маску
    mask = pd.Series(True, index=df_registry.index)
    for col in group_cols:
        # mask &= (df_registry[col] == target_row[col])
        mask &= (df_registry[col].astype(float) == float(target_row[col]))
    real_cycle_ids = df_registry[mask]['cycle_id'].unique()

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.figure(figsize=(12, 7))
    
    # "Серый туман"
    subset_logs = df_logs[df_logs['cycle_id'].isin(real_cycle_ids)]
    for c_id in real_cycle_ids:
        one_cycle = subset_logs[subset_logs['cycle_id'] == c_id]
        plt.plot(one_cycle['t_relative'], one_cycle[sensor], 
                 color='gray', alpha=0.1, linewidth=0.8)

    # Среднее
    avg_data = df_avg_logs[df_avg_logs['cycle_id'] == avg_id]
    plt.plot(avg_data['t_relative'], avg_data[sensor], 
             color='red', linewidth=2.5, label=f'Среднее (n={len(real_cycle_ids)})')

    # Медиана
    median_col = f"{sensor}_median"
    if median_col in avg_data.columns:
        plt.plot(avg_data['t_relative'], avg_data[median_col], 
                 color='blue', linestyle='--', linewidth=1.5, label='Медиана')

    plt.title(f"Группа: {avg_id}\nДатчик: {sensor}")
    plt.xlabel("Время, с")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.ylim(ymin, ymax)

    # plt.savefig(graf_dir / "график.png")

    if graf_dir:
        # Формируем имя БЕЗ f-строк внутри replace (чтобы не злить старый Python)
        s_safe = sensor.replace('\\', '_').replace('/', '_')
        final_path = Path(graf_dir) / f"{avg_id}_{s_safe}.png"
        # Сохраняем
        plt.savefig(final_path)

# def plot_group_diagnostics(df_logs, df_registry, df_avg_logs, avg_id, sensor):
#     """
#     Автоматическая диагностика: находит исходные циклы по метаданным из df_avg_logs.
    
#     1. df_logs: Исходная база всех продувок.
#     2. df_registry: Регистр (с колонками Dia_Shell и L_Shell).
#     3. df_avg_logs: Осредненная база (с колонками Dia_Shell и L_Shell).
#     4. avg_id: ID усредненного профиля (напр. "AVG_182_5000").
#     5. sensor: Имя датчика.
#     """
#     # 1. Выясняем параметры группы из усредненного датафрейма
#     # Берем первую попавшуюся строку для этого avg_id
#     avg_meta = df_avg_logs[df_avg_logs['cycle_id'] == avg_id].iloc[0]
#     d_shell = avg_meta['Dia_Shell']
#     l_shell = avg_meta['L_Shell']

#     # 2. Находим в регистре все реальные cycle_id, которые подпадают под этот типоразмер
#     real_cycle_ids = df_registry[
#         (df_registry['Dia_Shell'] == d_shell) & 
#         (df_registry['L_Shell'] == l_shell)
#     ]['cycle_id'].unique()

#     plt.figure(figsize=(12, 7))
    
#     # 3. Рисуем "Серый туман" из исходных данных
#     # Фильтруем логи один раз для скорости
#     subset_logs = df_logs[df_logs['cycle_id'].isin(real_cycle_ids)]
    
#     for c_id in real_cycle_ids:
#         one_cycle = subset_logs[subset_logs['cycle_id'] == c_id]
#         plt.plot(one_cycle['t_relative'], one_cycle[sensor], 
#                  color='gray', alpha=0.1, linewidth=0.8)

#     # 4. Рисуем Среднее (из df_avg_logs)
#     avg_data = df_avg_logs[df_avg_logs['cycle_id'] == avg_id]
#     plt.plot(avg_data['t_relative'], avg_data[sensor], 
#              color='red', linewidth=2.5, label=f'Среднее (n={len(real_cycle_ids)})')

#     # 5. Рисуем Медиану (если она была рассчитана)
#     median_col = f"{sensor}_median"
#     if median_col in avg_data.columns:
#         plt.plot(avg_data['t_relative'], avg_data[median_col], 
#                  color='blue', linestyle='--', linewidth=2, label='Медиана')

#     plt.title(f"Диагностика: {avg_id} (Диаметр: {d_shell}, Длина: {l_shell})")
#     plt.xlabel("Время, с")
#     plt.ylabel(sensor)
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.5)
