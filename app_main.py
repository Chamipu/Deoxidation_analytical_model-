# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt

# Импорт конфигурации
import config as cnfg

# Импорт рабочих модулей
from scripts import import_data as idt
from scripts import data_manager as dm
from scripts import plot_manager as pm
# from scripts import pressure_predictor as ppm
from scripts import pressure_reference_predictor as ppm

# Импорт компонентов интерфейса
from gui import gui_components as gui_cmp
from gui import gui_plotter as gui_plt

class PneumaticTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumatic System Tuner Pro v4.0")
        
        # Протокол закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # --- КОНФИГУРАЦИЯ ПУТЕЙ (из config.py) ---
        self.LOGS_FILE = cnfg.LOGS_AVG_FILE
        self.REGISTRY_FILE = cnfg.REGISTRY_AVG_FILE
        self.DEFAULT_MODEL_PARAMS_FILE = cnfg.DEFAULT_MODEL_PARAMS_FILE
        self.GUI_CONFIG_FILE = cnfg.GUI_CONFIG_FILE # Путь к JSON настройкам UI
        
        # Список датчиков из конфига (динамические данные)
        self.SENSORS_LIST = cnfg.SENSORS_LIST

        # --- ЗАГРУЗКА ДАННЫХ ---
        # 1. Загрузка раздельных баз (Логи и Реестр)
        self.df_logs = dm.load_database(self.LOGS_FILE)
        self.df_registry = dm.load_database(self.REGISTRY_FILE)
        
        if self.df_logs is None or self.df_registry is None:
            messagebox.showerror("Ошибка", "База данных (Логи или Реестр) не найдена!")
            self.root.destroy(); return

        # 2. Загрузка настроек интерфейса (цвета, стили)
        self.gui_config = dm.load_ui_config(self.GUI_CONFIG_FILE)
        self._init_ui_config_defaults()

        # --- ПОСТРОЕНИЕ ИНТЕРФЕЙСА ---
        self.setup_layout()
        
        # --- ЗАГРУЗКА ФИЗИКИ ПО УМОЛЧАНИЮ ---
        self.reset_physics()

    def _init_ui_config_defaults(self):
        """Заполнение настроек UI значениями по умолчанию."""
        if "markers" not in self.gui_config: 
            self.gui_config["markers"] = {
                "OPN": {"visible": True, "color": "green", "ls": "-"},
                "CLS": {"visible": True, "color": "red", "ls": "-"}
            }
        if "sensors" not in self.gui_config: 
            self.gui_config["sensors"] = {}
        
        # Масштабы осей по умолчанию
        if "axis_limits" not in self.gui_config:
            self.gui_config["axis_limits"] = {
                "x_min": 0.0, "x_max": 10.0, "x_fix": False,
                "y1_min": 0.0, "y1_max": 6.0, "y1_fix": False,
                "y2_min": 0.0, "y2_max": 50.0, "y2_fix": False
            }
        
        for s in self.SENSORS_LIST:
            if s not in self.gui_config["sensors"]:
                is_vis = True if "theory" in s or "PT1009" in s else False
                self.gui_config["sensors"][s] = {"visible": is_vis, "color": "blue", "ls": "-"}

    def setup_layout(self):
        """Конфигурация интерфейса строго по списку требований пользователя."""
        self.main_f = ttk.Frame(self.root)
        self.main_f.pack(fill="both", expand=True, padx=10, pady=10)

        # --- ЛЕВАЯ ПАНЕЛЬ ---
        self.left_f = ttk.Frame(self.main_f)
        self.left_f.pack(side="left", fill="y", padx=5)

        # 1. КНОПКИ (Загрузить / Сохранить / Сбросить)
        btn_f = ttk.Frame(self.left_f)
        btn_f.pack(fill="x", pady=5)
        ttk.Button(btn_f, text="ОТКРЫТЬ ФИЗИКУ", command=self.load_physics_from_file).pack(side="left", expand=True)
        ttk.Button(btn_f, text="СОХРАНИТЬ", command=self.save_physics).pack(side="left", expand=True)
        ttk.Button(btn_f, text="СБРОСИТЬ", command=self.reset_physics).pack(side="left", expand=True)

        # Выбор цикла (сразу под кнопками)
        summary = dm.get_summary(self.df_registry)
        labels = [f"{r['cycle_start_time']} [{r['case_tag']}]" for _, r in summary.iterrows()]
        self.cycle_var = tk.StringVar(value=labels[-1] if labels else "")
        self.combo = ttk.Combobox(self.left_f, textvariable=self.cycle_var, values=labels, width=55, state="readonly")
        self.combo.pack(pady=5)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.update_view())

        # 2. МАРКЕРЫ (Фронты порошка)
        ttk.Label(self.left_f, text="МАРКЕРЫ (ФРОНТЫ ПОРОШКА):", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10, 0))
        self.marker_ctrls = {
            "OPN": gui_cmp.MarkerControl(self.left_f, "Открытие (OPN)", self.gui_config["markers"]["OPN"], self.update_view),
            "CLS": gui_cmp.MarkerControl(self.left_f, "Закрытие (CLS)", self.gui_config["markers"]["CLS"], self.update_view)
        }

        # 3. ОТОБРАЖЕНИЕ ДАТЧИКОВ (Жирный заголовок)
        ttk.Label(self.left_f, text="ОТОБРАЖЕНИЕ ДАТЧИКОВ:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10, 0))
        self.sens_f = ttk.Frame(self.left_f)
        self.sens_f.pack(fill="x", pady=5)
        self.sensor_controls = []
        # Список датчиков из конфига + теория
        display_sensors = self.SENSORS_LIST + ["p_tank_theory"]
        for s in display_sensors:
            cfg = self.gui_config["sensors"].get(s, {"visible": False, "color": "blue", "ls": "-"})
            sc = gui_cmp.SensorControl(self.sens_f, s, cfg, self.update_view)
            self.sensor_controls.append(sc)

        # 4. ПАРАМЕТРЫ МОДЕЛИ
        ttk.Label(self.left_f, text="ПАРАМЕТРЫ МОДЕЛИ (наведи на имя):", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10, 0))
        self.param_canvas = tk.Canvas(self.left_f, width=520, height=250) # Уменьшена высота для осей
        self.param_frame = ttk.Frame(self.param_canvas)
        sb = ttk.Scrollbar(self.left_f, orient="vertical", command=self.param_canvas.yview)
        self.param_canvas.create_window((0,0), window=self.param_frame, anchor="nw")
        self.param_canvas.configure(yscrollcommand=sb.set)
        self.param_canvas.pack(side="top", fill="both", expand=True)
        sb.pack(side="right", fill="y", before=self.param_canvas)
        self.param_frame.bind("<Configure>", lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all")))

        # 5. МАСШТАБЫ ОСЕЙ (В самом низу)
        ttk.Label(self.left_f, text="МАСШТАБЫ ОСЕЙ (Мин / Макс / Фикс):", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(15, 5))
        self.axis_ui = {}
        for ax_id, label in [("x", "Время (с)"), ("y1", "Давл (бар)"), ("y2", "Вес (кг) ")]:
            f = ttk.Frame(self.left_f)
            f.pack(fill="x", pady=1)
            ttk.Label(f, text=label, width=15).pack(side="left")
            
            lims = self.gui_config.get("axis_limits", {})
            v_min = tk.StringVar(value=str(lims.get(f"{ax_id}_min", 0)))
            v_max = tk.StringVar(value=str(lims.get(f"{ax_id}_max", 10)))
            v_fix = tk.BooleanVar(value=lims.get(f"{ax_id}_fix", False))
            
            e_min = ttk.Entry(f, textvariable=v_min, width=8)
            e_min.pack(side="left", padx=2)
            e_max = ttk.Entry(f, textvariable=v_max, width=8)
            e_max.pack(side="left", padx=2)
            ttk.Checkbutton(f, variable=v_fix, command=self.update_view).pack(side="left", padx=5)
            
            self.axis_ui[ax_id] = {"min": v_min, "max": v_max, "fix": v_fix}
            e_min.bind("<Return>", lambda e: self.update_view())
            e_max.bind("<Return>", lambda e: self.update_view())

        # --- ПРАВАЯ ПАНЕЛЬ ---
        self.plot_frame = ttk.Frame(self.main_f)
        self.plot_frame.pack(side="right", fill="both", expand=True)
        self.plotter = gui_plt.Plotter(self.plot_frame)

    # --- ЛОГИКА ПАРАМЕТРОВ ---

    def reset_physics(self):
        self._load_physics_from_path(self.DEFAULT_MODEL_PARAMS_FILE)

    def load_physics_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("JSON", "*.json")])
        if path: self._load_physics_from_path(path)

    def _load_physics_from_path(self, path):
        res = dm.load_config(path)
        if res:
            self.physics_full, _ = res
            for w in self.param_frame.winfo_children(): w.destroy()
            self.sliders = [gui_cmp.ParamSlider(self.param_frame, k, v, self.update_view) for k, v in self.physics_full.items()]
            self.update_view()

    def save_physics(self):
        for s in self.sliders:
            self.physics_full[s.key]["value"] = s.v_val.get()
            self.physics_full[s.key]["min"] = float(s.v_min.get())
            self.physics_full[s.key]["max"] = float(s.v_max.get())
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path: dm.save_config(self.physics_full, path)

    # --- ОБНОВЛЕНИЕ ---

    def update_view(self, *args):
        """Обновление модели и графика при любом изменении параметров."""
        
        # 1. Синхронизация настроек ОСЕЙ из полей ввода в конфиг
        if "axis_limits" not in self.gui_config:
            self.gui_config["axis_limits"] = {}
            
        for ax_id, vars in self.axis_ui.items():
            try:
                self.gui_config["axis_limits"][f"{ax_id}_min"] = float(vars["min"].get())
                self.gui_config["axis_limits"][f"{ax_id}_max"] = float(vars["max"].get())
                self.gui_config["axis_limits"][f"{ax_id}_fix"] = bool(vars["fix"].get())
            except ValueError:
                # Если в полях ввода осей мусор, просто не обновляем эти значения
                pass

        # 2. Синхронизация настроек МАРКЕРОВ (линии OPN/CLS)
        self.gui_config["markers"]["OPN"] = self.marker_ctrls["OPN"].get_settings()
        self.gui_config["markers"]["CLS"] = self.marker_ctrls["CLS"].get_settings()

        # 3. Синхронизация настроек ДАТЧИКОВ (видимость, цвет, тип линии)
        for sc in self.sensor_controls:
            self.gui_config["sensors"][sc.sensor_name] = sc.get_settings()

        # 4. Сбор ФИЗИЧЕСКИХ параметров со слайдеров/полей ввода
        p_flat = {s.key: s.v_val.get() for s in self.sliders}
        
        # 5. Определение выбранного цикла
        c_label = self.cycle_var.get()
        if not c_label:
            return
        # Извлекаем дату/время (то, что до скобок с тегом)
        c_time_str = c_label.split(" [")[0] if " [" in c_label else c_label

        # 6. РАСЧЕТ МОДЕЛИ (Прямая задача)
        # Функция возвращает обновленные логи, где в колонке 'p_tank_theory' лежат расчетные данные
        # self.df_logs, self.df_registry = ppm.get_cycle_model(
        #     self.df_logs, 
        #     self.df_registry, 
        #     c_time_str, 
        #     "LD31W.VALVE 1007 - клапан бака, бар", # Управляющий сигнал
        #     p_flat,
        #     cnfg.EXTRACTION_SETTINGS
        # )

        self.df_logs, self.df_registry = ppm.get_cycle_model(
            self.df_logs, 
            self.df_registry, 
            c_time_str, 
            "LD31W.VALVE 1007 - клапан бака, бар", # Управляющий сигнал
            p_flat,
            cnfg.EXTRACTION_SETTINGS
        )
        
        # 7. Подготовка данных для отрисовки (срез конкретного цикла)
        mask_reg = (self.df_registry['cycle_start_time'] == pd.to_datetime(c_time_str))
        if mask_reg.any():
            c_id = self.df_registry.loc[mask_reg, 'cycle_id'].values[0]
            df_plot = self.df_logs[self.df_logs['cycle_id'] == c_id].sort_values('t_relative')
            
            # 8. Вызов отрисовки в GUI_PLOTTER
            self.plotter.draw(df_plot, self.gui_config, c_label)

        # 9. Сохранение всех настроек интерфейса (включая новые оси) в файл
        dm.save_ui_config(self.gui_config, self.GUI_CONFIG_FILE)

    def on_exit(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = PneumaticTunerApp(root)
    root.mainloop()