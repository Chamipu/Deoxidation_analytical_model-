# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import config as cnfg
import config_paths as cnfg_p
from gui.widgets import SensorControl, MarkerControl, PhysicsModelBlock
from gui.plotter import Plotter
from scripts import data_manager as dm
from scripts.independent_linear_predictor import pressure_predictor_lite as prm

class SimulationTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self._setup_layout()
        self.update_view()  # Принудительное обновление графика при запуске приложения


    def _setup_layout(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- ЛЕВАЯ ПАНЕЛЬ (САЙДБАР) ---
        self.left_f = ttk.Frame(paned, padding=5)
        paned.add(self.left_f, weight=1)

        # 1. Выбор цикла со стрелочками переключения
        cycle_frame = ttk.Frame(self.left_f)
        cycle_frame.pack(fill="x", pady=5)

        labels = self.state.df_registry['cycle_id'].dropna().unique().tolist()
        self.cycle_var = tk.StringVar(value=labels[-1] if labels else "")

        btn_prev = ttk.Button(cycle_frame, text="◀", width=3, command=self._prev_cycle)
        btn_prev.pack(side="left", padx=(0, 2))

        cb = ttk.Combobox(cycle_frame, textvariable=self.cycle_var, values=labels, state="readonly")
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.update_view())

        btn_next = ttk.Button(cycle_frame, text="▶", width=3, command=self._next_cycle)
        btn_next.pack(side="left", padx=(2, 0))

        # 2. Область прокрутки для параметров
        canvas = tk.Canvas(self.left_f)
        sb = ttk.Scrollbar(self.left_f, orient="vertical", command=canvas.yview)
        self.scroll_f = ttk.Frame(canvas)
        canvas.create_window((0,0), window=self.scroll_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        self.scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # --- МОДЕЛИ (УНИФИЦИРОВАННЫЕ БЛОКИ) ---
        self.tank_model = PhysicsModelBlock(self.scroll_f, "МОДЕЛЬ БАКА (P1)", 
                                           cnfg_p.DEFAULT_MODEL_PARAMS_TANK_FILE, self.update_view)
        self.tank_model.pack(fill="x", pady=5)

        self.pipe_model = PhysicsModelBlock(self.scroll_f, "МОДЕЛЬ ТРУБЫ (P2)", 
                                           cnfg_p.DEFAULT_MODEL_PARAMS_PIPE_FILE, self.update_view)
        self.pipe_model.pack(fill="x", pady=5)

        # --- ОСТАЛЬНЫЕ НАСТРОЙКИ (МАРКЕРЫ, ДАТЧИКИ, ОСИ) ---
        # Маркеры
        ttk.Label(self.scroll_f, text="МАРКЕРЫ:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        self.marker_ctrls = {
            "OPN": MarkerControl(self.scroll_f, "Открытие (OPN)", self.state.gui_config["markers"]["OPN"], self.update_view),
            "CLS": MarkerControl(self.scroll_f, "Закрытие (CLS)", self.state.gui_config["markers"]["CLS"], self.update_view)
        }

        # Датчики
        ttk.Label(self.scroll_f, text="ДАТЧИКИ:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        self.sensor_ctrls = [SensorControl(self.scroll_f, s, self.state.gui_config["sensors"].get(s, {}), self.update_view) 
                            for s in cnfg.SENSORS_LIST]

        # Оси (перемещены в scroll_f под блок датчиков)
        ttk.Label(self.scroll_f, text="ОСИ (Мин/Макс/Фикс):", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(15,5))
        self.axis_ui = {}
        for ax_id, label in [("x", "Время"), ("y1", "Давление"), ("y2", "Вес")]:
            f = ttk.Frame(self.scroll_f)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=label, width=10).pack(side="left")
            lims = self.state.gui_config["axis_limits"]
            
            v_min = tk.StringVar(value=str(lims[f"{ax_id}_min"]))
            v_max = tk.StringVar(value=str(lims[f"{ax_id}_max"]))
            v_fix = tk.BooleanVar(value=lims[f"{ax_id}_fix"])
            
            # Трассировка изменений полей ввода для автоматического обновления графика в процессе ввода
            v_min.trace_add("write", lambda *args: self.update_view())
            v_max.trace_add("write", lambda *args: self.update_view())
            
            ttk.Entry(f, textvariable=v_min, width=7).pack(side="left", padx=2)
            ttk.Entry(f, textvariable=v_max, width=7).pack(side="left", padx=2)
            ttk.Checkbutton(f, variable=v_fix, command=self.update_view).pack(side="left", padx=5)
            self.axis_ui[ax_id] = {"min": v_min, "max": v_max, "fix": v_fix}

        # --- ПРАВАЯ ЧАСТЬ (ГРАФИК) ---
        self.plot_area = ttk.Frame(paned)
        paned.add(self.plot_area, weight=4)
        self.plotter = Plotter(self.plot_area)

    def update_view(self, *args):
        # 1. Синхронизация осей
        for ax, v in self.axis_ui.items():
            try: self.state.gui_config["axis_limits"].update({
                f"{ax}_min": float(v["min"].get()), f"{ax}_max": float(v["max"].get()), f"{ax}_fix": v["fix"].get()
            })
            except: pass
        
        # 2. Синхронизация маркеров и датчиков
        self.state.gui_config["markers"] = {k: v.get_settings() for k, v in self.marker_ctrls.items()}
        for sc in self.sensor_ctrls: self.state.gui_config["sensors"][sc.sensor_name] = sc.get_settings()
        
        # 3. РАСЧЕТ МОДЕЛЕЙ
        # Прогноз БАКА
        self.state.df_logs, _ = prm.apply_analytic_model(
            self.state.df_logs, self.state.df_registry, 
            cnfg.CONFIG_PREDICTOR_TANK, self.tank_model.get_params()
        )
        # Прогноз ТРУБЫ
        self.state.df_logs, _ = prm.apply_analytic_model(
            self.state.df_logs, self.state.df_registry, 
            cnfg.CONFIG_PREDICTOR_PIPE, self.pipe_model.get_params()
        )
        
        # 4. Отрисовка
        cid = self.cycle_var.get()
        df_curr = self.state.df_logs[self.state.df_logs['cycle_id'] == cid]
        self.plotter.draw(df_curr, self.state.gui_config, cid)
        dm.save_ui_config(self.state.gui_config, cnfg_p.GUI_CONFIG_FILE)

    def _prev_cycle(self):
        """Переключение на предыдущий цикл в списке"""
        labels = self.state.df_registry['cycle_id'].dropna().unique().tolist()
        if not labels: return
        current = self.cycle_var.get()
        if current in labels:
            idx = labels.index(current)
            new_idx = max(0, idx - 1)
            self.cycle_var.set(labels[new_idx])
            self.update_view()

    def _next_cycle(self):
        """Переключение на следующий цикл в списке"""
        labels = self.state.df_registry['cycle_id'].dropna().unique().tolist()
        if not labels: return
        current = self.cycle_var.get()
        if current in labels:
            idx = labels.index(current)
            new_idx = min(len(labels) - 1, idx + 1)
            self.cycle_var.set(labels[new_idx])
            self.update_view()