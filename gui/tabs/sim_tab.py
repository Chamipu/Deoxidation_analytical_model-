# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog
import config as cnfg
import config_paths as cnfg_p
from gui.widgets import ParamSlider, SensorControl, MarkerControl
from gui.plotter import Plotter
from scripts import data_manager as dm
from scripts import pressure_predictor_lite as prm

class SimulationTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self._setup_layout()
        self._load_initial_physics()

    def _setup_layout(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- LEFT SIDEBAR ---
        self.left_f = ttk.Frame(paned, padding=5)
        paned.add(self.left_f, weight=1)

        # Buttons
        btn_f = ttk.Frame(self.left_f)
        btn_f.pack(fill="x", pady=5)
        ttk.Button(btn_f, text="ОТКРЫТЬ", command=self.load_physics).pack(side="left", expand=True)
        ttk.Button(btn_f, text="СОХРАНИТЬ", command=self.save_physics).pack(side="left", expand=True)
        ttk.Button(btn_f, text="СБРОС", command=self.reset_physics).pack(side="left", expand=True)

        # Cycle Selection
        labels = self.state.df_registry['cycle_id'].dropna().unique().tolist()
        self.cycle_var = tk.StringVar(value=labels[-1] if labels else "")
        cb = ttk.Combobox(self.left_f, textvariable=self.cycle_var, values=labels, state="readonly")
        cb.pack(fill="x", pady=5); cb.bind("<<ComboboxSelected>>", lambda e: self.update_view())

        # Scrollable Area
        canvas = tk.Canvas(self.left_f, width=500)
        sb = ttk.Scrollbar(self.left_f, orient="vertical", command=canvas.yview)
        self.scroll_f = ttk.Frame(canvas)
        self.scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.scroll_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)

        # 1. Markers
        ttk.Label(self.scroll_f, text="МАРКЕРЫ:", font=('Arial', 9, 'bold')).pack(anchor="w")
        self.marker_ctrls = {
            "OPN": MarkerControl(self.scroll_f, "Открытие (OPN)", self.state.gui_config["markers"]["OPN"], self.update_view),
            "CLS": MarkerControl(self.scroll_f, "Закрытие (CLS)", self.state.gui_config["markers"]["CLS"], self.update_view)
        }

        # 2. Sensors
        ttk.Label(self.scroll_f, text="ДАТЧИКИ:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        self.sensor_ctrls = []
        for s in cnfg.SENSORS_LIST:
            cfg = self.state.gui_config["sensors"].get(s, {"visible": False})
            self.sensor_ctrls.append(SensorControl(self.scroll_f, s, cfg, self.update_view))

        # 3. Physics Sliders
        ttk.Label(self.scroll_f, text="ПАРАМЕТРЫ МОДЕЛИ:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        self.physics_container = ttk.Frame(self.scroll_f)
        self.physics_container.pack(fill="x")

        # 4. Axis Limits
        ttk.Label(self.left_f, text="ОСИ (Мин/Макс/Фикс):", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        self.axis_ui = {}
        for ax_id, label in [("x", "Время"), ("y1", "Давление"), ("y2", "Вес")]:
            f = ttk.Frame(self.left_f)
            f.pack(fill="x")
            ttk.Label(f, text=label, width=10).pack(side="left")
            lims = self.state.gui_config["axis_limits"]
            v_min, v_max, v_fix = tk.StringVar(value=lims[f"{ax_id}_min"]), tk.StringVar(value=lims[f"{ax_id}_max"]), tk.BooleanVar(value=lims[f"{ax_id}_fix"])
            ttk.Entry(f, textvariable=v_min, width=7).pack(side="left")
            ttk.Entry(f, textvariable=v_max, width=7).pack(side="left")
            ttk.Checkbutton(f, variable=v_fix, command=self.update_view).pack(side="left")
            self.axis_ui[ax_id] = {"min": v_min, "max": v_max, "fix": v_fix}

        # --- RIGHT PLOT ---
        self.plot_area = ttk.Frame(paned)
        paned.add(self.plot_area, weight=4)
        self.plotter = Plotter(self.plot_area)

    def _load_initial_physics(self):
        self._load_phys_from_path(cnfg_p.DEFAULT_MODEL_PARAMS_FILE)

    def _load_phys_from_path(self, path):
        res = dm.load_config(path)
        if res:
            self.physics_full = res[0]
            for w in self.physics_container.winfo_children(): w.destroy()
            self.sliders = [ParamSlider(self.physics_container, k, v, self.update_view) for k, v in self.physics_full.items()]
            self.update_view()

    def reset_physics(self): self._load_phys_from_path(cnfg_p.DEFAULT_MODEL_PARAMS_FILE)
    
    def load_physics(self):
        path = filedialog.askopenfilename()
        if path: self._load_phys_from_path(path)

    def save_physics(self):
        for s in self.sliders:
            self.physics_full[s.key].update({"value": s.v_val.get(), "min": float(s.v_min.get()), "max": float(s.v_max.get())})
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if path: dm.save_config(self.physics_full, path)

    def update_view(self, *args):
        # Sync Axis
        for ax, v in self.axis_ui.items():
            try:
                self.state.gui_config["axis_limits"].update({f"{ax}_min": float(v["min"].get()), f"{ax}_max": float(v["max"].get()), f"{ax}_fix": v["fix"].get()})
            except: pass
        # Sync Markers & Sensors
        self.state.gui_config["markers"] = {k: v.get_settings() for k, v in self.marker_ctrls.items()}
        for sc in self.sensor_ctrls: self.state.gui_config["sensors"][sc.sensor_name] = sc.get_settings()
        
        # Model Calculation
        p_flat = {s.key: s.v_val.get() for s in self.sliders}
        self.state.df_logs, _ = prm.apply_analytic_model(self.state.df_logs, self.state.df_registry, cnfg.CONFIG_PREDICTOR_TANK, p_flat)
        
        cid = self.cycle_var.get()
        df_curr = self.state.df_logs[self.state.df_logs['cycle_id'] == cid]
        self.plotter.draw(df_curr, self.state.gui_config, cid)
        dm.save_ui_config(self.state.gui_config, cnfg_p.GUI_CONFIG_FILE)