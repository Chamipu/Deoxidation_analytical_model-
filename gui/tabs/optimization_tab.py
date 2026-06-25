# -*- coding: utf-8 -*-
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import config as cnfg
import config_paths as cnfg_p
from gui.widgets import PhysicsModelBlock, COLOR_LIST, STYLE_LIST
from scripts import data_manager as dm
from scripts.independent_linear_predictor import inverse_predictorrrr as inv


class OptPlotter:
    """Встроенный плоттер для визуализации результатов оптимизации."""

    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)

    def draw(self, time_grid, goal_raw, prediction=None, target=None, presets_t=None, presets_y=None, ui_config=None):
        """Draw plot using optional UI configuration for colors/linestyles/visibility/scale."""
        self.ax.clear()
        if ui_config is None:
            ui_config = {}

        plots_cfg = ui_config.get("plots", {})

        # goal
        g_cfg = plots_cfg.get("goal", {})
        if g_cfg.get("visible", True):
            self.ax.plot(time_grid, goal_raw, label=g_cfg.get("label", "Желаемый профиль (цель)"),
                         color=g_cfg.get("color", "#1f77b4"), lw=float(g_cfg.get("lw", 2.5)), ls=g_cfg.get("ls", "-"))

        # prediction
        if prediction is not None:
            p_cfg = plots_cfg.get("prediction", {})
            if p_cfg.get("visible", True):
                self.ax.plot(
                    time_grid, prediction,
                    label=p_cfg.get("label", "Физический отклик модели (прогноз)"),
                    color=p_cfg.get("color", "#d62728"), lw=float(p_cfg.get("lw", 2)), ls=p_cfg.get("ls", "--"),
                )

        # target (apply scale from config instead of hardcoded 0.1)
        if target is not None:
            t_cfg = plots_cfg.get("target", {})
            if t_cfg.get("visible", True):
                scale = float(t_cfg.get("scale", 0.1))
                self.ax.step(
                    time_grid, target * scale,
                    label=t_cfg.get("label", "Подобранные уставки ПЛК (бар)"),
                    color=t_cfg.get("color", "#2ca02c"), alpha=float(t_cfg.get("alpha", 0.7)), where="post",
                )

        # presets
        if presets_t is not None and presets_y is not None:
            pr_cfg = plots_cfg.get("presets", {})
            if pr_cfg.get("visible", True):
                self.ax.scatter(
                    presets_t, presets_y,
                    color=pr_cfg.get("color", "#e377c2"), marker=pr_cfg.get("marker", "o"), s=int(pr_cfg.get("s", 60)), zorder=5,
                    label=pr_cfg.get("label", "Заданные точки"), edgecolor=pr_cfg.get("edgecolor", "black"),
                )

        self.ax.set_title(
            "Сравнение желаемого профиля и физического отклика системы",
            fontsize=10, fontweight="bold",
        )
        self.ax.set_xlabel("Относительное время, с", fontsize=9)
        self.ax.set_ylabel("Давление, бар", fontsize=9)
        self.ax.grid(True, ls=":", alpha=0.6)
        self.ax.legend(loc="upper left", prop={'size': 8})
        self.canvas.draw()
    


class StdoutRedirector:
    """Перенаправляет стандартный вывод print() в текстовое поле Tkinter."""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class OptimizationTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self.queue = queue.Queue()
        self.sliders = []
        self.opt_thread = None
        self.optimized_results = None
        self.stdout_orig = sys.stdout
        self._phases = cnfg.CONFIG_GENERATE_TARGET_TANK['phases']

        self._setup_layout()
        self._rebuild_preset_sliders()
        self.update_raw_preview()

    def _setup_layout(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_panel = ttk.Frame(paned, padding=5)
        paned.add(left_panel, weight=1)

        canvas = tk.Canvas(left_panel, width=320)
        sb = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scroll_f = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scroll_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        f_pts = ttk.LabelFrame(self.scroll_f, text=" ЖЕЛАЕМЫЙ ПРОФИЛЬ ДАВЛЕНИЯ ", padding=8)
        f_pts.pack(fill="x", pady=5)

        ttk.Label(f_pts, text="Кол-во ключевых точек:").pack(anchor="w")
        self.v_pts_count = tk.IntVar(value=5)
        sp = ttk.Spinbox(
            f_pts, from_=3, to=15, textvariable=self.v_pts_count,
            width=10, command=self._rebuild_preset_sliders,
        )
        sp.pack(fill="x", pady=(0, 10))
        sp.bind("<Return>", lambda e: self._rebuild_preset_sliders())

        self.sliders_container = ttk.Frame(f_pts)
        self.sliders_container.pack(fill="x")

        self.tank_model = PhysicsModelBlock(
            self.scroll_f,
            "ФИЗИКА СИСТЕМЫ (ПАРАМЕТРЫ БАКА)",
            cnfg_p.DEFAULT_MODEL_PARAMS_TANK_FILE,
            self.update_raw_preview,
        )
        self.tank_model.pack(fill="x", pady=5)

        f_action = ttk.LabelFrame(self.scroll_f, text=" ПАРАМЕТРЫ ПОДБОРА ", padding=8)
        f_action.pack(fill="x", pady=5)

        ttk.Label(f_action, text="Длительность импульса (сек):").pack(anchor="w")
        self.v_duration = tk.DoubleVar(value=6.0)
        ttk.Entry(f_action, textvariable=self.v_duration, width=10).pack(fill="x", pady=3)

        self.lbl_status = ttk.Label(
            f_action, text="Статус: Ожидание запуска",
            font=("Arial", 9, "bold"), foreground="#1f77b4",
        )
        self.lbl_status.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(
            f_action, text="🚀 Подобрать уставки ПЛК", command=self.start_optimization,
        )
        self.btn_run.pack(fill="x", pady=2)

        self.btn_reset_opt = ttk.Button(
            f_action, text="🔄 Сбросить расчет", command=self.reset_optimization_results,
        )
        self.btn_reset_opt.pack(fill="x", pady=2)

        # --- ПАРАМЕТРЫ ОТОБРАЖЕНИЯ ГРАФИКА ---
        f_plot = ttk.LabelFrame(self.scroll_f, text=" НАСТРОЙКИ ГРАФИКА ", padding=8)
        f_plot.pack(fill="x", pady=5)

        # Ensure plots config exists
        if "plots" not in self.state.gui_config:
            self.state.gui_config["plots"] = {}

        self.plot_controls = {}
        plot_items = [
            ("goal", "Желаемый профиль"),
            ("prediction", "Прогноз модели"),
            ("target", "Уставки ПЛК"),
            ("presets", "Заданные точки"),
        ]

        for key, title in plot_items:
            cfg = self.state.gui_config["plots"].get(key, {})
            f = ttk.Frame(f_plot)
            f.pack(fill="x", pady=2)

            var_vis = tk.BooleanVar(value=cfg.get("visible", True))
            ttk.Checkbutton(f, variable=var_vis, command=self._on_plot_ctrl_change).pack(side="left")

            ttk.Label(f, text=title, width=18).pack(side="left")

            var_color = tk.StringVar(value=cfg.get("color", "blue"))
            cb_c = ttk.Combobox(f, textvariable=var_color, values=COLOR_LIST, width=8, state="readonly")
            cb_c.pack(side="left", padx=2)
            cb_c.bind("<<ComboboxSelected>>", lambda e: self._on_plot_ctrl_change())

            var_ls = tk.StringVar(value=cfg.get("ls", "-"))
            cb_l = ttk.Combobox(f, textvariable=var_ls, values=STYLE_LIST, width=5, state="readonly")
            cb_l.pack(side="left", padx=2)
            cb_l.bind("<<ComboboxSelected>>", lambda e: self._on_plot_ctrl_change())

            ttk.Label(f, text="k:").pack(side="left", padx=(6,0))
            v_scale = tk.StringVar(value=str(cfg.get("scale", 1.0)))
            ent_s = ttk.Entry(f, textvariable=v_scale, width=6)
            ent_s.pack(side="left", padx=2)
            ent_s.bind("<Return>", lambda e: self._on_plot_ctrl_change())

            self.plot_controls[key] = {"vis": var_vis, "color": var_color, "ls": var_ls, "scale": v_scale}

        right_panel = ttk.PanedWindow(paned, orient="vertical")
        paned.add(right_panel, weight=3)

        self.plot_container = ttk.Frame(right_panel)
        right_panel.add(self.plot_container, weight=3)
        self.plotter = OptPlotter(self.plot_container)

        log_container = ttk.LabelFrame(right_panel, text=" ТЕХНИЧЕСКИЙ ЛОГ ОПТИМИЗАЦИИ ")
        right_panel.add(log_container, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_container, height=8, font=("Consolas", 9),
            bg="#1e1e1e", fg="#d4d4d4",
        )
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)

    def _rebuild_preset_sliders(self):
        for w in self.sliders_container.winfo_children():
            w.destroy()
        self.sliders = []

        count = self.v_pts_count.get()
        default_values = np.linspace(0.2, 0.35, count)

        for i in range(count):
            f = ttk.Frame(self.sliders_container)
            f.pack(fill="x", pady=2)

            ttk.Label(f, text=f"Точка {i+1}:", width=8).pack(side="left")

            val_var = tk.DoubleVar(value=default_values[i])
            val_lbl = ttk.Label(f, text=f"{val_var.get():.2f} бар", width=8)

            slider = ttk.Scale(
                f, from_=0.0, to=1.0,
                variable=val_var, orient="horizontal",
            )
            slider.configure(command=self._make_slider_callback(val_lbl, val_var))
            slider.pack(side="left", fill="x", expand=True, padx=5)
            val_lbl.pack(side="right")

            self.sliders.append(val_var)

        self.update_raw_preview()

    def _on_plot_ctrl_change(self):
        """Sync plot controls into state.gui_config and persist."""
        plots = self.state.gui_config.get("plots", {})
        for key, ctrl in self.plot_controls.items():
            try:
                scale = float(ctrl["scale"].get())
            except:
                scale = 1.0
            plots[key] = {
                "visible": bool(ctrl["vis"].get()),
                "color": str(ctrl["color"].get()),
                "ls": str(ctrl["ls"].get()),
                "scale": scale,
            }
        self.state.gui_config["plots"] = plots
        try:
            dm.save_ui_config(self.state.gui_config, cnfg_p.GUI_CONFIG_FILE)
        except: pass
        # Redraw preview using updated settings
        self.update_raw_preview()

    def _make_slider_callback(self, label_widget, var):
        def callback(value):
            label_widget.configure(text=f"{float(value):.2f} бар")
            self.update_raw_preview()
        return callback

    def get_current_presets(self):
        return [s.get() for s in self.sliders]

    def _get_duration(self):
        duration = self.v_duration.get()
        if duration <= 0:
            raise ValueError("Длительность импульса должна быть больше нуля")
        return duration

    def reset_optimization_results(self):
        self.optimized_results = None
        self.lbl_status.configure(text="Статус: Результаты сброшены", foreground="#1f77b4")
        self.update_raw_preview()

    def _draw_plot(self, time_grid, goal_raw, preset_times, presets_y):
        prediction = None
        target = None
        if self.optimized_results is not None:
            prediction = self.optimized_results.get("prediction")
            target = self.optimized_results.get("target")

        self.plotter.draw(
            time_grid=time_grid,
            goal_raw=goal_raw,
            prediction=prediction,
            target=target,
            presets_t=preset_times,
            presets_y=presets_y,
            ui_config=self.state.gui_config,
        )

    def update_raw_preview(self, *args):
        try:
            presets = self.get_current_presets()
            duration = self._get_duration()
            time_grid, goal_raw, preset_times = inv.build_goal_profile_preview(
                presets, duration, self._phases,
            )
            self._draw_plot(time_grid, goal_raw, preset_times, presets)
        except Exception as e:
            print(f"[UI Preview Error] {e}", file=sys.stderr)

    def start_optimization(self):
        if self.opt_thread and self.opt_thread.is_alive():
            return

        try:
            duration = self._get_duration()
        except ValueError as e:
            self.lbl_status.configure(text=f"Статус: {e}", foreground="#d62728")
            return

        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="Статус: Выполняется расчет...", foreground="orange")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, ">>> Запуск оптимизатора дифференциальной эволюции...\n")

        self.stdout_orig = sys.stdout
        sys.stdout = StdoutRedirector(self.log_text)

        presets = self.get_current_presets()
        flat_params = self.tank_model.get_params()

        self.opt_thread = threading.Thread(
            target=self._run_optimization_process,
            args=(presets, duration, flat_params),
            daemon=True,
        )
        self.opt_thread.start()
        self.after(100, self._check_queue)

    def _run_optimization_process(self, presets, duration, flat_params):
        try:
            result = inv.run_preset_optimization(
                presets, duration, flat_params, self._phases,
            )
            self.queue.put({"status": "success", **result})
        except Exception as e:
            self.queue.put({"status": "error", "message": str(e)})

    def _restore_stdout(self):
        if hasattr(self, 'stdout_orig'):
            sys.stdout = self.stdout_orig

    def _check_queue(self):
        try:
            res = self.queue.get_nowait()
        except queue.Empty:
            self.after(100, self._check_queue)
            return

        self._restore_stdout()
        self.btn_run.configure(state="normal")

        if res["status"] == "success":
            self.optimized_results = res
            self._draw_plot(
                res["time_grid"],
                res["goal_raw"],
                res["preset_times"],
                self.get_current_presets(),
            )
            self.lbl_status.configure(
                text="Статус: Подбор успешно завершен", foreground="#2ca02c",
            )
            self.log_text.insert(tk.END, "\n>>> Подбор успешно завершен!\n")
            self.log_text.insert(
                tk.END, f"Оптимальные пресеты пульта (%): {res['best_presets']}\n",
            )
            self.log_text.see(tk.END)
        else:
            self.optimized_results = None
            self.lbl_status.configure(text="Статус: Ошибка подбора", foreground="#d62728")
            self.log_text.insert(tk.END, f"\nОшибка при оптимизации: {res['message']}\n")
