# Файл gui_plotter.py с классом Plotter для отрисовки графиков в интерфейсе. Использует matplotlib для визуализации данных и отображения маркеров событий.
# -*- coding: utf-8 -*-
from matplotlib import scale
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Plotter:
    def __init__(self, parent_frame):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax2 = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)

    def draw(self, df_plot, ui_config, cycle_label):
        self.ax.clear()
        self.ax2.clear()
        if df_plot.empty:
            self.canvas.draw()
            return

        marker_cfg = ui_config.get("markers", {})

        # 1. Маркеры
        for sig, key in [("X1213_OPN - клапан порошка откр", "OPN"), 
                         ("X1213_CLS - клапан порошка закр_median", "CLS")]:
            cfg = marker_cfg.get(key, {})
            if cfg.get("visible", True) and sig in df_plot.columns:
                times = df_plot[df_plot[sig] == 1]['t_relative']
                for t in times:
                    self.ax.axvline(x=t, color=cfg.get("color", "gray"), 
                                    linestyle=cfg.get("ls", "-"), linewidth=2, alpha=0.8)

        # 2. Отрисовка датчиков
        sensor_settings = ui_config.get("sensors", {})
        for sensor, settings in sensor_settings.items():
            if not settings.get("visible", False) or sensor not in df_plot.columns:
                continue
            
            scale = settings.get("scale", 1.0)
            data_to_plot = df_plot[sensor] * scale  # Применение масштаба

            target_ax = self.ax2 if "Weigh" in sensor else self.ax
            target_ax.plot(
                df_plot['t_relative'], data_to_plot,
                color=settings.get("color", "blue"),
                linestyle=settings.get("ls", "-"),
                label=sensor.split('\\')[-1].split(' - ')[0],
                linewidth=1.5
            )

        # 3. Применение фиксированных масштабов
        ax_lim = ui_config.get("axis_limits", {})
        try:
            if ax_lim.get("x_fix"): self.ax.set_xlim(ax_lim["x_min"], ax_lim["x_max"])
            if ax_lim.get("y1_fix"): self.ax.set_ylim(ax_lim["y1_min"], ax_lim["y1_max"])
            if ax_lim.get("y2_fix"): self.ax2.set_ylim(ax_lim["y2_min"], ax_lim["y2_max"])
        except Exception as e:
            print(f"Ошибка масштабирования осей: {e}")

        self.ax.set_title(f"Цикл: {cycle_label}", fontsize=10)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.set_ylabel("Давление (бар) / Сигналы")
        self.ax2.set_ylabel("Вес (кг)")
        
        h1, l1 = self.ax.get_legend_handles_labels()
        h2, l2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(h1 + h2, l1 + l2, loc='upper left', prop={'size': 8})
        
        self.canvas.draw()