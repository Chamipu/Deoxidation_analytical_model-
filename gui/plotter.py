# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Plotter:
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax2 = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)


    def draw(self, df, ui_config, cycle_label):
        self.ax.clear(); self.ax2.clear()
        if df is None or df.empty:
            self.canvas.draw(); return

        # 1. Маркеры
        m_cfg = ui_config.get("markers", {})
        for sig, key in [("X1213_OPN - клапан порошка откр", "OPN"), ("X1213_CLS - клапан порошка закр_median", "CLS")]:
            c = m_cfg.get(key, {})
            if c.get("visible") and sig in df.columns:
                for t in df[df[sig] == 1]['t_relative']:
                    self.ax.axvline(x=t, color=c.get("color"), ls=c.get("ls"), lw=2, alpha=0.8)

        # 2. Датчики
        s_cfg = ui_config.get("sensors", {})
        for s, settings in s_cfg.items():
            if settings.get("visible") and s in df.columns:
                ax = self.ax2 if "Weigh" in s else self.ax
                # Срез [:15] удален для вывода полного названия датчика в легенде
                ax.plot(df['t_relative'], df[s] * settings.get("scale", 1.0),
                        color=settings.get("color"), ls=settings.get("ls"),
                        label=s.split('\\')[-1], lw=1.5)

        # 3. Оси
        lims = ui_config.get("axis_limits", {})
        try:
            if lims.get("x_fix"): self.ax.set_xlim(lims["x_min"], lims["x_max"])
            if lims.get("y1_fix"): self.ax.set_ylim(lims["y1_min"], lims["y1_max"])
            if lims.get("y2_fix"): self.ax2.set_ylim(lims["y2_min"], lims["y2_max"])
        except: pass

        self.ax.set_title(f"Цикл: {cycle_label}")
        self.ax.grid(True, ls=':', alpha=0.5)
        h1, l1 = self.ax.get_legend_handles_labels()
        h2, l2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(h1+h2, l1+l2, loc='upper left', prop={'size': 8})
        self.canvas.draw()