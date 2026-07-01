# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import config_paths as cnfg_p
from scripts import data_manager as dm
from gui.tabs.sim_tab import SimulationTab
from gui.tabs.optimization_tab import OptimizationTab


class AppState:
    def __init__(self):
        self.df_logs = dm.load_database(cnfg_p.LOGS_AVG_FILE)
        self.df_registry = dm.load_database(cnfg_p.REGISTRY_AVG_FILE)
        self.gui_config = dm.load_ui_config(cnfg_p.GUI_CONFIG_FILE)

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumatic Tuner Pro v4.0")
        self.state = AppState()
        
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        
        self.sim_tab = SimulationTab(nb, self.state)
        nb.add(self.sim_tab, text=" Моделирование ")
        
        self.optimization_tab = OptimizationTab(nb, self.state)
        nb.add(self.optimization_tab, text=" Оптимизация (WIP) ")
        
        self.root.state('zoomed')

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()