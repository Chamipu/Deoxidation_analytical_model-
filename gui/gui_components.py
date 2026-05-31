#  Файл gui_components.py с определением компонентов интерфейса: контролы для датчиков, параметров модели и вертикальных линий.
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk

COLOR_LIST = ["blue", "red", "green", "black", "orange", "purple", "brown", "gray", "cyan"]
STYLE_LIST = ["-", "--", ":", "-."]

class Tooltip:
    """Вспомогательный класс для всплывающих подсказок."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text: return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "9", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw: tw.destroy()

class SensorControl:
    """Ряд управления датчиком."""
    def __init__(self, parent, sensor_name, settings, on_change):
        self.sensor_name = sensor_name
        self.on_change = on_change
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)

        self.var_vis = tk.BooleanVar(value=settings.get("visible", False))
        ttk.Checkbutton(frame, variable=self.var_vis, command=self.on_change).pack(side="left")

        short_name = sensor_name.split('\\')[-1].split(' - ')[0]
        ttk.Label(frame, text=short_name, width=40, anchor="w").pack(side="left", padx=(0, 10))

        self.var_color = tk.StringVar(value=settings.get("color", "blue"))
        cb_c = ttk.Combobox(frame, textvariable=self.var_color, values=COLOR_LIST, width=8, state="readonly")
        cb_c.pack(side="left", padx=2)
        cb_c.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        self.var_ls = tk.StringVar(value=settings.get("ls", "-"))
        cb_l = ttk.Combobox(frame, textvariable=self.var_ls, values=STYLE_LIST, width=5, state="readonly")
        cb_l.pack(side="left", padx=2)

        # 4. Поле ввода масштаба (справа)
        ttk.Label(frame, text="k:").pack(side="left") # Подпись для ясности
        self.v_scale = tk.StringVar(value=str(settings.get("scale", 1.0)))
        self.e_scale = ttk.Entry(frame, textvariable=self.v_scale, width=6)
        self.e_scale.pack(side="right", padx=2)

        cb_l.bind("<<ComboboxSelected>>", lambda e: self.on_change())

    def get_settings(self):
        return {
            "visible": self.var_vis.get(), 
            "color": self.var_color.get(), 
            "ls": self.var_ls.get(),
            "scale": float(self.v_scale.get() or 1.0) # Новое поле
            }

class ParamSlider:
    """Ряд управления физическим параметром с вводом значения и Tooltip."""
    def __init__(self, parent, key, info, on_change):
        self.key = key
        self.on_change = on_change
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        # Метка с Tooltip
        lbl_text = f"{info.get('label', key)} ({info.get('marker')})"
        self.label = ttk.Label(row, text=lbl_text, width=22)
        self.label.pack(side="left")
        Tooltip(self.label, info.get('desc', 'Нет описания'))
        
        # Мин
        self.v_min = tk.StringVar(value=str(info.get('min')))
        ttk.Entry(row, textvariable=self.v_min, width=6).pack(side="left")

        # Слайдер
        self.v_val = tk.DoubleVar(value=info.get('value'))
        self.slider = ttk.Scale(row, from_=float(self.v_min.get()), to=float(info.get('max', 10)), 
                                variable=self.v_val, orient="horizontal", length=150, 
                                command=self._on_slider_move)
        self.slider.pack(side="left", padx=5)

        # Макс
        self.v_max = tk.StringVar(value=str(info.get('max')))
        ttk.Entry(row, textvariable=self.v_max, width=6).pack(side="left")

        # Поле ввода текущего значения (вместо Label)
        self.v_entry_str = tk.StringVar(value=f"{self.v_val.get():.3f}")
        self.entry_val = ttk.Entry(row, textvariable=self.v_entry_str, width=8, font=('Consolas', 10, 'bold'))
        self.entry_val.pack(side="left", padx=5)
        
        # События синхронизации
        self.entry_val.bind("<Return>", self._on_entry_confirm)
        self.entry_val.bind("<FocusOut>", self._on_entry_confirm)
        self.v_min.trace_add("write", self._upd_lim)
        self.v_max.trace_add("write", self._upd_lim)

    def _on_slider_move(self, val):
        self.v_entry_str.set(f"{float(val):.3f}")
        self.on_change()

    def _on_entry_confirm(self, event):
        try:
            val = float(self.v_entry_str.get())
            self.v_val.set(val)
            self.on_change()
        except ValueError:
            self.v_entry_str.set(f"{self.v_val.get():.3f}")

    def _upd_lim(self, *args):
        try: self.slider.configure(from_=float(self.v_min.get()), to=float(self.v_max.get()))
        except: pass

class MarkerControl:
    """Специальный контроллер для вертикальных линий."""
    def __init__(self, parent, label_text, settings, on_change):
        self.on_change = on_change
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)

        self.var_vis = tk.BooleanVar(value=settings.get("visible", True))
        ttk.Checkbutton(frame, variable=self.var_vis, command=self.on_change).pack(side="left")

        ttk.Label(frame, text=label_text, width=40, font=('Arial', 8, 'italic')).pack(side="left", padx=(0, 10))

        self.var_color = tk.StringVar(value=settings.get("color", "green"))
        cb_c = ttk.Combobox(frame, textvariable=self.var_color, values=COLOR_LIST, width=8, state="readonly")
        cb_c.pack(side="left", padx=2)
        cb_c.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        self.var_ls = tk.StringVar(value=settings.get("ls", "-"))
        cb_l = ttk.Combobox(frame, textvariable=self.var_ls, values=STYLE_LIST, width=5, state="readonly")
        cb_l.pack(side="left", padx=2)
        cb_l.bind("<<ComboboxSelected>>", lambda e: self.on_change())

    def get_settings(self):
        return {"visible": self.var_vis.get(), "color": self.var_color.get(), "ls": self.var_ls.get()}