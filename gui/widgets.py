# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk

COLOR_LIST = ["blue", "red", "green", "black", "orange", "purple", "brown", "gray", "cyan"]
STYLE_LIST = ["-", "--", ":", "-."]

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window or not self.text: return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify='left', background="#ffffe0", 
                 relief='solid', borderwidth=1, font=("tahoma", "9", "normal")).pack(ipadx=1)

    def _hide(self, event=None):
        if self.tip_window: self.tip_window.destroy(); self.tip_window = None

class SensorControl(ttk.Frame):
    def __init__(self, parent, sensor_name, settings, on_change):
        super().__init__(parent)
        self.sensor_name = sensor_name
        self.on_change = on_change
        self.pack(fill="x", pady=1)

        self.var_vis = tk.BooleanVar(value=settings.get("visible", False))
        ttk.Checkbutton(self, variable=self.var_vis, command=self.on_change).pack(side="left")

        short_name = sensor_name.split('\\')[-1].split(' - ')[0]
        ttk.Label(self, text=short_name, width=35, anchor="w").pack(side="left", padx=(0, 5))

        self.var_color = tk.StringVar(value=settings.get("color", "blue"))
        cb_c = ttk.Combobox(self, textvariable=self.var_color, values=COLOR_LIST, width=7, state="readonly")
        cb_c.pack(side="left", padx=2)
        cb_c.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        self.var_ls = tk.StringVar(value=settings.get("ls", "-"))
        cb_l = ttk.Combobox(self, textvariable=self.var_ls, values=STYLE_LIST, width=4, state="readonly")
        cb_l.pack(side="left", padx=2)
        cb_l.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        ttk.Label(self, text="k:").pack(side="left")
        self.v_scale = tk.StringVar(value=str(settings.get("scale", 1.0)))
        ent_s = ttk.Entry(self, textvariable=self.v_scale, width=6)
        ent_s.pack(side="right", padx=2)
        ent_s.bind("<Return>", lambda e: self.on_change())

    def get_settings(self):
        try: scale = float(self.v_scale.get())
        except: scale = 1.0
        return {"visible": self.var_vis.get(), "color": self.var_color.get(), "ls": self.var_ls.get(), "scale": scale}

class ParamSlider(ttk.Frame):
    def __init__(self, parent, key, info, on_change):
        super().__init__(parent)
        self.key = key
        self.on_change = on_change
        self.pack(fill="x", pady=2)

        lbl_text = f"{info.get('label', key)} ({info.get('marker', '')})"
        lbl = ttk.Label(self, text=lbl_text, width=22)
        lbl.pack(side="left")
        Tooltip(lbl, info.get('desc', 'Нет описания'))
        
        self.v_min = tk.StringVar(value=str(info.get('min', 0)))
        ttk.Entry(self, textvariable=self.v_min, width=6).pack(side="left")

        self.v_val = tk.DoubleVar(value=info.get('value', 0))
        self.slider = ttk.Scale(self, from_=float(self.v_min.get()), to=float(info.get('max', 10)), 
                                variable=self.v_val, orient="horizontal", length=150, command=self._move)
        self.slider.pack(side="left", padx=5)

        self.v_max = tk.StringVar(value=str(info.get('max', 10)))
        ttk.Entry(self, textvariable=self.v_max, width=6).pack(side="left")

        self.v_entry_str = tk.StringVar(value=f"{self.v_val.get():.3f}")
        ent = ttk.Entry(self, textvariable=self.v_entry_str, width=8, font=('Consolas', 10, 'bold'))
        ent.pack(side="left", padx=5)
        
        ent.bind("<Return>", self._confirm)
        self.v_min.trace_add("write", self._upd_lim)
        self.v_max.trace_add("write", self._upd_lim)

    def _move(self, val):
        self.v_entry_str.set(f"{float(val):.3f}")
        self.on_change()

    def _confirm(self, e):
        try:
            self.v_val.set(float(self.v_entry_str.get()))
            self.on_change()
        except: self.v_entry_str.set(f"{self.v_val.get():.3f}")

    def _upd_lim(self, *args):
        try: self.slider.configure(from_=float(self.v_min.get()), to=float(self.v_max.get()))
        except: pass

class MarkerControl(ttk.Frame):
    def __init__(self, parent, label, settings, on_change):
        super().__init__(parent)
        self.on_change = on_change
        self.pack(fill="x", pady=1)
        self.var_vis = tk.BooleanVar(value=settings.get("visible", True))
        ttk.Checkbutton(self, variable=self.var_vis, command=on_change).pack(side="left")
        ttk.Label(self, text=label, width=35, font=('Arial', 8, 'italic')).pack(side="left")
        self.var_color = tk.StringVar(value=settings.get("color", "gray"))
        cb = ttk.Combobox(self, textvariable=self.var_color, values=COLOR_LIST, width=8, state="readonly")
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: on_change())
        self.var_ls = tk.StringVar(value=settings.get("ls", "-"))
        ttk.Combobox(self, textvariable=self.var_ls, values=STYLE_LIST, width=5, state="readonly").pack(side="left")

    def get_settings(self):
        return {"visible": self.var_vis.get(), "color": self.var_color.get(), "ls": self.var_ls.get()}