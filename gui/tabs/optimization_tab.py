# -*- coding: utf-8 -*-
import sys
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import config as cnfg
import config_paths as cnfg_p
from scripts import data_manager as dm
from scripts.independent_linear_predictor import pressure_predictor_lite as prm
from gui.widgets import PhysicsModelBlock

from scripts.independent_linear_predictor import invers_optimizator as iopt



class OptPlotter:
    """Встроенный плоттер для визуализации результатов оптимизации"""
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)

    def draw(self, time_grid, goal_raw, prediction=None, target=None, presets_t=None, presets_y=None):
        self.ax.clear()
        
        # 1. Желаемая кривая (таргет)
        self.ax.plot(time_grid, goal_raw, label="Желаемый профиль (цель)", color="#1f77b4", lw=2.5)
        
        # 2. Результат симуляции по подобранным уставкам
        if prediction is not None:
            self.ax.plot(time_grid, prediction, label="Физический отклик модели (прогноз)", color="#d62728", lw=2, ls="--")
            
        # 3. Дискретные ступени уставок ПЛК (приведенные к барам)
        if target is not None:
            # target переводит в бары [0..6.0], что идеально ложится на ту же ось давления.
            self.ax.step(time_grid, target * 0.1, label="Подобранные уставки ПЛК (бар)", color="#2ca02c", alpha=0.7, where="post")
            
        # 4. Точки, которые задал пользователь
        if presets_t is not None and presets_y is not None:
            self.ax.scatter(presets_t, presets_y, color="#e377c2", marker="o", s=60, zorder=5, label="Заданные точки", edgecolor="black")

        self.ax.set_title("Сравнение желаемого профиля и физического отклика системы", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Относительное время, с", fontsize=9)
        self.ax.set_ylabel("Давление, бар", fontsize=9)
        self.ax.grid(True, ls=":", alpha=0.6)
        self.ax.legend(loc="upper left", prop={'size': 8})
        self.canvas.draw()


class StdoutRedirector:
    """Перенаправляет стандартный вывод print() в текстовое поле Tkinter"""
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
        self.optimized_results = None  # Здесь будут храниться результаты последнего успешного расчета
        
        self._setup_layout()
        self._rebuild_preset_sliders()
        self.update_raw_preview()

    def _setup_layout(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- ЛЕВАЯ ПАНЕЛЬ (САЙДБАР) ---
        left_panel = ttk.Frame(paned, padding=5)
        paned.add(left_panel, weight=1)

        # Скроллбар для левой панели
        canvas = tk.Canvas(left_panel, width=320)
        sb = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scroll_f = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scroll_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Блок 1. Настройка опорных точек
        f_pts = ttk.LabelFrame(self.scroll_f, text=" ЖЕЛАЕМЫЙ ПРОФИЛЬ ДАВЛЕНИЯ ", padding=8)
        f_pts.pack(fill="x", pady=5)

        ttk.Label(f_pts, text="Кол-во ключевых точек:").pack(anchor="w")
        self.v_pts_count = tk.IntVar(value=5)
        sp = ttk.Spinbox(f_pts, from_=3, to=15, textvariable=self.v_pts_count, width=10, command=self._rebuild_preset_sliders)
        sp.pack(fill="x", pady=(0, 10))
        sp.bind("<Return>", lambda e: self._rebuild_preset_sliders())

        # Контейнер для динамических ползунков точек
        self.sliders_container = ttk.Frame(f_pts)
        self.sliders_container.pack(fill="x")

        # Блок 2. Параметры физической модели бака
        self.tank_model = PhysicsModelBlock(
            self.scroll_f, 
            "ФИЗИКА СИСТЕМЫ (ПАРАМЕТРЫ БАКА)", 
            cnfg_p.DEFAULT_MODEL_PARAMS_TANK_FILE, 
            self.update_raw_preview
        )
        self.tank_model.pack(fill="x", pady=5)

        # Блок 3. Параметры оптимизации и кнопка запуска
        f_action = ttk.LabelFrame(self.scroll_f, text=" ПАРАМЕТРЫ ПОДБОРА ", padding=8)
        f_action.pack(fill="x", pady=5)

        ttk.Label(f_action, text="Длительность импульса (сек):").pack(anchor="w")
        self.v_duration = tk.DoubleVar(value=6.0)
        ttk.Entry(f_action, textvariable=self.v_duration, width=10).pack(fill="x", pady=3)

        # Поле вывода текущего статуса работы
        self.lbl_status = ttk.Label(f_action, text="Статус: Ожидание запуска", font=("Arial", 9, "bold"), foreground="#1f77b4")
        self.lbl_status.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(f_action, text="🚀 Подобрать уставки ПЛК", command=self.start_optimization)
        self.btn_run.pack(fill="x", pady=2)

        self.btn_reset_opt = ttk.Button(f_action, text="🔄 Сбросить расчет", command=self.reset_optimization_results)
        self.btn_reset_opt.pack(fill="x", pady=2)

        # --- ПРАВАЯ ЧАСТЬ (ГРАФИК И КЛАССИЧЕСКИЙ ЛОГ) ---
        right_panel = ttk.PanedWindow(paned, orient="vertical")
        paned.add(right_panel, weight=3)

        # Верхняя половина правой части - График
        self.plot_container = ttk.Frame(right_panel)
        right_panel.add(self.plot_container, weight=3)
        self.plotter = OptPlotter(self.plot_container)

        # Нижняя половина правой части - Лог оптимизации
        log_container = ttk.LabelFrame(right_panel, text=" ТЕХНИЧЕСКИЙ ЛОГ ОПТИМИЗАЦИИ ")
        right_panel.add(log_container, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_container, height=8, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)

    def _rebuild_preset_sliders(self):
        """Перестраивает список ползунков при изменении количества опорных точек"""
        for w in self.sliders_container.winfo_children():
            w.destroy()
        self.sliders = []

        count = self.v_pts_count.get()
        # Равномерно распределяем значения по умолчанию от 0.2 до 0.35 бар
        default_values = np.linspace(0.2, 0.35, count)

        for i in range(count):
            f = ttk.Frame(self.sliders_container)
            f.pack(fill="x", pady=2)
            
            ttk.Label(f, text=f"Точка {i+1}:", width=8).pack(side="left")
            
            val_var = tk.DoubleVar(value=default_values[i])
            val_lbl = ttk.Label(f, text=f"{val_var.get():.2f} бар", width=8)
            
            # Слайдеры теперь имеют диапазон от 0.0 до 1.0 бар (реальное физическое давление)
            slider = ttk.Scale(
                f, from_=0.0, to=1.0, 
                variable=val_var, 
                orient="horizontal"
            )
            # Привязываем изменение значения к функции обратного вызова
            slider.configure(command=self._make_slider_callback(val_lbl, val_var))
            slider.pack(side="left", fill="x", expand=True, padx=5)
            val_lbl.pack(side="right")
            
            self.sliders.append(val_var)
            
        self.update_raw_preview()

    def _make_slider_callback(self, label_widget, var):
        """Генерирует замыкание для корректного обновления виджетов и графиков при сдвиге ползунка"""
        def callback(value):
            label_widget.configure(text=f"{float(value):.2f} бар")
            self.update_raw_preview()
        return callback

    def get_current_presets(self):
        """Возвращает текущие значения желаемых давлений из ползунков"""
        return [s.get() for s in self.sliders]

    def reset_optimization_results(self):
        """Очищает графики симуляции и переводит плоттер в режим просмотра уставки"""
        self.optimized_results = None
        self.lbl_status.configure(text="Статус: Результаты сброшены", foreground="#1f77b4")
        self.update_raw_preview()

    def update_raw_preview(self, *args):
        """Мгновенно перерисовывает желаемый график при перемещении ползунков"""
        try:
            presets = self.get_current_presets()
            duration = self.v_duration.get()
            phases = cnfg.CONFIG_GENERATE_TARGET_TANK['phases']
            
            dt = 0.2
            time_grid = np.arange(-20.0, 20.0 + dt, dt)
            
            # Генерация уставки
            goal_pressure_raw = dm.generate_step_profile(
                t_array=time_grid,
                presets=presets,
                duration=duration,
                phases=phases,
                active_phase_calculator=dm.calculate_smooth_strategy
            )
            
            # Локации ключевых точек по времени для отображения маркеров
            x_active_phases = np.linspace(phases['t_start'], phases['t_start'] + duration, len(presets))
            
            # Если у нас уже есть результаты успешного подбора, нарисуем их поверх
            prediction = None
            target = None
            if self.optimized_results is not None:
                prediction = self.optimized_results.get("prediction")
                target = self.optimized_results.get("target")

            # Отрисовка превью
            self.plotter.draw(
                time_grid=time_grid, 
                goal_raw=goal_pressure_raw, 
                prediction=prediction, 
                target=target, 
                presets_t=x_active_phases, 
                presets_y=presets
            )
            
        except Exception as e:
            # Выводим ошибку в консоль разработчика, чтобы не гасить важные предупреждения
            print(f"[UI Preview Error] {e}", file=sys.stderr)

    def start_optimization(self):
        """Запускает фоновый поток расчетов, чтобы GUI не зависал"""
        if self.opt_thread and self.opt_thread.is_alive():
            return
            
        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="Статус: Выполняется расчет...", foreground="orange")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, ">>> Запуск оптимизатора дифференциальной эволюции...\n")
        
        # Перенаправляем stdout принтов в текстовое поле
        self.stdout_orig = sys.stdout
        sys.stdout = StdoutRedirector(self.log_text)
        
        # Получаем данные
        presets = self.get_current_presets()
        duration = self.v_duration.get()
        flat_params = self.tank_model.get_params()
        phases = cnfg.CONFIG_GENERATE_TARGET_TANK['phases']
        
        # Создаем и запускаем поток
        self.opt_thread = threading.Thread(
            target=self._run_optimization_process,
            args=(presets, duration, flat_params, phases),
            daemon=True
        )
        self.opt_thread.start()
        
        # Начинаем опрос очереди результатов
        self.after(100, self._check_queue)

    def _run_optimization_process(self, presets, duration, flat_params, phases):
        """Код, выполняемый в фоновом потоке"""
        try:
            dt = 0.2
            time_grid = np.arange(-20.0, 20.0 + dt, dt)
            
            # Шаг 1: Формируем желаемый сглаженный профиль на основе точек
            goal_pressure_raw = dm.generate_step_profile(
                t_array=time_grid,
                presets=presets,
                duration=duration,
                phases=phases,
                active_phase_calculator=dm.calculate_smooth_strategy
            )
            
            # Шаг 2: Симулируем инерционный профиль бака
            goal_pressure = np.zeros_like(time_grid)
            damping = flat_params['damping']
            k_gain, b_gain = 0.1, 1
            p_current = 0.0
            for idx, t in enumerate(time_grid):
                if t < 0 or t > duration:
                    p_current = prm.predict_pressure(
                        p_prev=p_current, target_p=goal_pressure_raw[idx],
                        dt=dt, damping=damping, k_gain=k_gain, b_gain=b_gain
                    )
                else:
                    p_current = goal_pressure_raw[idx]
                goal_pressure[idx] = p_current

            # Шаг 3: Запуск оптимизации (подбор 10 ступеней)
            best_presets_percent = iopt.optimize_presets(
                time_grid=time_grid,
                goal_pressure=goal_pressure,
                phases=phases,
                flat_params=flat_params,
                eval_window=(0.0, duration),
                injection_duration=duration
            )
            
            # Шаг 4: Расчет обратной связи - финальный прогноз давления по найденным уставкам
            goal_set_plc = np.floor(np.array(best_presets_percent) * 0.6)  # Конвертация % в ПЛК [0..60]
            target_profile = dm.generate_step_profile(
                time_grid, 
                goal_set_plc, 
                duration, 
                phases, 
                dm.calculate_step_strategy
            )
            
            prediction = prm.analitic_model(time_grid, target_profile, flat_params)
            
            # Передаем результаты в основной поток
            self.queue.put({
                "status": "success",
                "time_grid": time_grid,
                "goal_raw": goal_pressure_raw,
                "prediction": prediction,
                "target": target_profile,
                "best_presets": best_presets_percent
            })
            
        except Exception as e:
            self.queue.put({"status": "error", "message": str(e)})

    def _check_queue(self):
        """Опрос очереди результатов в главном потоке GUI"""
        try:
            res = self.queue.get_nowait()
            # Восстанавливаем стандартный stdout
            sys.stdout = self.stdout_orig
            self.btn_run.configure(state="normal")
            
            if res["status"] == "success":
                # Сохраняем расчетное состояние
                self.optimized_results = res
                
                phases = cnfg.CONFIG_GENERATE_TARGET_TANK['phases']
                x_active_phases = np.linspace(phases['t_start'], phases['t_start'] + self.v_duration.get(), len(self.sliders))
                
                # Обновляем график результатами оптимизации
                self.plotter.draw(
                    time_grid=res["time_grid"],
                    goal_raw=res["goal_raw"],
                    prediction=res["prediction"],
                    target=res["target"],
                    presets_t=x_active_phases,
                    presets_y=self.get_current_presets()
                )
                
                self.lbl_status.configure(text="Статус: Подбор успешно завершен", foreground="#2ca02c")
                self.log_text.insert(tk.END, f"\n>>> Подбор успешно завершен!\n")
                self.log_text.insert(tk.END, f"Оптимальные пресеты пульта (%): {res['best_presets']}\n")
                self.log_text.see(tk.END)
            else:
                self.optimized_results = None
                self.lbl_status.configure(text="Статус: Ошибка подбора", foreground="#d62728")
                self.log_text.insert(tk.END, f"\n Ошибка при оптимизации: {res['message']}\n")
                
        except queue.Empty:
            # Если расчет еще идет, проверяем очередь снова через 100 мс
            self.after(100, self._check_queue)