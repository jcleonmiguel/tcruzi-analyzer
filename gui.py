"""
T. cruzi Infection Analyzer — GUI v4
Batch processing + overlay panel + comprehensive CSV report.
Run with: python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from profile_manager import ProfileManager
import threading
import numpy as np
from pathlib import Path
import csv

PIL_available = False
try:
    from PIL import Image, ImageTk
    PIL_available = True
except ImportError:
    pass

# ─── Color palette ────────────────────────────────────────────────────────────
BG       = "#0f1117"
BG2      = "#1a1d27"
BG3      = "#232736"
ACCENT   = "#00e5ff"
ACCENT2  = "#ff4081"
SUCCESS  = "#69ff84"
WARNING  = "#ffd740"
TEXT     = "#e8eaf6"
TEXT_DIM = "#7986cb"
BORDER   = "#2d3152"

FONT_MONO  = ("Courier New", 10)
FONT_TITLE = ("Georgia", 20, "bold")
FONT_HEAD  = ("Georgia", 12, "bold")
FONT_LABEL = ("Helvetica", 10)
FONT_SMALL = ("Helvetica", 9)
FONT_BIG   = ("Helvetica", 26, "bold")


class SectionFrame(tk.LabelFrame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, text=f"  {title}  ",
                         bg=BG2, fg=ACCENT, font=FONT_HEAD,
                         bd=1, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         padx=10, pady=8, **kw)


class Row(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, **kw)

    def label(self, text, width=16):
        tk.Label(self, text=text, bg=BG2, fg=TEXT_DIM,
                 font=FONT_LABEL, anchor="w", width=width).pack(side="left")
        return self

    def spinbox(self, var, from_, to, increment=1, width=7):
        tk.Spinbox(self, textvariable=var, from_=from_, to=to,
                   increment=increment, width=width,
                   bg=BG3, fg=TEXT, insertbackground=TEXT,
                   buttonbackground=BG3, relief="flat",
                   font=FONT_MONO, highlightbackground=BORDER,
                   highlightthickness=1).pack(side="left", padx=(0, 6))
        return self

    def combo(self, var, values, width=11):
        ttk.Combobox(self, textvariable=var, values=values,
                     width=width, state="readonly",
                     font=FONT_LABEL).pack(side="left", padx=(0, 6))
        return self

    def info(self, text):
        tk.Label(self, text=text, bg=BG2, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side="left")
        return self


class StatCard(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=BG3, padx=10, pady=6,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        tk.Label(self, text=title, bg=BG3, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
        self._val = tk.Label(self, text="—", bg=BG3, fg=ACCENT, font=FONT_BIG)
        self._val.pack(anchor="w")

    def set(self, val, color=None):
        self._val.config(text=str(val), fg=color or ACCENT)


class TcruzApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TCruzi Analyzer")
        self.configure(bg=BG)
        self.geometry("1440x880")
        self.minsize(1100, 700)

        # Icono de ventana (funciona en desarrollo y empaquetado con PyInstaller)
        try:
            import sys, os
            if hasattr(sys, "_MEIPASS"):
                _icon_dir = sys._MEIPASS
            else:
                _icon_dir = os.path.dirname(os.path.abspath(__file__))
            _ico = os.path.join(_icon_dir, "tcruzi_icon.ico")
            if os.path.exists(_ico):
                self.iconbitmap(_ico)
        except Exception as _e:
            print(f"[Icono] No se pudo cargar el icono de ventana: {_e}")
        
        # Sistema de perfiles
        self.profile_manager = ProfileManager()
        self._current_profile = tk.StringVar(value=self.profile_manager.last_used)

        self._status   = tk.StringVar(value="Añade imágenes y pulsa Analizar Todo.")
        self._progress = tk.IntVar(value=0)

        # Parameters
        self._blue_ch  = tk.IntVar(value=1)
        self._red_ch   = tk.IntVar(value=0)
        self._hw_mode  = tk.StringVar(value="auto")  # Hardware mode: auto/gpu/cpu
        self._diameter = tk.StringVar(value="auto")
        self._flow_thr = tk.DoubleVar(value=0.4)
        self._cell_thr = tk.DoubleVar(value=-1.0)  # Más permisivo (antes -3.5)
        self._amas_min = tk.IntVar(value=2)
        self._amas_max = tk.IntVar(value=150)
        self._thr_meth = tk.StringVar(value="otsu")
        self._dil_rad  = tk.IntVar(value=100)
        self._min_amas = tk.IntVar(value=1)
        self._avg_amas_size = tk.StringVar(value="35")  # Fallback razonable (antes 15)
        self._amas_color = tk.StringVar(value="red")  # Color amastigotes: red/green

        # Nuevos: canales de amastigotes y modo detección
        self._amas_ch_red   = tk.BooleanVar(value=True)   # Canal rojo activo
        self._amas_ch_green = tk.BooleanVar(value=False)  # Canal verde activo
        self._amas_ch_blue  = tk.BooleanVar(value=False)  # Canal azul activo (≠ núcleos)
        self._amas_ch_custom = tk.BooleanVar(value=False) # Canal personalizado
        self._amas_ch_custom_idx = tk.IntVar(value=3)     # Índice canal extra
        self._amas_detect_mode = tk.StringVar(value="auto")   # auto/cellpose/classic
        self._amas_cp_diameter = tk.StringVar(value="auto")   # Diámetro CellPose amastigotes

        # Batch state
        self._file_list     = []
        self._batch_results = {}
        self._current_file  = None
        self._batch_running = False
        self._view_mode     = tk.StringVar(value="overlay")

        # Auto-limpieza de imágenes antes de analizar (estandariza a rojo)
        self._auto_clean    = tk.BooleanVar(value=True)
        self._auto_clean.trace_add("write", self._update_channel_controls_visibility)

        self._build_ui()
        
        # Cargar el último perfil usado
        self._load_profile(self.profile_manager.last_used)
        self._apply_ttk_style()

        # Ajustar visibilidad inicial de controles de canal según limpieza
        self._update_channel_controls_visibility()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=10)
        hdr.pack(fill="x", padx=18)
        tk.Label(hdr, text="T. CRUZI", bg=BG, fg=ACCENT, font=FONT_TITLE).pack(side="left")
        tk.Label(hdr, text="  Infection Analyzer", bg=BG, fg=TEXT,
                 font=("Georgia", 14)).pack(side="left")
        
        # Indicador de modo dinámico
        self._mode_indicator_text = tk.StringVar(value="🔍 Auto")
        self._mode_indicator_color = tk.StringVar(value=TEXT_DIM)
        self._mode_indicator = tk.Label(hdr, textvariable=self._mode_indicator_text, 
                                        bg=BG, fg=TEXT_DIM,
                                        font=("Helvetica", 10, "bold"))
        self._mode_indicator.pack(side="left", padx=10)
        
        tk.Label(hdr, text="Amastigote Quantification Tool",
                 bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        # Column 1: parameters (CON SCROLL VERTICAL)
        # Contenedor exterior de ancho fijo
        col1_outer = tk.Frame(main, bg=BG, width=288)
        col1_outer.pack(side="left", fill="y", padx=(0, 6))
        col1_outer.pack_propagate(False)

        # Canvas + Scrollbar para permitir desplazamiento
        col1_canvas = tk.Canvas(col1_outer, bg=BG, highlightthickness=0,
                                bd=0, width=270)
        col1_scroll = ttk.Scrollbar(col1_outer, orient="vertical",
                                    command=col1_canvas.yview)
        col1_canvas.configure(yscrollcommand=col1_scroll.set)

        col1_scroll.pack(side="right", fill="y")
        col1_canvas.pack(side="left", fill="both", expand=True)

        # Frame interior donde van TODAS las secciones de parámetros
        col1 = tk.Frame(col1_canvas, bg=BG)
        col1_window = col1_canvas.create_window((0, 0), window=col1, anchor="nw")

        # Ajustar la región de scroll cuando cambie el tamaño del contenido
        def _on_col1_configure(event):
            col1_canvas.configure(scrollregion=col1_canvas.bbox("all"))
        col1.bind("<Configure>", _on_col1_configure)

        # Hacer que el frame interior ocupe el ancho del canvas
        def _on_canvas_configure(event):
            col1_canvas.itemconfig(col1_window, width=event.width)
        col1_canvas.bind("<Configure>", _on_canvas_configure)

        # Scroll con la rueda del ratón cuando el cursor está sobre el panel
        def _on_mousewheel(event):
            # Windows/Mac: event.delta ; Linux: event.num (4 arriba, 5 abajo)
            if hasattr(event, "delta") and event.delta:
                col1_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                col1_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                col1_canvas.yview_scroll(1, "units")

        def _bind_wheel(_e):
            col1_canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows/Mac
            col1_canvas.bind_all("<Button-4>", _on_mousewheel)     # Linux
            col1_canvas.bind_all("<Button-5>", _on_mousewheel)     # Linux

        def _unbind_wheel(_e):
            col1_canvas.unbind_all("<MouseWheel>")
            col1_canvas.unbind_all("<Button-4>")
            col1_canvas.unbind_all("<Button-5>")

        # Solo capturar la rueda cuando el ratón está encima de este panel
        col1_canvas.bind("<Enter>", _bind_wheel)
        col1_canvas.bind("<Leave>", _unbind_wheel)

        self._build_params(col1)

        # Column 2: batch panel
        col2 = tk.Frame(main, bg=BG, width=350)
        col2.pack(side="left", fill="y", padx=(0, 6))
        col2.pack_propagate(False)
        self._build_batch(col2)

        # Column 3: viewer
        col3 = tk.Frame(main, bg=BG)
        col3.pack(side="left", fill="both", expand=True)
        self._build_viewer(col3)

        # Status bar
        bar = tk.Frame(self, bg=BG2, pady=5)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self._status, bg=BG2, fg=TEXT_DIM,
                 font=FONT_SMALL, anchor="w", padx=12).pack(side="left")
        self._pbar = ttk.Progressbar(bar, variable=self._progress,
                                     maximum=100, length=200, mode="determinate")
        self._pbar.pack(side="right", padx=12)

    def _build_params(self, parent):
        # ─── Sección de Perfiles ────────────────────────────────────────────
        profilesec = SectionFrame(parent, "💾  Perfiles de Configuración")
        profilesec.pack(fill="x", pady=(0, 5))
        
        prof_frame = tk.Frame(profilesec, bg=BG2)
        prof_frame.pack(fill="x", padx=5, pady=3)
        
        tk.Label(prof_frame, text="Perfil:", bg=BG2, fg=TEXT, 
                font=("Helvetica", 9)).pack(side="left", padx=(0, 5))
        
        # Combobox de perfiles
        profile_names = self.profile_manager.get_profile_names()
        self._profile_combo = ttk.Combobox(
            prof_frame,
            textvariable=self._current_profile,
            values=profile_names,
            state="readonly",
            width=18
        )
        self._profile_combo.pack(side="left", padx=5)
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_profile_changed)
        
        # Botones de gestión
        tk.Button(prof_frame, text="💾", command=self._save_current_profile,
                 bg=BG3, fg=ACCENT, relief="flat", padx=6, pady=2, cursor="hand2",
                 font=("Helvetica", 10)
                 ).pack(side="left", padx=1)
        
        tk.Button(prof_frame, text="➕", command=self._create_new_profile,
                 bg=BG3, fg=ACCENT, relief="flat", padx=6, pady=2, cursor="hand2",
                 font=("Helvetica", 10)
                 ).pack(side="left", padx=1)
        
        tk.Button(prof_frame, text="✏️", command=self._rename_profile,
                 bg=BG3, fg=ACCENT, relief="flat", padx=6, pady=2, cursor="hand2",
                 font=("Helvetica", 10)
                 ).pack(side="left", padx=1)
        
        tk.Button(prof_frame, text="🗑️", command=self._delete_profile,
                 bg=BG3, fg=WARNING, relief="flat", padx=6, pady=2, cursor="hand2",
                 font=("Helvetica", 10)
                 ).pack(side="left", padx=1)
        
        # Hardware mode selector
        hwsec = SectionFrame(parent, "⚙️  Modo de Procesamiento")
        hwsec.pack(fill="x", pady=(0, 5))
        
        hw_frame = tk.Frame(hwsec, bg=BG2)
        hw_frame.pack(fill="x", padx=5, pady=3)
        
        tk.Label(hw_frame, text="Modo:", bg=BG2, fg=TEXT, 
                font=("Helvetica", 9)).pack(anchor="w", padx=(0, 5))
        
        # Radio buttons en VERTICAL con callback para actualizar indicador
        def update_mode_indicator(*args):
            mode = self._hw_mode.get()
            if mode == "auto":
                self._mode_indicator_text.set("🔍 Auto")
                self._mode_indicator.config(fg=TEXT_DIM)
            elif mode == "gpu":
                self._mode_indicator_text.set("⚡ GPU (NVIDIA)")
                self._mode_indicator.config(fg=SUCCESS)
            elif mode == "cpu":
                self._mode_indicator_text.set("🖥 CPU")
                self._mode_indicator.config(fg=WARNING)
        
        self._hw_mode.trace_add("write", update_mode_indicator)
        
        tk.Radiobutton(hw_frame, text="🔍 Auto", variable=self._hw_mode, value="auto",
                      bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                      activeforeground=ACCENT, font=("Helvetica", 9, "bold"),
                      cursor="hand2").pack(anchor="w", padx=15)
        
        tk.Radiobutton(hw_frame, text="⚡ GPU (NVIDIA)", variable=self._hw_mode, value="gpu",
                      bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                      activeforeground=ACCENT, font=("Helvetica", 9),
                      cursor="hand2").pack(anchor="w", padx=15)
        
        tk.Radiobutton(hw_frame, text="🖥 CPU", variable=self._hw_mode, value="cpu",
                      bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                      activeforeground=ACCENT, font=("Helvetica", 9),
                      cursor="hand2").pack(anchor="w", padx=15)
        
        
        
        nsec = SectionFrame(parent, "🔵  Núcleos")
        nsec.pack(fill="x", pady=(0, 5))
        # "Canal Azul" solo es relevante sin limpieza (con limpieza se fuerza a 2)
        self._blue_ch_row = Row(nsec); self._blue_ch_row.pack(fill="x")
        self._blue_ch_row.label("Canal Azul").spinbox(self._blue_ch, 0, 7)
        r = Row(nsec); r.pack(fill="x", pady=2); r.label("Diámetro")
        tk.Entry(r, textvariable=self._diameter, bg=BG3, fg=TEXT, font=FONT_MONO,
                 width=7, relief="flat", insertbackground=TEXT,
                 highlightbackground=BORDER, highlightthickness=1).pack(side="left")
        r = Row(nsec); r.pack(fill="x", pady=2)
        r.label("Flow thresh.").spinbox(self._flow_thr, 0.0, 1.0, 0.05, 6)
        r = Row(nsec); r.pack(fill="x", pady=2)
        r.label("Cellprob").spinbox(self._cell_thr, -6.0, 6.0, 0.5, 6)

        asec = SectionFrame(parent, "🔴  Amastigotes")
        asec.pack(fill="x", pady=(0, 5))

        # ── Contenedor de canales manuales (se oculta con limpieza activa) ───────
        # Color + canales activos + canal extra solo importan cuando NO se limpia;
        # con limpieza, todo se estandariza a rojo (canal 0) y núcleos a canal 2.
        self._manual_channels_frame = tk.Frame(asec, bg=BG2)
        self._manual_channels_frame.pack(fill="x")
        mcf = self._manual_channels_frame

        # ── Modo color de amastigotes ──────────────────────────────────────────
        tk.Label(mcf, text="Color de amastigotes:", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(4, 0))

        color_frame = tk.Frame(mcf, bg=BG2)
        color_frame.pack(fill="x", padx=8, pady=(0, 4))

        def _on_color_change():
            # Modo rojo → canal 0, modo verde → canal 1
            if self._amas_color.get() == "green":
                self._red_ch.set(1)
            else:
                self._red_ch.set(0)

        tk.Radiobutton(color_frame, text="🔴 Rojo (canal 0)", variable=self._amas_color,
                       value="red", command=_on_color_change, bg=BG2, fg="#FF5555",
                       selectcolor=BG3, activebackground=BG2, font=("Helvetica", 9, "bold"),
                       cursor="hand2").pack(side="left", padx=6)
        tk.Radiobutton(color_frame, text="🟢 Verde (canal 1)", variable=self._amas_color,
                       value="green", command=_on_color_change, bg=BG2, fg="#55FF55",
                       selectcolor=BG3, activebackground=BG2, font=("Helvetica", 9, "bold"),
                       cursor="hand2").pack(side="left", padx=6)

        # ── Canales activos ────────────────────────────────────────────────────
        tk.Label(mcf, text="Canales con amastigotes:", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(4, 0))

        ch_frame = tk.Frame(mcf, bg=BG2)
        ch_frame.pack(fill="x", padx=8, pady=2)

        def _make_ch_cb(parent, text, var, color):
            cb = tk.Checkbutton(parent, text=text, variable=var,
                                bg=BG2, fg=color, selectcolor=BG3,
                                activebackground=BG2, activeforeground=color,
                                font=("Helvetica", 9, "bold"), cursor="hand2")
            cb.pack(side="left", padx=4)
            return cb

        _make_ch_cb(ch_frame, "🔴 Rojo",   self._amas_ch_red,   "#FF5555")
        _make_ch_cb(ch_frame, "🟢 Verde",  self._amas_ch_green, "#55FF55")
        _make_ch_cb(ch_frame, "🔵 Azul",   self._amas_ch_blue,  "#5599FF")

        # Canal personalizado (índice libre)
        custom_frame = tk.Frame(mcf, bg=BG2)
        custom_frame.pack(fill="x", padx=8, pady=(0, 3))
        tk.Checkbutton(custom_frame, text="Canal extra:", variable=self._amas_ch_custom,
                       bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                       font=("Helvetica", 9), cursor="hand2").pack(side="left")
        tk.Spinbox(custom_frame, from_=0, to=7, textvariable=self._amas_ch_custom_idx,
                   width=4, bg=BG3, fg=TEXT, buttonbackground=BG3,
                   font=FONT_MONO, relief="flat").pack(side="left", padx=4)
        tk.Label(custom_frame, text="(índice)", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        # ── Modo de detección ──────────────────────────────────────────────────
        tk.Label(asec, text="Modo de detección:", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(6, 0))

        mode_frame = tk.Frame(asec, bg=BG2)
        mode_frame.pack(fill="x", padx=8, pady=(0, 4))

        mode_info = {
            "auto":     ("🔍 Auto",    TEXT,    "CellPose si señal dispersa, clásico si densa"),
            "cellpose": ("⚡ CellPose", SUCCESS, "CellPose siempre (más preciso en rangos bajos)"),
            "classic":  ("🧮 Clásico",  TEXT_DIM, "Método umbral clásico (más rápido)"),
        }
        self._mode_desc_var = tk.StringVar(value="CellPose si señal dispersa, clásico si densa")

        def _on_mode_change(*args):
            m = self._amas_detect_mode.get()
            self._mode_desc_var.set(mode_info[m][2])
            # Mostrar/ocultar diámetro CellPose solo si es relevante
            if m in ("cellpose", "auto"):
                cp_diam_frame.pack(fill="x", padx=8, pady=2)
            else:
                cp_diam_frame.pack_forget()

        self._amas_detect_mode.trace_add("write", _on_mode_change)

        for val, (label, color, _) in mode_info.items():
            tk.Radiobutton(mode_frame, text=label, variable=self._amas_detect_mode,
                           value=val, bg=BG2, fg=color, selectcolor=BG3,
                           activebackground=BG2, font=("Helvetica", 9, "bold"),
                           cursor="hand2", anchor="w").pack(side="top", anchor="w", padx=6, pady=1)

        tk.Label(asec, textvariable=self._mode_desc_var, bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 7, "italic")).pack(anchor="w", padx=12, pady=(0, 2))

        # Diámetro CellPose para amastigotes
        cp_diam_frame = tk.Frame(asec, bg=BG2)
        r = Row(cp_diam_frame); r.pack(fill="x")
        r.label("Diám. CellPose px")
        tk.Entry(r, textvariable=self._amas_cp_diameter, bg=BG3, fg=TEXT,
                 font=FONT_MONO, width=7, relief="flat", insertbackground=TEXT,
                 highlightbackground=BORDER, highlightthickness=1).pack(side="left")
        tk.Label(r, text="(auto=estimado)", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 7)).pack(side="left", padx=4)
        cp_diam_frame.pack(fill="x", padx=8, pady=2)  # visible por defecto (modo auto)

        # ── Parámetros clásicos ────────────────────────────────────────────────
        tk.Label(asec, text="Parámetros umbral (modo clásico/auto):", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(4, 0))
        r = Row(asec); r.pack(fill="x", pady=2)
        r.label("Tam. mín px²").spinbox(self._amas_min, 1, 500)
        r = Row(asec); r.pack(fill="x", pady=2)
        r.label("Tam. máx px²").spinbox(self._amas_max, 10, 2000, 10)
        r = Row(asec); r.pack(fill="x", pady=2)
        r.label("Umbral").combo(self._thr_meth, ["otsu", "triangle", "yen"])

        isec = SectionFrame(parent, "🦠  Infección")
        isec.pack(fill="x", pady=(0, 5))
        r = Row(isec); r.pack(fill="x")
        r.label("Radio citosol").spinbox(self._dil_rad, 5, 300, 5)
        r = Row(isec); r.pack(fill="x", pady=2)
        r.label("Mín. amas.").spinbox(self._min_amas, 1, 10)
        r = Row(isec); r.pack(fill="x", pady=2)
        r.label("Tam. medio px²")
        
        # Frame para Entry + botón Auto
        size_frame = tk.Frame(r, bg=BG2)
        size_frame.pack(side="left", fill="x", expand=True)
        
        # Entry para valor numérico o "auto"
        self._avg_amas_entry = tk.Entry(
            size_frame, 
            textvariable=self._avg_amas_size, 
            bg=BG3, 
            fg=TEXT, 
            font=FONT_MONO,
            width=7, 
            relief="flat", 
            insertbackground=TEXT,
            highlightbackground=BORDER, 
            highlightthickness=1
        )
        self._avg_amas_entry.pack(side="left", padx=(0, 5))
        
        # Botón "Auto"
        tk.Button(
            size_frame,
            text="Auto",
            command=self._set_auto_amas_size,
            bg=BG3,
            fg=ACCENT,
            font=("Helvetica", 8, "bold"),
            relief="flat",
            padx=5,
            pady=2,
            cursor="hand2"
        ).pack(side="left")
        r.info("(para estimar densos)")

        leg = SectionFrame(parent, "🎨  Leyenda")
        leg.pack(fill="x")
        for sym, col, lbl in [("■", SUCCESS, "Núcleo sano"),
                               ("■", WARNING, "Núcleo infectado"),
                               ("■", "#ff00ff", "Amastigote"),
                               ("■", "#ff8c00", "Masa densa (72hpi+)")]:
            row = tk.Frame(leg, bg=BG2); row.pack(anchor="w")
            tk.Label(row, text=sym, fg=col, bg=BG2, font=("Helvetica", 12)).pack(side="left")
            tk.Label(row, text=f"  {lbl}", fg=TEXT_DIM, bg=BG2, font=FONT_SMALL).pack(side="left")
        
        # Nota sobre rangos
        note = tk.Label(leg, text="Rangos densos: ≤15 | 15-50 | >50 amastigotes",
                       bg=BG2, fg=TEXT_DIM, font=("Helvetica", 8, "italic"),
                       anchor="w", wraplength=250)
        note.pack(fill="x", pady=(4, 0))

    def _build_batch(self, parent):
        # File list
        fsec = SectionFrame(parent, "📂  Cola de Imágenes")
        fsec.pack(fill="x", pady=(0, 5))

        brow = tk.Frame(fsec, bg=BG2); brow.pack(fill="x")
        tk.Button(brow, text="+ Añadir", command=self._add_files,
                  bg=ACCENT, fg=BG, font=("Helvetica", 9, "bold"),
                  relief="flat", padx=8, cursor="hand2").pack(side="left")
        tk.Button(brow, text="✕ Limpiar", command=self._clear_files,
                  bg=BG3, fg=TEXT_DIM, font=("Helvetica", 9),
                  relief="flat", padx=6, cursor="hand2").pack(side="left", padx=4)
        self._count_lbl = tk.Label(brow, text="0 imágenes", bg=BG2,
                                    fg=TEXT_DIM, font=FONT_SMALL)
        self._count_lbl.pack(side="right")

        # Auto-limpieza: estandariza cada imagen (amastigotes→rojo) antes de analizar
        clean_row = tk.Frame(fsec, bg=BG2); clean_row.pack(fill="x", pady=(4, 0))
        clean_cb = tk.Checkbutton(
            clean_row,
            text="🧹 Limpiar y estandarizar imágenes antes de analizar",
            variable=self._auto_clean,
            bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
            activeforeground=ACCENT, font=FONT_SMALL, cursor="hand2",
            anchor="w", highlightthickness=0, bd=0)
        clean_cb.pack(side="left")
        tk.Label(clean_row,
                 text="(fondo negro + amastigotes a rojo, rojos o verdes)",
                 bg=BG2, fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="left", padx=(4, 0))

        lf = tk.Frame(fsec, bg=BG2); lf.pack(fill="x", pady=(5, 0))
        self._file_lb = tk.Listbox(lf, bg=BG3, fg=TEXT_DIM, font=FONT_SMALL,
                                    height=6, selectbackground=BG2,
                                    selectforeground=ACCENT, relief="flat",
                                    highlightthickness=0, activestyle="none")
        lsb = ttk.Scrollbar(lf, orient="vertical", command=self._file_lb.yview)
        self._file_lb.configure(yscrollcommand=lsb.set)
        self._file_lb.pack(side="left", fill="x", expand=True)
        lsb.pack(side="right", fill="y")
        self._file_lb.bind("<<ListboxSelect>>", self._on_lb_select)

        # Aggregate stats
        ssec = SectionFrame(parent, "")  # Título vacío, lo ponemos dinámicamente
        title_frame = tk.Frame(ssec, bg=BG2)
        title_frame.pack(anchor="w", fill="x", padx=8, pady=(0, 5))
        
        self._totals_title_label = tk.Label(title_frame, text="📊  Totales (todas las imágenes)", 
                                            bg=BG2, fg=TEXT, font=("Helvetica", 11, "bold"))
        self._totals_title_label.pack(side="left")
        
        # Botón para volver a totales globales
        self._global_totals_btn = tk.Button(title_frame, text="[Ver globales]", 
                                            command=lambda: self._update_totals(specific_image=None),
                                            bg=BG2, fg=TEXT_DIM, font=("Helvetica", 9),
                                            relief="flat", cursor="hand2", padx=5)
        self._global_totals_btn.pack(side="left", padx=(10, 0))
        # Inicialmente oculto (solo se muestra cuando hay una imagen seleccionada)
        self._global_totals_btn.pack_forget()
        
        ssec.pack(fill="x", pady=(0, 5))
        sg = tk.Frame(ssec, bg=BG2); sg.pack(fill="x")
        self._card_total    = StatCard(sg, "CÉLULAS")
        self._card_infected = StatCard(sg, "INFECTADAS")
        self._card_pct      = StatCard(sg, "% INFECCIÓN")
        self._card_amas     = StatCard(sg, "AMASTIGOTES")
        for i, c in enumerate([self._card_total, self._card_infected,
                                self._card_pct, self._card_amas]):
            c.grid(row=i//2, column=i%2, sticky="ew", padx=2, pady=2)
        sg.columnconfigure(0, weight=1); sg.columnconfigure(1, weight=1)

        # Per-image results table
        rsec = SectionFrame(parent, "📋  Por Imagen")
        rsec.pack(fill="both", expand=True, pady=(0, 5))
        cols = ("imagen", "total", "inf", "pct", "amas")
        self._btree = ttk.Treeview(rsec, columns=cols, show="headings", height=7)
        for c, h, w in [("imagen","Imagen",130),("total","Células",58),
                         ("inf","Infect.",58),("pct","%",50),("amas","Amas.",50)]:
            self._btree.heading(c, text=h)
            self._btree.column(c, width=w, anchor="center")
        btsb = ttk.Scrollbar(rsec, orient="vertical", command=self._btree.yview)
        self._btree.configure(yscrollcommand=btsb.set)
        self._btree.pack(side="left", fill="both", expand=True)
        btsb.pack(side="right", fill="y")
        self._btree.bind("<<TreeviewSelect>>", self._on_btree_select)

        # Buttons
        self._run_btn = tk.Button(parent, text="▶  ANALIZAR TODO",
                                   command=self._run_batch,
                                   bg=ACCENT2, fg="white",
                                   font=("Helvetica", 12, "bold"),
                                   relief="flat", pady=8, cursor="hand2")
        self._run_btn.pack(fill="x", pady=(2, 3))
        self._exp_btn = tk.Button(parent, text="💾  Exportar Informe Excel",
                                   command=self._export_csv,
                                   bg=BG3, fg=ACCENT, font=("Helvetica", 10),
                                   relief="flat", pady=5, cursor="hand2",
                                   state="disabled")
        self._exp_btn.pack(fill="x")

    def _build_viewer(self, parent):
        tab_row = tk.Frame(parent, bg=BG); tab_row.pack(fill="x", pady=(0, 5))
        tk.Label(tab_row, text="Vista:", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        for mode, lbl in [("original","Original"),("overlay","Overlay"),("blue","Azul"),
                           ("red","Rojo"),("nuclei","Segmentación")]:
            tk.Radiobutton(tab_row, text=lbl, variable=self._view_mode,
                           value=mode, command=self._refresh_view,
                           bg=BG, fg=TEXT_DIM, selectcolor=BG2,
                           activebackground=BG, font=FONT_SMALL,
                           indicatoron=False, relief="flat",
                           padx=9, pady=3, cursor="hand2",
                           highlightthickness=0).pack(side="left", padx=(0,2))

        cf = tk.Frame(parent, bg=BG3, highlightbackground=BORDER, highlightthickness=1)
        cf.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(cf, bg=BG3, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda e: self._refresh_view())

        # Per-cell table
        tf = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        tf.pack(fill="x", pady=(5, 0))
        cols = ("id","y","x","amas","inf")
        self._ctree = ttk.Treeview(tf, columns=cols, show="headings", height=4)
        for c, h, w in [("id","ID",45),("y","Y px",65),("x","X px",65),
                         ("amas","Amastig.",80),("inf","Infectada",80)]:
            self._ctree.heading(c, text=h)
            self._ctree.column(c, width=w, anchor="center")
        csb = ttk.Scrollbar(tf, orient="vertical", command=self._ctree.yview)
        self._ctree.configure(yscrollcommand=csb.set)
        self._ctree.pack(side="left", fill="x", expand=True)
        csb.pack(side="right", fill="y")

    # ── File management ───────────────────────────────────────────────────────
    def _set_auto_amas_size(self):
        """Establece el tamaño medio en 'auto' para calibración automática"""
        self._avg_amas_size.set("auto")
        self._status.set("💡 Tamaño medio: AUTO (se calculará automáticamente de cada imagen)")

    def _update_channel_controls_visibility(self, *_):
        """
        Muestra u oculta los controles de canal manual según el checkbox
        de auto-limpieza. Con limpieza activa, esos controles no hacen nada
        (todo se estandariza a rojo/canal 2), así que se ocultan para no
        sobrecargar la interfaz. Al desactivar la limpieza, reaparecen.
        """
        if not hasattr(self, "_manual_channels_frame"):
            return  # aún no construida la UI

        if self._auto_clean.get():
            self._manual_channels_frame.pack_forget()
            self._blue_ch_row.pack_forget()
        else:
            # Reaparecen al principio de su sección (antes del primer hermano restante)
            nsec_siblings = self._blue_ch_row.master.pack_slaves()
            self._blue_ch_row.pack(fill="x",
                                   before=nsec_siblings[0] if nsec_siblings else None)
            asec_siblings = self._manual_channels_frame.master.pack_slaves()
            self._manual_channels_frame.pack(fill="x",
                                             before=asec_siblings[0] if asec_siblings else None)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE GESTIÓN DE PERFILES
    # ══════════════════════════════════════════════════════════════════════════
    
    def _get_current_config(self):
        """Obtiene la configuración actual de la GUI"""
        avg_size = self._avg_amas_size.get()
        try:
            avg_size_int = int(avg_size) if avg_size != "auto" else 0
        except:
            avg_size_int = 35

        # Construir lista de canales de amastigotes activos
        amas_channels = []
        if self._amas_ch_red.get():   amas_channels.append(self._red_ch.get())
        if self._amas_ch_green.get(): amas_channels.append(1)  # verde = canal 1 típicamente
        if self._amas_ch_blue.get():  amas_channels.append(self._blue_ch.get())
        if self._amas_ch_custom.get(): amas_channels.append(self._amas_ch_custom_idx.get())
        if not amas_channels:  # fallback si nada marcado
            amas_channels = [self._red_ch.get()]

        cp_diam_raw = self._amas_cp_diameter.get().strip()
        try:
            cp_diam = int(cp_diam_raw) if cp_diam_raw.lower() != "auto" else None
        except:
            cp_diam = None

        return {
            "blue_ch": self._blue_ch.get(),
            "red_ch": self._red_ch.get(),
            "amas_color": self._amas_color.get(),
            "amas_channel_indices": amas_channels,
            "amas_ch_red":   self._amas_ch_red.get(),
            "amas_ch_green": self._amas_ch_green.get(),
            "amas_ch_blue":  self._amas_ch_blue.get(),
            "amas_ch_custom": self._amas_ch_custom.get(),
            "amas_ch_custom_idx": self._amas_ch_custom_idx.get(),
            "amas_detection_mode": self._amas_detect_mode.get(),
            "amas_cellpose_diameter": cp_diam,
            "diameter": self._diameter.get(),
            "flow_threshold": self._flow_thr.get(),
            "cellprob_threshold": self._cell_thr.get(),
            "amas_min_size": self._amas_min.get(),
            "amas_max_size": self._amas_max.get(),
            "threshold_method": self._thr_meth.get(),
            "dilation_radius": self._dil_rad.get(),
            "min_amastigotes": self._min_amas.get(),
            "avg_amastigote_size": avg_size_int,
            "use_auto_calibration": (avg_size == "auto"),
            "hardware_mode": self._hw_mode.get()
        }
    
    def _load_profile(self, name):
        """Carga un perfil en la GUI"""
        profile = self.profile_manager.get_profile(name)

        self._blue_ch.set(profile.get("blue_ch", 1))
        self._red_ch.set(profile.get("red_ch", 0))
        self._amas_color.set(profile.get("amas_color", "red"))
        self._diameter.set(profile.get("diameter", "auto"))
        self._flow_thr.set(profile.get("flow_threshold", 0.4))
        self._cell_thr.set(profile.get("cellprob_threshold", -1.0))
        self._amas_min.set(profile.get("amas_min_size", 2))
        self._amas_max.set(profile.get("amas_max_size", 150))
        self._thr_meth.set(profile.get("threshold_method", "otsu"))
        self._dil_rad.set(profile.get("dilation_radius", 100))
        self._min_amas.set(profile.get("min_amastigotes", 1))

        if profile.get("use_auto_calibration", True):
            self._avg_amas_size.set("auto")
        else:
            self._avg_amas_size.set(str(profile.get("avg_amastigote_size", 35)))

        self._hw_mode.set(profile.get("hardware_mode", "auto"))

        # Nuevos: canales y modo detección
        self._amas_ch_red.set(profile.get("amas_ch_red", True))
        self._amas_ch_green.set(profile.get("amas_ch_green", False))
        self._amas_ch_blue.set(profile.get("amas_ch_blue", False))
        self._amas_ch_custom.set(profile.get("amas_ch_custom", False))
        self._amas_ch_custom_idx.set(profile.get("amas_ch_custom_idx", 3))
        self._amas_detect_mode.set(profile.get("amas_detection_mode", "auto"))
        cp_diam = profile.get("amas_cellpose_diameter", None)
        self._amas_cp_diameter.set("auto" if cp_diam is None else str(cp_diam))

        self._status.set(f"✓ Perfil '{name}' cargado")
    
    def _on_profile_changed(self, event=None):
        """Cuando cambia el perfil seleccionado"""
        name = self._current_profile.get()
        self._load_profile(name)
        self.profile_manager.set_last_used(name)
    
    def _save_current_profile(self):
        """Guarda el perfil actual"""
        name = self._current_profile.get()
        config = self._get_current_config()
        
        if self.profile_manager.save_profile(name, config):
            self._status.set(f"✓ Perfil '{name}' guardado")
            messagebox.showinfo("Guardado", f"Perfil '{name}' guardado correctamente")
    
    def _create_new_profile(self):
        """Crea un perfil nuevo"""
        name = simpledialog.askstring(
            "Nuevo Perfil",
            "Nombre del nuevo perfil:",
            parent=self
        )
        
        if name:
            config = self._get_current_config()
            if self.profile_manager.save_profile(name, config):
                self._profile_combo['values'] = self.profile_manager.get_profile_names()
                self._current_profile.set(name)
                self.profile_manager.set_last_used(name)
                self._status.set(f"✓ Perfil '{name}' creado")
    
    def _rename_profile(self):
        """Renombra el perfil actual"""
        old_name = self._current_profile.get()
        
        if self.profile_manager.is_protected(old_name):
            return
        
        new_name = simpledialog.askstring(
            "Renombrar Perfil",
            f"Nuevo nombre para '{old_name}':",
            parent=self,
            initialvalue=old_name
        )
        
        if new_name and self.profile_manager.rename_profile(old_name, new_name):
            self._profile_combo['values'] = self.profile_manager.get_profile_names()
            self._current_profile.set(new_name)
            self._status.set(f"✓ Perfil renombrado a '{new_name}'")
    
    def _delete_profile(self):
        """Elimina el perfil actual"""
        name = self._current_profile.get()
        
        if self.profile_manager.delete_profile(name):
            self._profile_combo['values'] = self.profile_manager.get_profile_names()
            self._current_profile.set("Default")
            self._load_profile("Default")
            self._status.set(f"✓ Perfil '{name}' eliminado")
    
    # ══════════════════════════════════════════════════════════════════════════
    
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Seleccionar imágenes",
            filetypes=[("Microscopía", "*.nd2 *.tif *.tiff"), ("Todos", "*.*")]
        )
        added = 0
        for p in paths:
            if p not in self._file_list:
                self._file_list.append(p)
                self._file_lb.insert("end", Path(p).name)
                added += 1
        self._count_lbl.config(text=f"{len(self._file_list)} imágenes")
        if added:
            self._status.set(f"{added} imagen(es) añadida(s). Total: {len(self._file_list)}.")

    def _clear_files(self):
        self._file_list.clear()
        self._batch_results.clear()
        self._current_file = None
        self._file_lb.delete(0, "end")
        self._count_lbl.config(text="0 imágenes")
        for t in [self._btree, self._ctree]:
            for row in t.get_children(): t.delete(row)
        self._canvas.delete("all")
        for c in [self._card_total, self._card_infected, self._card_pct, self._card_amas]:
            c.set("—")
        self._status.set("Lista limpiada.")

    def _on_lb_select(self, _):
        sel = self._file_lb.curselection()
        if sel and self._file_list[sel[0]] in self._batch_results:
            self._switch_to(self._file_list[sel[0]])

    def _on_btree_select(self, _):
        sel = self._btree.selection()
        if not sel: return
        idx = self._btree.index(sel[0])
        if idx < len(self._file_list):
            path = self._file_list[idx]
            if path in self._batch_results:
                self._switch_to(path)
                self._file_lb.selection_clear(0, "end")
                self._file_lb.selection_set(idx)

    def _switch_to(self, path):
        self._current_file = path
        data = self._batch_results[path]
        is_dense = data.get("is_dense", False)
        # Fill cell table
        for row in self._ctree.get_children(): self._ctree.delete(row)
        for cell in data["results"]["per_cell"]:
            tag = "inf" if cell["infected"] else "ok"
            
            # Mostrar siempre el display correcto según el modo
            amas_display = cell["amastigotes_nearby"]
            
            self._ctree.insert("", "end", values=(
                cell["nucleus_id"],
                f"{cell['centroid_y']:.0f}", f"{cell['centroid_x']:.0f}",
                amas_display,
                "✓ SÍ" if cell["infected"] else "✗ NO"
            ), tags=(tag,))
        self._ctree.tag_configure("inf", foreground=ACCENT2)
        self._ctree.tag_configure("ok",  foreground=SUCCESS)

        # Actualizar totales para mostrar los de esta imagen específica
        self._update_totals(specific_image=path)

        # Show dense mode badge on canvas
        if is_dense:
            self.after(100, lambda: self._canvas.create_text(
                self._canvas.winfo_width() - 10, 10, anchor="ne",
                text="⚠ MODO DENSO (72hpi+)",
                fill=WARNING, font=("Helvetica", 10, "bold")
            ))
        self._refresh_view()

    # ── Batch ─────────────────────────────────────────────────────────────────
    def _run_batch(self):
        if not self._file_list:
            messagebox.showwarning("Sin imágenes", "Añade imágenes primero.")
            return
        if self._batch_running: return

        self._batch_running = True
        self._run_btn.config(state="disabled", text="Analizando...")
        self._exp_btn.config(state="disabled")
        self._progress.set(0)
        # Clear previous results
        self._batch_results.clear()
        self._current_file = None
        for t in [self._btree, self._ctree]:
            for row in t.get_children(): t.delete(row)

        diam_str = self._diameter.get().strip()
        diameter = None
        if diam_str.lower() != "auto":
            try: diameter = float(diam_str)
            except ValueError: pass
        
        # Manejar valor "auto" para tamaño medio
        avg_size_value = self._avg_amas_size.get()
        if isinstance(avg_size_value, str) and avg_size_value.lower() == "auto":
            avg_amas_param = 0  # 0 activa la calibración automática en el analyzer
        else:
            try:
                avg_amas_param = int(avg_size_value)
            except (ValueError, TypeError):
                avg_amas_param = 15  # Fallback a valor por defecto

        # ¿Limpiar antes de analizar? Lee el estado del checkbox (en hilo UI).
        auto_clean = self._auto_clean.get()

        # Construir lista de canales activos igual que en _get_current_config
        amas_channels = []
        if self._amas_ch_red.get():    amas_channels.append(self._red_ch.get())
        if self._amas_ch_green.get():  amas_channels.append(1)
        if self._amas_ch_blue.get():   amas_channels.append(self._blue_ch.get())
        if self._amas_ch_custom.get(): amas_channels.append(self._amas_ch_custom_idx.get())
        if not amas_channels:
            amas_channels = [self._red_ch.get()]

        cp_diam_raw = self._amas_cp_diameter.get().strip()
        try:
            cp_diam = int(cp_diam_raw) if cp_diam_raw.lower() != "auto" else None
        except:
            cp_diam = None

        params = dict(
            blue_ch_idx=self._blue_ch.get(), red_ch_idx=self._red_ch.get(),
            amas_channel_indices=amas_channels,
            amas_detection_mode=self._amas_detect_mode.get(),
            amas_cellpose_diameter=cp_diam,
            nucleus_diameter=diameter, flow_threshold=self._flow_thr.get(),
            cellprob_threshold=self._cell_thr.get(),
            amas_min_size=self._amas_min.get(), amas_max_size=self._amas_max.get(),
            threshold_method=self._thr_meth.get(),
            dilation_radius=self._dil_rad.get(), min_amastigotes=self._min_amas.get(),
            avg_amastigote_size=avg_amas_param,
            hardware_mode=self._hw_mode.get(),
            amas_color=self._amas_color.get(),
        )

        # Si limpiamos, las imágenes salen ESTANDARIZADAS:
        #   núcleos → canal azul (2), amastigotes → canal rojo (0).
        # Forzamos el análisis a esa convención, independientemente de
        # cómo estuvieran configurados los canales para el original.
        if auto_clean:
            params["blue_ch_idx"] = 2
            params["red_ch_idx"] = 0
            params["amas_channel_indices"] = [0]
            params["amas_color"] = "red"

        def worker():
            from analyzer import run_analysis
            import tempfile, shutil
            n = len(self._file_list)

            tmp_dir = None
            if auto_clean:
                tmp_dir = tempfile.mkdtemp(prefix="tcruzi_clean_")
                try:
                    from limpiador_imagenes import clean_image_to_path
                except Exception as e:
                    self.after(0, lambda err=str(e): self._status.set(
                        f"⚠ No se pudo cargar el limpiador: {err}"))
                    clean_image_to_path = None

            try:
                for i, path in enumerate(self._file_list):
                    name = Path(path).name
                    self.after(0, lambda p=int(i/n*100): self._progress.set(p))

                    analyze_path = path
                    # ── Paso 1: limpiar (si está activado) ──────────────────
                    if auto_clean and clean_image_to_path is not None:
                        self.after(0, lambda m=f"[{i+1}/{n}] 🧹 Limpiando {name}":
                                   self._status.set(m))
                        try:
                            stem = Path(path).stem
                            tmp_out = str(Path(tmp_dir) / f"{stem}_clean.tif")
                            info = clean_image_to_path(path, tmp_out, verbose=False)
                            if info is not None:
                                analyze_path = tmp_out
                            else:
                                # formato no soportado por el limpiador → usar original
                                self.after(0, lambda nm=name: self._status.set(
                                    f"⚠ {nm}: no se pudo limpiar, se analiza el original"))
                        except Exception as e:
                            self.after(0, lambda nm=name, err=str(e): self._status.set(
                                f"⚠ {nm}: error al limpiar ({err}), se analiza el original"))

                    # ── Paso 2: analizar ────────────────────────────────────
                    self.after(0, lambda m=f"[{i+1}/{n}] Analizando {name}":
                               self._status.set(m))
                    try:
                        data = run_analysis(filepath=analyze_path, **params)
                        # Indexar SIEMPRE bajo la ruta original (lista/tabla/visor)
                        self._batch_results[path] = data
                        self.after(0, lambda d=data, p=path: self._img_done(d, p))
                    except Exception as e:
                        self.after(0, lambda p=path, err=str(e): self._img_error(p, err))
            finally:
                # Borrar temporales pase lo que pase
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            self.after(0, self._batch_done)

        threading.Thread(target=worker, daemon=True).start()

    def _img_done(self, data, path):
        r = data["results"]
        is_dense = data.get("is_dense", False)
        
        # Calcular total real de amastigotes
        total_amas = sum(c['amastigotes_count'] for c in r['per_cell'] if c['infected'])
        
        # Display: mostrar estimación si hay densos
        if is_dense:
            amas_str = f"~{total_amas}"
        else:
            amas_str = str(total_amas)
        
        pct = r["infection_pct"]
        name = Path(path).name
        tag = "hi" if pct >= 50 else "mid" if pct >= 20 else "lo"
        self._btree.insert("", "end", values=(
            name[:20]+"…" if len(name)>20 else name,
            r["total_cells"], r["infected_cells"], f"{pct}%", amas_str
        ), tags=(tag,))
        self._btree.tag_configure("hi",  foreground=ACCENT2)
        self._btree.tag_configure("mid", foreground=WARNING)
        self._btree.tag_configure("lo",  foreground=SUCCESS)

        idx = self._file_list.index(path)
        self._file_lb.itemconfig(idx, fg=SUCCESS)

        if self._current_file is None:
            self._switch_to(path)

        self._update_totals()

    def _img_error(self, path, err):
        idx = self._file_list.index(path)
        self._file_lb.itemconfig(idx, fg=ACCENT2)
        name = Path(path).name
        self._btree.insert("", "end",
                           values=(name[:20], "ERROR", "—", "—", "—"),
                           tags=("err",))
        self._btree.tag_configure("err", foreground=ACCENT2)
        
        # MOSTRAR ERROR COMPLETO EN CONSOLA
        print(f"\n{'='*70}")
        print(f"ERROR al analizar: {name}")
        print(f"{'='*70}")
        print(f"Ruta: {path}")
        print(f"Error: {err}")
        print(f"{'='*70}\n")

    def _batch_done(self):
        self._batch_running = False
        self._run_btn.config(state="normal", text="▶  ANALIZAR TODO")
        self._exp_btn.config(state="normal")
        self._progress.set(100)
        self._status.set(f"✓ Completado — {len(self._batch_results)} imágenes analizadas.")

    def _update_totals(self, specific_image=None):
        """Actualiza los totales. Si specific_image es None, muestra totales globales.
        Si se proporciona, muestra totales de esa imagen específica."""
        
        if specific_image and specific_image in self._batch_results:
            # Mostrar totales de una imagen específica
            d = self._batch_results[specific_image]
            r = d["results"]
            tc = r["total_cells"]
            ti = r["infected_cells"]
            
            # Contar amastigotes de esta imagen
            ta = sum(cell["amastigotes_count"] for cell in r["per_cell"] if cell["infected"])
            has_dense = d.get("is_dense", False)
            
            pct = round(ti/tc*100, 1) if tc else 0
            
            # Actualizar título para mostrar el nombre de la imagen
            img_name = Path(specific_image).name
            self._totals_title_label.config(text=f"📊  Totales ({img_name})")
            
            # Mostrar botón para volver a totales globales
            self._global_totals_btn.pack(side="left", padx=(10, 0))
        else:
            # Mostrar totales globales de todas las imágenes
            tc = sum(d["results"]["total_cells"]    for d in self._batch_results.values())
            ti = sum(d["results"]["infected_cells"] for d in self._batch_results.values())
            
            # Sumar conteo real de todas las células infectadas
            ta = 0
            has_dense = False
            for d in self._batch_results.values():
                for cell in d["results"]["per_cell"]:
                    if cell["infected"]:
                        ta += cell["amastigotes_count"]
                if d.get("is_dense"):
                    has_dense = True
            
            pct = round(ti/tc*100, 1) if tc else 0
            
            # Actualizar título para mostrar "todas las imágenes"
            self._totals_title_label.config(text="📊  Totales (todas las imágenes)")
            
            # Ocultar botón de totales globales (ya estamos en globales)
            self._global_totals_btn.pack_forget()
        
        # Actualizar las tarjetas
        self._card_total.set(tc)
        self._card_infected.set(ti, ACCENT2)
        self._card_pct.set(f"{pct}%", SUCCESS if pct<20 else WARNING if pct<50 else ACCENT2)
        
        # Mostrar con ~ si hay estimaciones
        amas_display = f"~{ta}" if has_dense else ta
        self._card_amas.set(amas_display)

    # ── Viewer ────────────────────────────────────────────────────────────────
    def _refresh_view(self, *_):
        if not PIL_available or not self._current_file: return
        data = self._batch_results.get(self._current_file)
        if not data: return

        mode = self._view_mode.get()
        if mode == "original":
            # Imagen original: rojo + azul sin anotaciones
            blue_ch = data["blue_channel"]
            red_ch = data["red_channel"]
            arr = np.stack([red_ch, np.zeros_like(blue_ch), blue_ch], axis=2)
        elif mode == "overlay":
            arr = data["overlay"]
        elif mode == "blue":
            ch = data["blue_channel"]
            arr = np.stack([np.zeros_like(ch)]*2 + [ch], axis=2)
        elif mode == "red":
            ch = data["red_channel"]
            arr = np.stack([ch] + [np.zeros_like(ch)]*2, axis=2)
        elif mode == "nuclei":
            from skimage.color import label2rgb
            arr = (label2rgb(data["nucleus_mask"], bg_label=0)*255).astype(np.uint8)
        else:
            arr = data["overlay"]

        cw = max(self._canvas.winfo_width(), 100)
        ch_ = max(self._canvas.winfo_height(), 100)
        ih, iw = arr.shape[:2]
        scale = min(cw/iw, ch_/ih)
        nw, nh = int(iw*scale), int(ih*scale)

        img = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch_//2, anchor="center", image=self._tk_img)
        self._canvas.create_text(8, 8, anchor="nw",
                                  text=Path(self._current_file).name,
                                  fill=TEXT_DIM, font=FONT_SMALL)

    # ── CSV Export ────────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._batch_results: return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="tcruzi_informe.xlsx"
        )
        if not path: return

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.drawing.image import Image as XLImage
            from openpyxl.utils import get_column_letter
            from skimage.measure import regionprops
            from PIL import Image as PILImage
            import io, tempfile, os, shutil

            wb = Workbook()
            wb.remove(wb.active)

            HDR_FILL  = PatternFill("solid", start_color="1A1D27")
            HDR_FONT  = Font(bold=True, color="00E5FF", name="Arial", size=10)
            INF_FILL  = PatternFill("solid", start_color="3D1020")
            INF_FONT  = Font(color="FF4081", name="Arial", size=10)
            OK_FONT   = Font(color="2E7D32", name="Arial", size=10)
            NORM_FONT = Font(name="Arial", size=10)
            thin      = Side(style="thin", color="2D3152")
            brd       = Border(left=thin, right=thin, top=thin, bottom=thin)

            tmp_dir = tempfile.mkdtemp()

            # ── Hoja Resumen ──────────────────────────────────────────────────
            ws_sum = wb.create_sheet("Resumen")
            ws_sum.sheet_view.showGridLines = False
            ws_sum["A1"] = "T. CRUZI - Informe de Infeccion"
            ws_sum["A1"].font = Font(bold=True, color="00E5FF", name="Arial", size=14)
            ws_sum.merge_cells("A1:E1")
            ws_sum["A1"].alignment = Alignment(horizontal="center")
            ws_sum.column_dimensions["A"].width = 32
            for col in ["B","C","D","E"]:
                ws_sum.column_dimensions[col].width = 18

            for ci, h in enumerate(["Imagen","Celulas Totales","Infectadas","% Infeccion","Amastigotes"], 1):
                c = ws_sum.cell(row=3, column=ci, value=h)
                c.font = HDR_FONT; c.fill = HDR_FILL
                c.alignment = Alignment(horizontal="center"); c.border = brd

            row = 4
            total_c = total_i = 0
            total_amas_global = 0
            has_any_dense = any(d.get("is_dense", False) for d in self._batch_results.values())

            for fp, data in self._batch_results.items():
                r = data["results"]
                
                # Calcular total de amastigotes: sumar conteos de todas las células infectadas
                total_amas_imagen = sum(c['amastigotes_count'] for c in r['per_cell'] if c['infected'])
                
                # Para display en resumen: mostrar conteo si es razonable, sino indicar rangos
                if data.get("is_dense"):
                    # Contar células por modo
                    modes = {}
                    for c in r['per_cell']:
                        if c['infected']:
                            modes[c['mode']] = modes.get(c['mode'], 0) + 1
                    
                    # Construir display inteligente
                    if total_amas_imagen <= 100:
                        # Si es razonable, mostrar el número estimado
                        amas_display = f"~{total_amas_imagen}"
                    else:
                        # Si es muy alto, mostrar rangos predominantes
                        if len(modes) == 1 and 'high_dense' in modes:
                            amas_display = ">50/cél"
                        elif 'high_dense' in modes:
                            amas_display = "Mix (>50)"
                        elif 'medium_dense' in modes:
                            amas_display = "15-50/cél"
                        else:
                            amas_display = "≤15/cél"
                else:
                    amas_display = total_amas_imagen
                
                for ci, v in enumerate([Path(fp).name, r["total_cells"],
                                         r["infected_cells"], f"{r['infection_pct']}%", amas_display], 1):
                    c = ws_sum.cell(row=row, column=ci, value=v)
                    c.font = NORM_FONT; c.border = brd
                    c.alignment = Alignment(horizontal="center")
                total_c += r["total_cells"]
                total_i += r["infected_cells"]
                total_amas_global += total_amas_imagen
                row += 1

            pct_g = round(total_i/total_c*100, 2) if total_c else 0
            # Total: mostrar solo si es razonable, sino "Mix"
            if total_amas_global <= 200:
                amas_total_display = f"~{total_amas_global}" if has_any_dense else total_amas_global
            else:
                amas_total_display = "Mix (ver hojas)"
            
            for ci, v in enumerate(["TOTAL", total_c, total_i, f"{pct_g}%", amas_total_display], 1):
                c = ws_sum.cell(row=row, column=ci, value=v)
                c.font = Font(bold=True, color="FFD740", name="Arial", size=10)
                c.fill = HDR_FILL; c.border = brd
                c.alignment = Alignment(horizontal="center")

            # ── Una hoja por imagen ───────────────────────────────────────────
            for fp, data in self._batch_results.items():
                r    = data["results"]
                name = Path(fp).name
                sname = name[:28].replace("/","_").replace("\\","_").replace(":","_")\
                                  .replace("?","_").replace("*","_")\
                                  .replace("[","_").replace("]","_")
                ws = wb.create_sheet(sname)
                ws.sheet_view.showGridLines = False

                # Titulo
                ws["A1"] = name
                ws["A1"].font = Font(bold=True, color="00E5FF", name="Arial", size=12)
                ws["A1"].fill = HDR_FILL
                ws.merge_cells("A1:F1")
                ws.row_dimensions[1].height = 22

                # Resumen imagen
                # Calcular total de amastigotes de células infectadas
                total_amas_imagen = sum(c['amastigotes_count'] for c in r['per_cell'] if c['infected'])
                
                # Display según el tipo de infección
                if data.get("is_dense"):
                    # Contar células por modo
                    modes_count = {}
                    for c in r['per_cell']:
                        if c['infected']:
                            modes_count[c['mode']] = modes_count.get(c['mode'], 0) + 1
                    
                    # Construir label descriptivo
                    mode_parts = []
                    if modes_count.get('sparse', 0) > 0:
                        mode_parts.append(f"{modes_count['sparse']} dispersas")
                    if modes_count.get('low_dense', 0) > 0:
                        mode_parts.append(f"{modes_count['low_dense']} ≤15")
                    if modes_count.get('medium_dense', 0) > 0:
                        mode_parts.append(f"{modes_count['medium_dense']} (15-50)")
                    if modes_count.get('high_dense', 0) > 0:
                        mode_parts.append(f"{modes_count['high_dense']} >50")
                    
                    modo_label = "Mix: " + ", ".join(mode_parts) if mode_parts else "Denso"
                    
                    # Mostrar estimación solo si es razonable
                    if total_amas_imagen <= 100:
                        amas_display = f"~{total_amas_imagen}"
                    else:
                        amas_display = "Ver tabla"
                else:
                    modo_label = "Disperso"
                    amas_display = total_amas_imagen
                
                # Añadir info de calibración con niveles de confianza
                calibration = r.get('calibration', {})
                cal_source = calibration.get('source', 'unknown')
                cal_confidence = calibration.get('confidence', 'unknown')
                
                if cal_source == 'auto':
                    n_used = calibration.get('n_used', 0)
                    n_outliers = calibration.get('n_outliers', 0)
                    size_cal = calibration.get('amas_size_used', 0)
                    quality = calibration.get('quality_score', 0)
                    cv = calibration.get('cv')
                    
                    # Icono según confianza
                    if cal_confidence == 'high':
                        conf_icon = "✓✓"  # Alta confianza
                    elif cal_confidence == 'medium':
                        conf_icon = "✓"   # Media confianza
                    elif cal_confidence == 'low':
                        conf_icon = "~"   # Baja confianza
                    else:
                        conf_icon = "?"
                    
                    # Construir mensaje informativo
                    modo_label += f" | {conf_icon}Cal: {n_used} amas, {size_cal:.1f}px"
                    
                    # Info adicional si hubo outliers relevantes
                    if n_outliers > 5:
                        modo_label += f" ({n_outliers} outliers)"
                        
                elif cal_source == 'manual':
                    size_cal = calibration.get('amas_size_used', 0)
                    modo_label += f" | Manual: {size_cal:.1f}px"
                elif cal_source == 'default':
                    modo_label += " | ⚠️ Sin calibración (usando 13px)"
                
                keys = ["Celulas totales","Infectadas","% Infeccion","Amastigotes","Modo deteccion"]
                vals = [r["total_cells"], r["infected_cells"], f"{r['infection_pct']}%",
                        amas_display, modo_label]
                for ci, (k, v) in enumerate(zip(keys, vals), 1):
                    ws.cell(row=2, column=ci, value=k).font = HDR_FONT
                    ws.cell(row=2, column=ci).fill = HDR_FILL
                    ws.cell(row=2, column=ci).alignment = Alignment(horizontal="center")
                    ws.cell(row=3, column=ci, value=v).font = NORM_FONT
                    ws.cell(row=3, column=ci).alignment = Alignment(horizontal="center")
                    ws.column_dimensions[get_column_letter(ci)].width = 16

                # Cabecera tabla con columna Intervalo
                for ci, h in enumerate(["ID Celula", "Amastigotes", "Intervalo", "Infectada"], 1):
                    c = ws.cell(row=5, column=ci, value=h)
                    c.font = HDR_FONT; c.fill = HDR_FILL
                    c.alignment = Alignment(horizontal="center"); c.border = brd

                # Datos celulas
                for ri, cd in enumerate(r["per_cell"], 6):
                    # Columna Amastigotes: solo número, nunca texto
                    amas_num = cd["amastigotes_count"]
                    
                    # Columna Intervalo: texto descriptivo
                    if cd["mode"] == "sparse":
                        intervalo = "Conteo exacto"
                    elif cd["mode"] == "low_dense":
                        intervalo = "≤15"
                    elif cd["mode"] == "medium_dense":
                        intervalo = "15-50"
                    elif cd["mode"] == "high_dense":
                        intervalo = ">50"
                    else:
                        intervalo = cd.get("amastigotes_nearby", "")
                    
                    row_vals = [cd["nucleus_id"],
                                amas_num,  # SIEMPRE número
                                intervalo,  # Texto descriptivo
                                "SI" if cd["infected"] else "NO"]
                    
                    for ci, v in enumerate(row_vals, 1):
                        c = ws.cell(row=ri, column=ci, value=v)
                        c.border = brd
                        c.alignment = Alignment(horizontal="center")
                        c.font = INF_FONT if cd["infected"] else OK_FONT
                        if cd["infected"]:
                            c.fill = INF_FILL

                # ═══════════════════════════════════════════════════════════════
                # IMÁGENES: Overlay + Original
                # ═══════════════════════════════════════════════════════════════
                
                # Imagen 1: Overlay (con anotaciones) en columna H
                overlay = data.get("overlay")
                if overlay is not None:
                    try:
                        img_pil = PILImage.fromarray(overlay)
                        ih, iw = overlay.shape[:2]
                        scale  = min(600/iw, 600/ih)
                        img_pil = img_pil.resize((int(iw*scale), int(ih*scale)), PILImage.LANCZOS)
                        tmp_path = os.path.join(tmp_dir, f"{sname}_overlay.png")
                        img_pil.save(tmp_path)
                        xl_img = XLImage(tmp_path)
                        ws.add_image(xl_img, "H1")
                        for col_i in range(8, 18):
                            ws.column_dimensions[get_column_letter(col_i)].width = 9
                    except Exception:
                        pass
                
                # Imagen 2: Original (sin anotaciones) en columna R (al lado)
                blue_ch = data.get("blue_channel")
                red_ch = data.get("red_channel")
                if blue_ch is not None and red_ch is not None:
                    try:
                        # Crear imagen RGB original
                        original_rgb = np.stack([red_ch, np.zeros_like(blue_ch), blue_ch], axis=2)
                        img_pil_orig = PILImage.fromarray(original_rgb)
                        ih_o, iw_o = original_rgb.shape[:2]
                        scale_o = min(600/iw_o, 600/ih_o)
                        img_pil_orig = img_pil_orig.resize((int(iw_o*scale_o), int(ih_o*scale_o)), PILImage.LANCZOS)
                        tmp_path_orig = os.path.join(tmp_dir, f"{sname}_original.png")
                        img_pil_orig.save(tmp_path_orig)
                        xl_img_orig = XLImage(tmp_path_orig)
                        ws.add_image(xl_img_orig, "R1")  # Columna R = 18
                        for col_i in range(18, 28):
                            ws.column_dimensions[get_column_letter(col_i)].width = 9
                    except Exception:
                        pass
                
                # ═══════════════════════════════════════════════════════════════════
                # GRÁFICAS POR IMAGEN (MEJORADAS Y VISTOSAS)
                # ═══════════════════════════════════════════════════════════════════
                try:
                    from openpyxl.chart import BarChart, Reference
                    from openpyxl.chart.label import DataLabelList
                    
                    # Recolectar datos de células infectadas
                    cells_1_15 = []  # Sin el 0
                    cells_16_50 = 0
                    cells_over_50 = 0
                    
                    for cd in r['per_cell']:
                        if cd['infected']:
                            amas = cd['amastigotes_count']
                            if 1 <= amas <= 15:  # Desde 1, no 0
                                cells_1_15.append(amas)
                            elif amas <= 50:
                                cells_16_50 += 1
                            else:
                                cells_over_50 += 1
                    
                    # Contar distribución 1-15 individual (sin 0)
                    dist_1_15 = {i: 0 for i in range(1, 16)}  # 1 a 15
                    for amas in cells_1_15:
                        if 1 <= amas <= 15:
                            dist_1_15[amas] += 1
                    
                    # ═══════════════════════════════════════════════════════════════
                    # GRÁFICA 1: Distribución General (≤15, 16-50, >50)
                    # ═══════════════════════════════════════════════════════════════
                    col_start = 19  # columna S
                    ws.cell(row=2, column=col_start, value="Intervalo")
                    ws.cell(row=2, column=col_start+1, value="Células")
                    ws.cell(row=3, column=col_start, value="1-15")  # Cambiado de ≤15
                    ws.cell(row=3, column=col_start+1, value=len(cells_1_15))
                    ws.cell(row=4, column=col_start, value="16-50")
                    ws.cell(row=4, column=col_start+1, value=cells_16_50)
                    ws.cell(row=5, column=col_start, value=">50")
                    ws.cell(row=5, column=col_start+1, value=cells_over_50)
                    
                    chart1 = BarChart()
                    chart1.type = "col"
                    chart1.style = 10
                    chart1.title = "Distribución de Células Infectadas"
                    chart1.y_axis.title = "Número de Células"
                    chart1.x_axis.title = "Rango de Amastigotes"
                    
                    data1 = Reference(ws, min_col=col_start+1, min_row=2, max_row=5)
                    cats1 = Reference(ws, min_col=col_start, min_row=3, max_row=5)
                    chart1.add_data(data1, titles_from_data=True)
                    chart1.set_categories(cats1)
                    
                    # ETIQUETAS DE DATOS encima de barras
                    chart1.dataLabels = DataLabelList()
                    chart1.dataLabels.showVal = True
                    chart1.dataLabels.showCatName = False
                    
                    # ═══════════════════════════════════════════════════════════════
                    # FORMATO CIENTÍFICO: Fuentes más grandes y en negrita
                    # ═══════════════════════════════════════════════════════════════
                    from openpyxl.chart.text import RichText
                    from openpyxl.drawing.text import Paragraph, ParagraphProperties, CharacterProperties
                    
                    # Título de la gráfica (14pt, negrita)
                    chart1.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1400, b=True))
                    
                    # Eje Y: etiquetas de números (12pt, negrita)
                    chart1.y_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                    # Eje Y: título (13pt, negrita)
                    chart1.y_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                    
                    # Eje X: etiquetas de categorías (12pt, negrita)
                    chart1.x_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                    # Eje X: título (13pt, negrita)
                    chart1.x_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                    
                    # Tamaño más grande y vistoso
                    chart1.height = 13.5
                    chart1.width = 21
                    
                    # Colocar gráfica debajo de la tabla con espacio suficiente
                    last_row = len(r['per_cell']) + 6
                    ws.add_chart(chart1, f"A{last_row + 2}")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # GRÁFICA 2: Detalle 1-15 con COLORES DIFERENTES y LEYENDA
                    # ═══════════════════════════════════════════════════════════════
                    col_detail = 32  # columna AF (más a la derecha para imágenes)
                    ws.cell(row=2, column=col_detail, value="Amas")
                    ws.cell(row=2, column=col_detail+1, value="Cels")
                    
                    # Solo de 1 a 15 (sin 0)
                    for i in range(1, 16):
                        ws.cell(row=2+i, column=col_detail, value=i)
                        ws.cell(row=2+i, column=col_detail+1, value=dist_1_15[i])
                    
                    # Crear gráfica con colores diferentes
                    from openpyxl.chart.series import SeriesLabel
                    
                    chart2 = BarChart()
                    chart2.type = "col"
                    chart2.title = "Detalle: Células con 1-15 Amastigotes"
                    chart2.y_axis.title = "Número de Células"
                    
                    # Paleta de 15 colores distintivos
                    colors = [
                        "FF6B6B", "4ECDC4", "45B7D1", "FFA07A", "98D8C8",
                        "F7DC6F", "BB8FCE", "85C1E2", "F8B88B", "ABEBC6",
                        "F1948A", "76D7C4", "F9E79F", "D7BDE2", "AED6F1"
                    ]
                    
                    # Agregar cada valor como una serie separada
                    for i in range(1, 16):
                        # Referencia para este valor específico
                        data_ref = Reference(ws, min_col=col_detail+1, min_row=2+i, max_row=2+i)
                        chart2.add_data(data_ref, titles_from_data=False)
                        
                        # Configurar color y título de la serie
                        serie = chart2.series[i-1]
                        serie.graphicalProperties.solidFill = colors[i-1]
                        serie.tx = SeriesLabel(v=str(i))
                    
                    # Configurar categorías (los números 1-15)
                    cats_ref = Reference(ws, min_col=col_detail, min_row=3, max_row=17)
                    chart2.set_categories(cats_ref)
                    
                    # ETIQUETAS DE DATOS sobre las barras
                    chart2.dataLabels = DataLabelList()
                    chart2.dataLabels.showVal = True
                    chart2.dataLabels.showCatName = False
                    chart2.dataLabels.showSerName = False
                    chart2.dataLabels.showLegendKey = False
                    
                    # LEYENDA a la derecha
                    chart2.legend.position = 'r'
                    
                    # ═══════════════════════════════════════════════════════════════
                    # FORMATO CIENTÍFICO
                    # ═══════════════════════════════════════════════════════════════
                    chart2.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1400, b=True))
                    
                    # Eje Y
                    chart2.y_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                    chart2.y_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                    
                    # Eje X
                    chart2.x_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1100, b=True)), endParaRPr=CharacterProperties(sz=1100, b=True))])
                    chart2.x_axis.tickLblSkip = 1
                    chart2.x_axis.tickMarkSkip = 1
                    
                    # Tamaño (más ancho para la leyenda)
                    chart2.height = 15
                    chart2.width = 35
                    
                    # Posición
                    ws.add_chart(chart2, f"A{last_row + 30}")
                    
                except Exception as e:
                    print(f"Error creando gráficas para {name}: {e}")
                    import traceback
                    traceback.print_exc()

            # ═══════════════════════════════════════════════════════════════════════
            # GRÁFICAS GLOBALES EN RESUMEN (MEJORADAS)
            # ═══════════════════════════════════════════════════════════════════════
            try:
                from openpyxl.chart import BarChart, Reference
                from openpyxl.chart.label import DataLabelList
                
                # Recolectar datos de TODAS las imágenes (sin 0)
                global_1_15 = []
                global_16_50 = 0
                global_over_50 = 0
                
                for fp, data in self._batch_results.items():
                    r = data["results"]
                    for cd in r['per_cell']:
                        if cd['infected']:
                            amas = cd['amastigotes_count']
                            if 1 <= amas <= 15:  # Sin 0
                                global_1_15.append(amas)
                            elif amas <= 50:
                                global_16_50 += 1
                            else:
                                global_over_50 += 1
                
                # Distribución detallada 1-15
                global_dist_1_15 = {i: 0 for i in range(1, 16)}
                for amas in global_1_15:
                    if 1 <= amas <= 15:
                        global_dist_1_15[amas] += 1
                
                # ═══════════════════════════════════════════════════════════════
                # GRÁFICA GLOBAL 1: Distribución General
                # ═══════════════════════════════════════════════════════════════
                col_g = 7  # columna G
                ws_sum.cell(row=row+3, column=col_g, value="Intervalo")
                ws_sum.cell(row=row+3, column=col_g+1, value="Total Células")
                ws_sum.cell(row=row+4, column=col_g, value="1-15")
                ws_sum.cell(row=row+4, column=col_g+1, value=len(global_1_15))
                ws_sum.cell(row=row+5, column=col_g, value="16-50")
                ws_sum.cell(row=row+5, column=col_g+1, value=global_16_50)
                ws_sum.cell(row=row+6, column=col_g, value=">50")
                ws_sum.cell(row=row+6, column=col_g+1, value=global_over_50)
                
                chart_g1 = BarChart()
                chart_g1.type = "col"
                chart_g1.style = 10
                chart_g1.title = "Distribución Global - Todas las Imágenes"
                chart_g1.y_axis.title = "Número Total de Células"
                chart_g1.x_axis.title = "Rango de Amastigotes"
                
                data_g1 = Reference(ws_sum, min_col=col_g+1, min_row=row+3, max_row=row+6)
                cats_g1 = Reference(ws_sum, min_col=col_g, min_row=row+4, max_row=row+6)
                chart_g1.add_data(data_g1, titles_from_data=True)
                chart_g1.set_categories(cats_g1)
                
                # ETIQUETAS DE DATOS
                chart_g1.dataLabels = DataLabelList()
                chart_g1.dataLabels.showVal = True
                chart_g1.dataLabels.showCatName = False
                
                # ═══════════════════════════════════════════════════════════════
                # FORMATO CIENTÍFICO: Fuentes más grandes y en negrita
                # ═══════════════════════════════════════════════════════════════
                # Título de la gráfica (14pt, negrita)
                chart_g1.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1400, b=True))
                
                # Eje Y: etiquetas de números (12pt, negrita)
                chart_g1.y_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                # Eje Y: título (13pt, negrita)
                chart_g1.y_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                
                # Eje X: etiquetas de categorías (12pt, negrita)
                chart_g1.x_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                # Eje X: título (13pt, negrita)
                chart_g1.x_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                
                chart_g1.height = 15
                chart_g1.width = 22
                
                # Mover más abajo para evitar solapamiento con los datos de la tabla (3 filas adicionales)
                ws_sum.add_chart(chart_g1, f"A{row + 8}")
                
                # ═══════════════════════════════════════════════════════════════
                # GRÁFICA GLOBAL 2: Detalle 1-15 con COLORES y LEYENDA
                # ═══════════════════════════════════════════════════════════════
                col_gd = 13  # columna M
                ws_sum.cell(row=row+3, column=col_gd, value="Amas")
                ws_sum.cell(row=row+3, column=col_gd+1, value="Cels")
                for i in range(1, 16):
                    ws_sum.cell(row=row+3+i, column=col_gd, value=i)
                    ws_sum.cell(row=row+3+i, column=col_gd+1, value=global_dist_1_15[i])
                
                chart_g2 = BarChart()
                chart_g2.type = "col"
                chart_g2.title = "Detalle Global: 1-15 Amastigotes - Todas las Imágenes"
                chart_g2.y_axis.title = "Número Total de Células"
                
                # Paleta de 15 colores distintivos
                colors = [
                    "FF6B6B", "4ECDC4", "45B7D1", "FFA07A", "98D8C8",
                    "F7DC6F", "BB8FCE", "85C1E2", "F8B88B", "ABEBC6",
                    "F1948A", "76D7C4", "F9E79F", "D7BDE2", "AED6F1"
                ]
                
                # Agregar cada valor como serie separada
                from openpyxl.chart.series import SeriesLabel
                
                for i in range(1, 16):
                    data_ref = Reference(ws_sum, min_col=col_gd+1, min_row=row+3+i, max_row=row+3+i)
                    chart_g2.add_data(data_ref, titles_from_data=False)
                    
                    # Color y título
                    serie = chart_g2.series[i-1]
                    serie.graphicalProperties.solidFill = colors[i-1]
                    serie.tx = SeriesLabel(v=str(i))
                
                # Configurar categorías
                cats_g_ref = Reference(ws_sum, min_col=col_gd, min_row=row+4, max_row=row+18)
                chart_g2.set_categories(cats_g_ref)
                
                # Etiquetas de datos
                chart_g2.dataLabels = DataLabelList()
                chart_g2.dataLabels.showVal = True
                chart_g2.dataLabels.showCatName = False
                chart_g2.dataLabels.showSerName = False
                chart_g2.dataLabels.showLegendKey = False
                
                # LEYENDA a la derecha
                chart_g2.legend.position = 'r'
                
                # ═══════════════════════════════════════════════════════════════
                # FORMATO CIENTÍFICO
                # ═══════════════════════════════════════════════════════════════
                chart_g2.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1400, b=True))
                
                chart_g2.y_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1200, b=True)), endParaRPr=CharacterProperties(sz=1200, b=True))])
                chart_g2.y_axis.title.text.rich.paragraphs[0].pPr = ParagraphProperties(defRPr=CharacterProperties(sz=1300, b=True))
                
                chart_g2.x_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=CharacterProperties(sz=1100, b=True)), endParaRPr=CharacterProperties(sz=1100, b=True))])
                chart_g2.x_axis.tickLblSkip = 1
                chart_g2.x_axis.tickMarkSkip = 1
                
                chart_g2.height = 16
                chart_g2.width = 38  # Más ancho para la leyenda
                
                ws_sum.add_chart(chart_g2, f"A{row + 37}")
                
            except Exception as e:
                print(f"Error creando gráficas globales: {e}")
                import traceback
                traceback.print_exc()

            wb.save(path)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            messagebox.showinfo("Exportado",
                f"Informe guardado:\n{path}\n\n"
                f"{len(self._batch_results)} imagenes - una hoja por imagen + Resumen")

        except ImportError:
            messagebox.showerror("Error", "Instala openpyxl:\npip install openpyxl")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    # ── Style ─────────────────────────────────────────────────────────────────
    def _apply_ttk_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Treeview", background=BG2, foreground=TEXT,
                    fieldbackground=BG2, rowheight=20, font=FONT_SMALL, borderwidth=0)
        s.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                    font=("Helvetica", 9, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", BG3)], foreground=[("selected", ACCENT)])
        s.configure("TCombobox", fieldbackground=BG3, background=BG3,
                    foreground=TEXT, selectbackground=BG3, arrowcolor=ACCENT)
        s.configure("Horizontal.TProgressbar", troughcolor=BG3, background=ACCENT, thickness=5)


if __name__ == "__main__":
    app = TcruzApp()
    app.mainloop()
