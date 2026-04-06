"""Tkinter UI for live threshold tuning."""

import colorsys
import numpy as np
import tkinter as tk
from tkinter import ttk


class ThresholdControlsUI:
    """Tkinter control panel for live HSV and contour-area threshold tuning."""

    def __init__(self, screen_width, camera_width, camera_height, lower_blue, upper_blue, min_area):
        self.screen_width = screen_width
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.default_lower_blue = np.array(lower_blue, dtype=np.uint8)
        self.default_upper_blue = np.array(upper_blue, dtype=np.uint8)
        self.default_min_area = int(min_area)

        self.root = None
        self.controls_vars = {}
        self.summary_label = None
        self.lower_color_box = None
        self.upper_color_box = None
        self.gradient_canvas = None
        self.closed = False

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Convert OpenCV-style HSV values to RGB for Tk widgets."""
        h_norm = max(0, min(int(h), 179)) / 179.0
        s_norm = max(0, min(int(s), 255)) / 255.0
        v_norm = max(0, min(int(v), 255)) / 255.0
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        return int(r * 255), int(g * 255), int(b * 255)

    @classmethod
    def _hsv_to_hex(cls, h, s, v):
        """Convert HSV values to a hex string for Tk color fields."""
        r, g, b = cls._hsv_to_rgb(h, s, v)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _contrast_text_color(r, g, b):
        """Pick a readable foreground color for a background RGB color."""
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "#000000" if luminance > 160 else "#ffffff"

    def _create_slider(self, parent, key, label, slider_from, slider_to, initial, length=320, resolution=1):
        """Create a labeled Tk slider and store its variable by key."""
        variable = tk.IntVar(value=int(initial))
        slider = tk.Scale(
            parent,
            from_=slider_from,
            to=slider_to,
            orient=tk.HORIZONTAL,
            label=label,
            variable=variable,
            length=length,
            resolution=resolution,
            command=self._on_slider_change,
        )
        slider.pack(fill="x", padx=8, pady=4)
        self.controls_vars[key] = variable

    def initialize(self):
        """Create the Tkinter control panel for live threshold tuning."""

        self.root = tk.Tk()
        self.root.title("Tracking Controls + Color Preview")
        self.root.resizable(True, True)
        self.root.minsize(920, 840)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        max_x = max(20, self.screen_width - 900)
        panel_x = min(max_x, 60 + (2 * self.camera_width))
        self.root.geometry(f"860x700+{panel_x}+30")

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        title = ttk.Label(container, text="Live Threshold Controls", font=("TkDefaultFont", 12, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w")

        subtitle = ttk.Label(
            container,
            text="Adjust sliders to tune HSV limits and noise filtering in real-time.",
        )
        subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        preview_frame = ttk.LabelFrame(container, text="Accepted Color Preview")
        preview_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        swatch_frame = ttk.Frame(preview_frame)
        swatch_frame.pack(fill="x", padx=8, pady=(8, 6))

        self.lower_color_box = tk.Label(
            swatch_frame,
            text="Lower",
            relief="sunken",
            width=36,
            bd=1,
            padx=6,
            pady=6,
        )
        self.lower_color_box.pack(side="left", padx=(0, 8))

        self.upper_color_box = tk.Label(
            swatch_frame,
            text="Upper",
            relief="sunken",
            width=36,
            bd=1,
            padx=6,
            pady=6,
        )
        self.upper_color_box.pack(side="left", padx=(8, 0))

        ttk.Label(
            preview_frame,
            text="Accepted color range",
        ).pack(anchor="w", padx=8)

        self.gradient_canvas = tk.Canvas(
            preview_frame,
            width=780,
            height=44,
            highlightthickness=1,
            highlightbackground="#888888",
            bd=0,
        )
        self.gradient_canvas.pack(fill="x", padx=8, pady=(4, 8))

        lower_frame = ttk.LabelFrame(container, text="Lower HSV Bounds")
        lower_frame.grid(row=3, column=0, padx=(0, 8), pady=(0, 8), sticky="nsew")

        upper_frame = ttk.LabelFrame(container, text="Upper HSV Bounds")
        upper_frame.grid(row=3, column=1, padx=(8, 0), pady=(0, 8), sticky="nsew")

        self._create_slider(lower_frame, "l_h", "Hue (0-179)", 0, 179, int(self.default_lower_blue[0]))
        self._create_slider(lower_frame, "l_s", "Saturation (0-255)", 0, 255, int(self.default_lower_blue[1]))
        self._create_slider(lower_frame, "l_v", "Value (0-255)", 0, 255, int(self.default_lower_blue[2]))

        self._create_slider(upper_frame, "u_h", "Hue (0-179)", 0, 179, int(self.default_upper_blue[0]))
        self._create_slider(upper_frame, "u_s", "Saturation (0-255)", 0, 255, int(self.default_upper_blue[1]))
        self._create_slider(upper_frame, "u_v", "Value (0-255)", 0, 255, int(self.default_upper_blue[2]))

        max_area = max(5000, (self.camera_width * self.camera_height) // 4)
        area_frame = ttk.LabelFrame(container, text="Noise Filter")
        area_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._create_slider(
            area_frame,
            "min_area",
            f"Minimum Contour Area (pixels, 0-{max_area})",
            0,
            max_area,
            int(max(0, min(self.default_min_area, max_area))),
            length=780,
            resolution=10,
        )

        actions = ttk.Frame(container)
        actions.grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Button(actions, text="Print Values", command=self.print_current_thresholds).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Reset Defaults", command=self.reset_thresholds).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Close & Stop", command=self.close).pack(side="left")

        self.summary_label = ttk.Label(container, text="", wraplength=720)
        self.summary_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self._update_summary()
        return True

    def _on_slider_change(self, _value=None):
        """Refresh summary text while sliders move."""
        self._update_summary()

    def get_thresholds(self):
        """Return current thresholds from UI values."""
        if not self.controls_vars:
            return self.default_lower_blue.copy(), self.default_upper_blue.copy(), self.default_min_area

        l_h_raw = self.controls_vars["l_h"].get()
        l_s_raw = self.controls_vars["l_s"].get()
        l_v_raw = self.controls_vars["l_v"].get()
        u_h_raw = self.controls_vars["u_h"].get()
        u_s_raw = self.controls_vars["u_s"].get()
        u_v_raw = self.controls_vars["u_v"].get()

        lower_blue = np.array(
            [
                min(l_h_raw, u_h_raw),
                min(l_s_raw, u_s_raw),
                min(l_v_raw, u_v_raw),
            ],
            dtype=np.uint8,
        )
        upper_blue = np.array(
            [
                max(l_h_raw, u_h_raw),
                max(l_s_raw, u_s_raw),
                max(l_v_raw, u_v_raw),
            ],
            dtype=np.uint8,
        )
        min_area = max(0, self.controls_vars["min_area"].get())

        return lower_blue, upper_blue, min_area

    def _update_summary(self):
        """Update the readable status line in the Tk control panel."""
        if self.summary_label is None:
            return

        lower_blue, upper_blue, min_area = self.get_thresholds()
        summary = (
            f"Active Lower HSV: {lower_blue.tolist()}   "
            f"Active Upper HSV: {upper_blue.tolist()}   "
            f"Min Area: {min_area}px"
        )
        self.summary_label.configure(text=summary)
        self._update_color_preview(lower_blue, upper_blue)

    def _update_color_preview(self, lower_blue, upper_blue):
        """Update lower/upper swatches and gradient preview for accepted HSV range."""
        if self.lower_color_box is None or self.upper_color_box is None or self.gradient_canvas is None:
            return

        lower_h, lower_s, lower_v = [int(value) for value in lower_blue.tolist()]
        upper_h, upper_s, upper_v = [int(value) for value in upper_blue.tolist()]

        lower_hex = self._hsv_to_hex(lower_h, lower_s, lower_v)
        upper_hex = self._hsv_to_hex(upper_h, upper_s, upper_v)
        lower_rgb = self._hsv_to_rgb(lower_h, lower_s, lower_v)
        upper_rgb = self._hsv_to_rgb(upper_h, upper_s, upper_v)

        self.lower_color_box.configure(
            bg=lower_hex,
            fg=self._contrast_text_color(*lower_rgb),
            text=f"Lower: {lower_hex}  HSV {lower_blue.tolist()}",
        )
        self.upper_color_box.configure(
            bg=upper_hex,
            fg=self._contrast_text_color(*upper_rgb),
            text=f"Upper: {upper_hex}  HSV {upper_blue.tolist()}",
        )

        self.gradient_canvas.delete("all")
        canvas_width = int(self.gradient_canvas.cget("width"))
        canvas_height = int(self.gradient_canvas.cget("height"))

        steps = 160
        step_width = canvas_width / float(steps)

        for idx in range(steps):
            t = 0.0 if steps == 1 else idx / float(steps - 1)
            hue = int(lower_h + (upper_h - lower_h) * t)
            sat = int(lower_s + (upper_s - lower_s) * t)
            val = int(lower_v + (upper_v - lower_v) * t)
            color_hex = self._hsv_to_hex(hue, sat, val)

            x0 = int(idx * step_width)
            x1 = int((idx + 1) * step_width) + 1
            self.gradient_canvas.create_rectangle(x0, 0, x1, canvas_height, fill=color_hex, outline="")

    def process_events(self):
        """Process Tk events once per frame to keep UI responsive."""
        if self.root is None:
            return False

        try:
            self.root.update_idletasks()
            self.root.update()
            return not self.closed
        except tk.TclError:
            self.root = None
            self.closed = True
            self.controls_vars = {}
            self.summary_label = None
            return False

    def print_current_thresholds(self):
        """Print the current threshold values for easy copy/paste."""
        lower_blue, upper_blue, min_area = self.get_thresholds()
        print("\nCurrent thresholds:")
        print(f"lower_blue = np.array({lower_blue.tolist()})")
        print(f"upper_blue = np.array({upper_blue.tolist()})")
        print(f"min_contour_area = {min_area}")

    def reset_thresholds(self):
        """Reset all control values to initial defaults."""
        if not self.controls_vars:
            return

        self.controls_vars["l_h"].set(int(self.default_lower_blue[0]))
        self.controls_vars["l_s"].set(int(self.default_lower_blue[1]))
        self.controls_vars["l_v"].set(int(self.default_lower_blue[2]))
        self.controls_vars["u_h"].set(int(self.default_upper_blue[0]))
        self.controls_vars["u_s"].set(int(self.default_upper_blue[1]))
        self.controls_vars["u_v"].set(int(self.default_upper_blue[2]))
        self.controls_vars["min_area"].set(int(self.default_min_area))
        self._update_summary()

    def close(self):
        """Close control panel and signal stop."""
        self.closed = True
        if self.root is not None:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.root = None

        self.controls_vars = {}
        self.summary_label = None
