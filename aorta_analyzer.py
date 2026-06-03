"""
Miguelyser — Atherosclerotic Plaque Analyzer  v2.0
Mouse Aorta (Oil Red O staining)

Build as a standalone app with PyInstaller:
  Windows:  pyinstaller --onefile --noconsole --name Miguelyser aorta_analyzer.py
  macOS:    pyinstaller --onefile --windowed  --name Miguelyser aorta_analyzer.py
"""

import os, sys, glob, threading, copy

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import pandas as pd

try:
    from scipy.ndimage import median_filter as _ndimage_median_filter
except Exception:
    def _ndimage_median_filter(data, size=21):
        half = int(size) // 2
        out = np.empty(len(data), dtype=float)
        for i in range(len(data)):
            out[i] = np.median(data[max(0, i - half):min(len(data), i + half + 1)])
        return out


# ════════════════════════════════════════════════════════════════════════════
#  EXE COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════════

def _res(rel_path=""):
    """Resolve a resource path — works from source and from a PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path) if rel_path else base


# ════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ════════════════════════════════════════════════════════════════════════════

C = dict(
    BG      = "#1A1A2E",   # main window background
    PANEL   = "#0D1B2A",   # image canvas background
    CARD    = "#16213E",   # frames / cards
    CARD2   = "#1E2D4A",   # slightly lighter card
    FG      = "#E8EAF0",   # primary text
    FG2     = "#8899AA",   # secondary / muted text
    ACCENT  = "#00DD55",   # green accent (brand)
    WARN    = "#FFB300",   # amber warning
    ERR     = "#FF4444",   # red error
    SEL     = "#1A5090",   # selection highlight

    # Toolbar button colors (bg, hover)
    B_FOLDER = ("#1565C0", "#1976D2"),
    B_ALL    = ("#1B5E20", "#2E7D32"),
    B_SEL    = ("#004D40", "#00695C"),
    B_XLS    = ("#4A148C", "#6A1B9A"),
    B_EDIT   = ("#BF360C", "#D84315"),
    B_HELP   = ("#37474F", "#455A64"),
    B_CANCEL = ("#B71C1C", "#C62828"),
)


# ════════════════════════════════════════════════════════════════════════════
#  DEFAULT PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = dict(
    aorta_bright_min = 55,
    aorta_open_k     = 13,
    aorta_close_k    = 27,
    plaque_h_max     = 22,
    plaque_s_min     = 110,
    plaque_v_min     = 75,
    plaque_open_k    = 5,
)

# Annotation colours (BGR for OpenCV)
C_AORTA  = (0,   220,   0)
C_PLAQUE = (220,  30,   0)
C_TRUNK  = (200,  80,   0)
C_LEFT   = (0,   200, 200)
C_RIGHT  = (180,   0, 200)
OVERLAY_ALPHA = 0.25


# ════════════════════════════════════════════════════════════════════════════
#  CORE IMAGE PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def detect_circle_mask(gray):
    """Return a binary mask for the circular microscope field."""
    _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8) * 255
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def detect_aorta(img_bgr, circle_mask, params):
    """
    Segment the aorta from the background.
    Key order: open → pick largest component → close → bridge-break.
    Fat globules (yellow-orange, H≈20-62) are excluded before the pipeline
    so they cannot be included in the aorta mask.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_masked = cv2.bitwise_and(gray, gray, mask=circle_mask)
    pixels = gray_masked[circle_mask > 0]
    otsu_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(int(otsu_val * 0.80), params["aorta_bright_min"])
    _, binary = cv2.threshold(gray_masked, thresh_val, 255, cv2.THRESH_BINARY)

    # Exclude yellow/brown fat globules.  Aorta tissue is near-achromatic (S≈19)
    # or blue-purple (H≈130-160); fat is yellow-orange (H≈20-62) with S>28.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    fat_mask = cv2.inRange(hsv, np.array([20, 28, 40]), np.array([62, 255, 255]))
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(fat_mask))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(params["aorta_open_k"]))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels < 2:
        return opened
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.where(labels == largest, np.uint8(255), np.uint8(0))
    closed = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, _kernel(params["aorta_close_k"]))
    bridge_k = max(3, params["aorta_close_k"] // 4)
    eroded = cv2.erode(closed, _kernel(bridge_k))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(eroded)
    if n2 >= 2:
        lg2 = 1 + np.argmax(st2[1:, cv2.CC_STAT_AREA])
        return cv2.dilate(np.where(lab2 == lg2, np.uint8(255), np.uint8(0)), _kernel(bridge_k))
    return closed


def detect_plaques(img_bgr, aorta_mask, params):
    """Segment Oil Red O plaques (orange-red) within the aorta mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo  = np.array([0,   params["plaque_s_min"], params["plaque_v_min"]])
    hi  = np.array([params["plaque_h_max"], 255, 255])
    plaque = cv2.inRange(hsv, lo, hi)
    lo2 = np.array([165, params["plaque_s_min"], params["plaque_v_min"]])
    plaque = cv2.bitwise_or(plaque, cv2.inRange(hsv, lo2, np.array([180, 255, 255])))
    opened = cv2.morphologyEx(plaque, cv2.MORPH_OPEN, _kernel(params["plaque_open_k"]))
    return cv2.bitwise_and(opened, opened, mask=aorta_mask)


def _row_runs(row, min_width=15):
    runs, in_run, start = [], False, 0
    for x in range(len(row)):
        if row[x] > 0 and not in_run:
            start, in_run = x, True
        elif row[x] == 0 and in_run:
            if x - start >= min_width:
                runs.append((start, x - 1))
            in_run = False
    if in_run and len(row) - start >= min_width:
        runs.append((start, len(row) - 1))
    return runs


def find_bifurcation(aorta_mask):
    """Find the Y-bifurcation row using row-scan (method A) or width-profile (method B)."""
    h, w = aorta_mask.shape
    aorta_rows = np.where(np.any(aorta_mask > 0, axis=1))[0]
    if len(aorta_rows) == 0:
        return w // 2, h // 2
    y_top, y_bot = int(aorta_rows.min()), int(aorta_rows.max())
    y_search_bot = y_top + int((y_bot - y_top) * 0.65)
    last_two_y = last_two_runs = None
    for y in range(y_top, y_search_bot + 1):
        runs = _row_runs(aorta_mask[y, :], min_width=15)
        if len(runs) >= 2 and runs[1][0] - runs[0][1] - 1 >= 10:
            last_two_y, last_two_runs = y, runs
    if last_two_y is not None:
        cx = [(r[0] + r[1]) // 2 for r in last_two_runs[:2]]
        return (cx[0] + cx[1]) // 2, last_two_y
    widths = np.array([int(np.count_nonzero(aorta_mask[y, :]))
                       for y in range(y_top, y_search_bot + 1)], dtype=float)
    ws = _ndimage_median_filter(widths, size=21)
    arm_end = len(ws) // 3
    max_w = ws[:arm_end].max() if arm_end > 0 else ws.max()
    above = np.where(ws > max_w * 0.55)[0]
    bif_idx = int(above.max()) if len(above) else len(widths) // 2
    bif_y = y_top + bif_idx
    cols = np.where(aorta_mask[bif_y, :] > 0)[0]
    return (int(cols.mean()) if len(cols) else w // 2), bif_y


def segment_regions(aorta_mask, bif_x, bif_y):
    """Split aorta mask into trunk, left_arm, right_arm sub-masks."""
    trunk, left_arm, right_arm = (np.zeros_like(aorta_mask) for _ in range(3))
    ys, xs = np.where(aorta_mask > 0)
    for y, x in zip(ys, xs):
        if y > bif_y:
            trunk[y, x] = 255
        elif x < bif_x:
            left_arm[y, x] = 255
        else:
            right_arm[y, x] = 255
    return trunk, left_arm, right_arm


def draw_dashed_contour(img, contours, color, thickness=2, dash_len=12, gap_len=8):
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        n, i, draw, seg = len(pts), 0, True, 0
        while i < n:
            j = (i + 1) % n
            if draw:
                cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)
                seg += 1
                if seg >= dash_len:
                    draw, seg = False, 0
            else:
                seg += 1
                if seg >= gap_len:
                    draw, seg = True, 0
            i += 1


def draw_region_overlay(img, mask, color, alpha=OVERLAY_ALPHA):
    overlay = img.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def annotate_image(img_bgr, aorta_mask, plaque_mask, trunk, left_arm, right_arm,
                   bif_x, bif_y, stats):
    out = img_bgr.copy()
    draw_region_overlay(out, trunk,     C_TRUNK,  OVERLAY_ALPHA)
    draw_region_overlay(out, left_arm,  C_LEFT,   OVERLAY_ALPHA)
    draw_region_overlay(out, right_arm, C_RIGHT,  OVERLAY_ALPHA)
    ctrs_a, _ = cv2.findContours(aorta_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, ctrs_a, -1, C_AORTA, 3)
    ctrs_p, _ = cv2.findContours(plaque_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_dashed_contour(out, ctrs_p, C_PLAQUE, thickness=2)
    cv2.circle(out, (bif_x, bif_y), 8, (255, 255, 255), -1)
    cv2.circle(out, (bif_x, bif_y), 8, (0, 0, 0), 2)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    def label_region(mask, text, fg=(255, 255, 255), bg=(0, 0, 0)):
        ys, xs = np.where(mask > 0)
        if not len(ys):
            return
        cx, cy = int(xs.mean()), int(ys.mean())
        (tw, thi), _ = cv2.getTextSize(text, font, fs, th)
        cv2.rectangle(out, (cx - tw//2 - 4, cy - thi - 4), (cx + tw//2 + 4, cy + 4), bg, -1)
        cv2.putText(out, text, (cx - tw//2, cy), font, fs, fg, th, cv2.LINE_AA)
    label_region(trunk,     f"Trunk {stats['trunk_plaque_pct']:.1f}%",     fg=(255,255,255), bg=(60,40,0))
    label_region(left_arm,  f"L-Arm {stats['left_arm_plaque_pct']:.1f}%",  fg=(0,0,0),       bg=(0,200,200))
    label_region(right_arm, f"R-Arm {stats['right_arm_plaque_pct']:.1f}%", fg=(0,0,0),       bg=(0,200,0))
    total_txt = f"Total plaque burden: {stats['plaque_pct']:.1f}%"
    (tw, thi), _ = cv2.getTextSize(total_txt, font, 1.3, 3)
    cv2.rectangle(out, (10, 10), (tw + 20, thi + 24), (0, 0, 0), -1)
    cv2.putText(out, total_txt, (16, thi + 14), font, 1.3, (0, 230, 0), 3, cv2.LINE_AA)
    return out


def process_image(img_bgr, params=None):
    """Full pipeline. Returns (annotated, aorta_mask, plaque_mask, trunk, la, ra, bif_xy, stats)."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circle_mask = detect_circle_mask(gray)
    aorta_mask  = detect_aorta(img_bgr, circle_mask, params)
    plaque_mask = detect_plaques(img_bgr, aorta_mask, params)
    bif_x, bif_y = find_bifurcation(aorta_mask)
    trunk, la, ra = segment_regions(aorta_mask, bif_x, bif_y)
    def area(m):   return int(np.count_nonzero(m))
    def pct(p, a): return round(100.0 * p / a, 2) if a else 0.0
    aa = area(aorta_mask); pa = area(plaque_mask)
    ta = area(trunk);  tp = area(cv2.bitwise_and(plaque_mask, plaque_mask, mask=trunk))
    la_a = area(la);   lp = area(cv2.bitwise_and(plaque_mask, plaque_mask, mask=la))
    ra_a = area(ra);   rp = area(cv2.bitwise_and(plaque_mask, plaque_mask, mask=ra))
    stats = dict(
        aorta_area_px=aa,  plaque_area_px=pa,  plaque_pct=pct(pa, aa),
        trunk_area_px=ta,  trunk_plaque_px=tp,  trunk_plaque_pct=pct(tp, ta),
        left_arm_area_px=la_a, left_arm_plaque_px=lp,  left_arm_plaque_pct=pct(lp, la_a),
        right_arm_area_px=ra_a,right_arm_plaque_px=rp, right_arm_plaque_pct=pct(rp, ra_a),
    )
    annotated = annotate_image(img_bgr, aorta_mask, plaque_mask, trunk, la, ra, bif_x, bif_y, stats)
    return annotated, aorta_mask, plaque_mask, trunk, la, ra, (bif_x, bif_y), stats


# ════════════════════════════════════════════════════════════════════════════
#  ICON
# ════════════════════════════════════════════════════════════════════════════

def _make_icon(size=64):
    img = Image.new("RGBA", (size, size), (26, 26, 46, 255))
    d, s = ImageDraw.Draw(img), size / 64.0
    def pt(x, y): return (int(x * s), int(y * s))
    ac = (100, 80, 125, 255)
    d.rectangle([pt(25, 36), pt(39, 62)], fill=ac)
    d.polygon([pt(37,32), pt(27,40), pt(3,12), pt(13,4)],  fill=ac)
    d.polygon([pt(27,32), pt(37,40), pt(61,12), pt(51,4)], fill=ac)
    pc, rp = (210, 60, 15, 255), max(3, int(4 * s))
    for cx, cy in [(32,48),(10,12),(54,12)]:
        x, y = int(cx * s), int(cy * s)
        d.ellipse([x-rp, y-rp, x+rp, y+rp], fill=pc)
    gc, bw = (0, 200, 60, 255), max(2, int(2 * s))
    for t in range(bw):
        d.rectangle([t, t, size-1-t, size-1-t], outline=gc)
    return img


# ════════════════════════════════════════════════════════════════════════════
#  THEME SETUP
# ════════════════════════════════════════════════════════════════════════════

def _apply_theme(root):
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except Exception:
        pass
    s.configure(".",          background=C["BG"],   foreground=C["FG"], font=("Helvetica", 11))
    s.configure("TFrame",     background=C["BG"])
    s.configure("Card.TFrame",background=C["CARD"])
    s.configure("TLabel",     background=C["BG"],   foreground=C["FG"])
    s.configure("Card.TLabel",background=C["CARD"], foreground=C["FG"])
    s.configure("Dim.TLabel", background=C["BG"],   foreground=C["FG2"], font=("Helvetica", 9))
    s.configure("Title.TLabel",background=C["BG"],  foreground=C["ACCENT"], font=("Helvetica", 13, "bold"))
    s.configure("Burden.TLabel",background=C["BG"], foreground=C["ACCENT"], font=("Helvetica", 16, "bold"))
    s.configure("TLabelframe",       background=C["BG"])
    s.configure("TLabelframe.Label", background=C["BG"], foreground=C["ACCENT"], font=("Helvetica", 10, "bold"))
    s.configure("TButton",    background=C["CARD"],  foreground=C["FG"],  relief="flat", padding=(8,5))
    s.map("TButton",
          background=[("active", C["CARD2"]), ("pressed", C["SEL"])],
          foreground=[("active", C["FG"])])
    s.configure("Treeview",   background=C["PANEL"],foreground=C["FG"],
                fieldbackground=C["PANEL"], rowheight=22, font=("Courier", 9))
    s.configure("Treeview.Heading", background=C["CARD"], foreground=C["ACCENT"],
                font=("Helvetica", 9, "bold"), relief="flat")
    s.map("Treeview",
          background=[("selected", C["SEL"])],
          foreground=[("selected", C["FG"])])
    s.configure("TScrollbar",  background=C["CARD"],  troughcolor=C["BG"], arrowcolor=C["FG2"])
    s.configure("TPanedwindow",background=C["BG"])
    s.configure("TScale",      background=C["CARD"],  troughcolor=C["PANEL"])
    s.configure("TRadiobutton",background=C["CARD"],  foreground=C["FG"])
    s.map("TRadiobutton",
          background=[("active", C["CARD"])],
          foreground=[("active", C["ACCENT"])])
    s.configure("Horizontal.TProgressbar",
                background=C["ACCENT"], troughcolor=C["PANEL"],
                bordercolor=C["BG"], lightcolor=C["ACCENT"], darkcolor=C["ACCENT"])
    s.configure("TSeparator", background="#334455")


# ════════════════════════════════════════════════════════════════════════════
#  UTILITY WIDGETS
# ════════════════════════════════════════════════════════════════════════════

class _Btn(tk.Button):
    """Flat coloured toolbar button with hover effect."""
    def __init__(self, parent, text, color_pair, command=None, **kw):
        bg, hover = color_pair
        super().__init__(parent, text=text, command=command,
                         bg=bg, fg="#FFFFFF", activebackground=hover,
                         activeforeground="#FFFFFF",
                         relief=tk.FLAT, bd=0, padx=12, pady=6,
                         font=("Helvetica", 11, "bold"),
                         cursor="hand2", **kw)
        self._bg, self._hover = bg, hover
        self.bind("<Enter>", lambda _: self.config(bg=self._hover))
        self.bind("<Leave>", lambda _: self.config(bg=self._bg))

    def set_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.config(state=state)
        self.config(bg=self._bg if enabled else "#333333")


class _Tip:
    """Simple hover tooltip."""
    def __init__(self, widget, text):
        self._win = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        self._text = text

    def _show(self, event):
        x, y = event.widget.winfo_rootx() + 20, event.widget.winfo_rooty() + 28
        self._win = tw = tk.Toplevel()
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, bg="#FFFFC0", fg="#222222",
                 relief=tk.SOLID, bd=1, padx=6, pady=3,
                 font=("Helvetica", 10)).pack()

    def _hide(self, _):
        if self._win:
            self._win.destroy()
            self._win = None


class _ZoomWindow(tk.Toplevel):
    """Full-screen zoom view for an image (double-click to open)."""
    def __init__(self, parent, bgr_img, title=""):
        super().__init__(parent)
        self.title(title or "Image Zoom")
        self.configure(bg=C["BG"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(sw-80,1400)}x{min(sh-80,900)}+40+40")
        self.resizable(True, True)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Fit to window
        w, h = pil.size
        scale = min((sw-120)/w, (sh-120)/h)
        nw, nh = int(w*scale), int(h*scale)
        pil = pil.resize((nw, nh), Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(pil)
        tk.Label(self, image=self._tk, bg=C["PANEL"]).pack(expand=True, fill=tk.BOTH)
        tk.Label(self, text="Press Escape or click to close",
                 bg=C["BG"], fg=C["FG2"], font=("Helvetica", 10)).pack(pady=4)
        self.bind("<Escape>", lambda _: self.destroy())
        self.bind("<Button-1>", lambda _: self.destroy())
        self.focus_set()


# ════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ════════════════════════════════════════════════════════════════════════════

class SplashScreen(tk.Toplevel):
    W, H = 520, 330

    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=C["BG"])
        self.attributes("-topmost", True)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{self.W}x{self.H}+{(sw-self.W)//2}+{(sh-self.H)//2}")

        icon_pil = _make_icon(80)
        self._icon_tk = ImageTk.PhotoImage(icon_pil)
        tk.Label(self, image=self._icon_tk, bg=C["BG"]).pack(pady=(20, 6))

        tk.Label(self, text="Atherosclerotic Plaque Analyzer",
                 font=("Helvetica", 22, "bold"),
                 fg=C["ACCENT"], bg=C["BG"]).pack()
        tk.Label(self, text="Oil Red O Staining  ·  Mouse Aorta  ·  v2.0",
                 font=("Helvetica", 11),
                 fg=C["FG2"], bg=C["BG"]).pack(pady=(4, 0))

        tk.Frame(self, bg="#3A3A6A", height=1).pack(fill=tk.X, padx=50, pady=14)

        self._hint = tk.Label(self, text="Loading…",
                              font=("Helvetica", 10), fg="#445566", bg=C["BG"])
        self._hint.pack()
        self.bind("<Button-1>", lambda _: self._close())
        self._animate(0)
        self.after(3200, self._close)

    def _animate(self, step):
        dots = "." * (step % 4)
        self._hint.config(text=f"Loading{dots}")
        if self.winfo_exists():
            self.after(400, self._animate, step + 1)

    def _close(self):
        if self.winfo_exists():
            self.destroy()


# ════════════════════════════════════════════════════════════════════════════
#  EDIT BOUNDARIES WINDOW
# ════════════════════════════════════════════════════════════════════════════

class EditBoundariesWindow(tk.Toplevel):
    BRUSH_MODES = ["Aorta ADD", "Aorta ERASE", "Plaque ADD", "Plaque ERASE"]
    CANVAS_W, CANVAS_H = 720, 620

    def __init__(self, parent, img_bgr, init_params, on_apply):
        super().__init__(parent)
        self.title("Edit Boundaries")
        self.configure(bg=C["BG"])
        self.resizable(False, False)
        self.transient(parent)

        self._orig     = img_bgr.copy()
        self._params   = copy.deepcopy(init_params)
        self._on_apply = on_apply
        self._aorta_mask = self._plaque_mask = self._bif_xy = self._stats = None
        self._brush_mode = tk.StringVar(value=self.BRUSH_MODES[0])
        self._brush_size = tk.IntVar(value=20)
        self._drawing    = False

        try:
            self._build_ui()
        except Exception as exc:
            messagebox.showerror("UI Error", f"Could not build editor:\n{exc}", parent=self)
            self.destroy()
            return

        self.update_idletasks()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.after(150, lambda: self.attributes("-topmost", False))
        self.after(50, self._safe_reprocess)

    def _safe_reprocess(self):
        try:
            self._reprocess()
        except Exception as exc:
            messagebox.showerror("Processing Error",
                                 f"Could not process image:\n{exc}", parent=self)

    def _build_ui(self):
        top = tk.Frame(self, bg=C["BG"], pady=6, padx=6)
        top.pack(side=tk.TOP, fill=tk.X)

        # Sliders
        sf = tk.LabelFrame(top, text="Threshold Sliders (live preview)",
                           bg=C["CARD"], fg=C["ACCENT"], font=("Helvetica", 10, "bold"),
                           padx=8, pady=6, bd=1, relief=tk.FLAT)
        sf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        def slider(parent, label, key, lo, hi):
            tk.Label(parent, text=label, bg=C["CARD"], fg=C["FG"],
                     font=("Helvetica", 10), anchor=tk.W).pack(fill=tk.X)
            var = tk.IntVar(value=self._params[key])
            val_lbl = tk.Label(parent, textvariable=var, bg=C["CARD"], fg=C["ACCENT"],
                               font=("Helvetica", 9, "bold"), anchor=tk.E)
            def cmd(v, k=key, sv=var):
                self._params[k] = int(float(v))
                try:
                    self._reprocess()
                except Exception:
                    pass
            ttk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      variable=var, command=cmd, length=220).pack(fill=tk.X)
            val_lbl.pack(anchor=tk.E)

        slider(sf, "Aorta brightness min",   "aorta_bright_min", 20, 150)
        slider(sf, "Aorta open kernel",      "aorta_open_k",      3,  40)
        slider(sf, "Aorta close kernel",     "aorta_close_k",     5,  60)
        slider(sf, "Plaque hue max (0–180)", "plaque_h_max",      5,  40)
        slider(sf, "Plaque saturation min",  "plaque_s_min",     40, 200)
        slider(sf, "Plaque brightness min",  "plaque_v_min",     30, 180)
        slider(sf, "Plaque open kernel",     "plaque_open_k",     1,  20)

        # Brush controls
        bf = tk.LabelFrame(top, text="Paint Brush",
                           bg=C["CARD"], fg=C["ACCENT"], font=("Helvetica", 10, "bold"),
                           padx=8, pady=6, bd=1, relief=tk.FLAT)
        bf.pack(side=tk.LEFT, fill=tk.Y)

        brush_colors = {"Aorta ADD": "#1B5E20", "Aorta ERASE": "#B71C1C",
                        "Plaque ADD": "#1565C0", "Plaque ERASE": "#BF360C"}
        for mode in self.BRUSH_MODES:
            fg_col = brush_colors.get(mode, C["FG"])
            tk.Radiobutton(bf, text=mode, variable=self._brush_mode, value=mode,
                           bg=C["CARD"], fg=fg_col, selectcolor=C["CARD2"],
                           activebackground=C["CARD"], activeforeground=fg_col,
                           font=("Helvetica", 10), anchor=tk.W).pack(fill=tk.X, pady=2)
        tk.Frame(bf, bg="#334455", height=1).pack(fill=tk.X, pady=6)
        tk.Label(bf, text="Brush size (px)", bg=C["CARD"], fg=C["FG2"],
                 font=("Helvetica", 9)).pack(anchor=tk.W)
        ttk.Scale(bf, from_=3, to=80, orient=tk.HORIZONTAL,
                  variable=self._brush_size, length=190).pack()
        tk.Label(bf, textvariable=self._brush_size, bg=C["CARD"], fg=C["ACCENT"],
                 font=("Helvetica", 9, "bold")).pack(anchor=tk.E)

        tk.Frame(bf, bg="#334455", height=1).pack(fill=tk.X, pady=8)
        tk.Label(bf, text="Tips:", bg=C["CARD"], fg=C["ACCENT"],
                 font=("Helvetica", 9, "bold")).pack(anchor=tk.W)
        for tip in ["• Click & drag on image", "• Adjust sliders first",
                    "• Reset undoes all strokes"]:
            tk.Label(bf, text=tip, bg=C["CARD"], fg=C["FG2"],
                     font=("Helvetica", 8), anchor=tk.W).pack(fill=tk.X)

        # Canvas
        self._canvas = tk.Canvas(self, width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg=C["PANEL"], cursor="crosshair", highlightthickness=0)
        self._canvas.pack(padx=6, pady=4)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        # Buttons
        btn_bar = tk.Frame(self, bg=C["BG"], pady=6)
        btn_bar.pack(fill=tk.X, padx=6)
        _Btn(btn_bar, "Reset to Auto",  C["B_HELP"],   command=self._reset).pack(side=tk.LEFT, padx=4)
        _Btn(btn_bar, "Apply & Save",   C["B_ALL"],    command=self._apply).pack(side=tk.LEFT, padx=4)
        _Btn(btn_bar, "Cancel",         C["B_CANCEL"], command=self.destroy).pack(side=tk.RIGHT, padx=4)

    def _reprocess(self):
        gray   = cv2.cvtColor(self._orig, cv2.COLOR_BGR2GRAY)
        circle = detect_circle_mask(gray)
        self._aorta_mask  = detect_aorta(self._orig, circle, self._params)
        self._plaque_mask = detect_plaques(self._orig, self._aorta_mask, self._params)
        self._bif_xy      = find_bifurcation(self._aorta_mask)
        self._refresh_canvas()

    def _refresh_canvas(self):
        bx, by = self._bif_xy
        trunk, la, ra = segment_regions(self._aorta_mask, bx, by)
        def area(m):   return int(np.count_nonzero(m))
        def pct(p, a): return round(100.0 * p / a, 2) if a else 0.0
        aa = area(self._aorta_mask); pa = area(self._plaque_mask)
        ta = area(trunk); tp = area(cv2.bitwise_and(self._plaque_mask, self._plaque_mask, mask=trunk))
        la_a=area(la);    lp = area(cv2.bitwise_and(self._plaque_mask, self._plaque_mask, mask=la))
        ra_a=area(ra);    rp = area(cv2.bitwise_and(self._plaque_mask, self._plaque_mask, mask=ra))
        self._stats = dict(
            aorta_area_px=aa, plaque_area_px=pa, plaque_pct=pct(pa,aa),
            trunk_area_px=ta, trunk_plaque_px=tp, trunk_plaque_pct=pct(tp,ta),
            left_arm_area_px=la_a, left_arm_plaque_px=lp, left_arm_plaque_pct=pct(lp,la_a),
            right_arm_area_px=ra_a, right_arm_plaque_px=rp, right_arm_plaque_pct=pct(rp,ra_a),
        )
        preview = annotate_image(self._orig, self._aorta_mask, self._plaque_mask,
                                 trunk, la, ra, bx, by, self._stats)
        self._show_on_canvas(preview)

    def _show_on_canvas(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.CANVAS_W / w, self.CANVAS_H / h)
        nw, nh = int(w * scale), int(h * scale)
        small = cv2.resize(bgr, (nw, nh))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._scale  = scale
        self._offset = ((self.CANVAS_W - nw) // 2, (self.CANVAS_H - nh) // 2)
        self._canvas.delete("all")
        self._canvas.create_image(self._offset[0], self._offset[1],
                                  anchor=tk.NW, image=self._tk_img)

    def _canvas_to_image(self, cx, cy):
        ox, oy = self._offset
        h, w = self._orig.shape[:2]
        return (max(0, min(int((cx - ox) / self._scale), w-1)),
                max(0, min(int((cy - oy) / self._scale), h-1)))

    def _paint(self, cx, cy):
        ix, iy = self._canvas_to_image(cx, cy)
        r    = max(1, int(self._brush_size.get() / self._scale))
        mode = self._brush_mode.get()
        mask = self._aorta_mask if "Aorta" in mode else self._plaque_mask
        val  = 255 if "ADD" in mode else 0
        cv2.circle(mask, (ix, iy), r, val, -1)
        if "Plaque" in mode and val == 255:
            self._plaque_mask = cv2.bitwise_and(self._plaque_mask, self._plaque_mask,
                                                mask=self._aorta_mask)
        self._bif_xy = find_bifurcation(self._aorta_mask)
        self._refresh_canvas()

    def _on_press(self, e):   self._drawing = True;  self._paint(e.x, e.y)
    def _on_drag(self, e):
        if self._drawing:     self._paint(e.x, e.y)
    def _on_release(self, e): self._drawing = False

    def _reset(self):
        self._reprocess()

    def _apply(self):
        bx, by = self._bif_xy
        trunk, la, ra = segment_regions(self._aorta_mask, bx, by)
        annotated = annotate_image(self._orig, self._aorta_mask, self._plaque_mask,
                                   trunk, la, ra, bx, by, self._stats)
        self._on_apply(annotated, self._aorta_mask, self._plaque_mask,
                       trunk, la, ra, self._bif_xy, self._stats, self._params)
        self.destroy()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

class AortaAnalyzer(tk.Tk):

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self):
        super().__init__()
        self.withdraw()

        _apply_theme(self)
        self.configure(bg=C["BG"])
        self.title("Atherosclerotic Plaque Analyzer")
        self.geometry("1300x840")
        self.minsize(960, 640)

        try:
            self._app_icon = ImageTk.PhotoImage(_make_icon(128))
            self.iconphoto(True, self._app_icon)
        except Exception:
            pass

        self._folder        = None
        self._output_folder = None
        self._files         = []
        self._sel_idx       = -1
        self._cache         = {}
        self._results       = {}
        self._orig_bgr      = {}
        self._cancel        = threading.Event()
        self._busy          = False

        self._build_ui()
        self.bind("<Left>",  lambda e: self._nav(-1))
        self.bind("<Right>", lambda e: self._nav(+1))

        splash = SplashScreen(self)
        self.update()

        def _show():
            if splash.winfo_exists():
                splash.destroy()
            self.deiconify()
            self.lift()

        self.after(3200, _show)

    # ── Theme-aware configure for all direct children ────────────────────────

    def _build_ui(self):
        # ── Menu bar ─────────────────────────────────────────────────────────
        menubar = tk.Menu(self, bg=C["CARD"], fg=C["FG"])
        help_menu = tk.Menu(menubar, tearoff=0, bg=C["CARD"], fg=C["FG"])
        help_menu.add_command(label="Quick Start Guide", command=self._show_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About Miguelyser", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=C["CARD"], pady=5, padx=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self._btn_folder = _Btn(toolbar, "📁  Open Folder", C["B_FOLDER"],
                                command=self._select_folder)
        self._btn_folder.pack(side=tk.LEFT, padx=3)
        _Tip(self._btn_folder, "Select a folder of JPEG/PNG images to analyse")

        self._btn_all = _Btn(toolbar, "⚡  Process All", C["B_ALL"],
                             command=self._process_all)
        self._btn_all.pack(side=tk.LEFT, padx=3)
        _Tip(self._btn_all, "Run analysis on every image in the folder")

        self._btn_sel = _Btn(toolbar, "⊙  Process Selected", C["B_SEL"],
                             command=self._process_selected)
        self._btn_sel.pack(side=tk.LEFT, padx=3)
        _Tip(self._btn_sel, "Analyse only the currently highlighted image")

        self._btn_xls = _Btn(toolbar, "📊  Export Excel", C["B_XLS"],
                             command=self._export_excel)
        self._btn_xls.pack(side=tk.LEFT, padx=3)
        _Tip(self._btn_xls, "Save all results to plaque_results.xlsx")

        self._btn_edit = _Btn(toolbar, "✏  Edit Boundaries", C["B_EDIT"],
                              command=self._edit_boundaries)
        self._btn_edit.pack(side=tk.LEFT, padx=3)
        _Tip(self._btn_edit, "Manually correct aorta / plaque masks for selected image")

        self._btn_cancel = _Btn(toolbar, "✕  Cancel", C["B_CANCEL"],
                                command=self._request_cancel)
        self._btn_cancel.pack(side=tk.LEFT, padx=3)
        self._btn_cancel.set_enabled(False)
        _Tip(self._btn_cancel, "Stop the current batch processing")

        tk.Frame(toolbar, bg=C["CARD"], width=10).pack(side=tk.LEFT)

        # Status label (right side of toolbar)
        self._status_var = tk.StringVar(value="Ready — open a folder to begin")
        self._status_lbl = tk.Label(toolbar, textvariable=self._status_var,
                                    bg=C["CARD"], fg=C["ACCENT"],
                                    font=("Helvetica", 10), anchor=tk.E)
        self._status_lbl.pack(side=tk.RIGHT, padx=10)

        # Nav hint
        tk.Label(toolbar, text="← → keys navigate", bg=C["CARD"], fg=C["FG2"],
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=6)

        # ── Progress bar ──────────────────────────────────────────────────────
        prog_row = tk.Frame(self, bg=C["BG"], pady=2)
        prog_row.pack(side=tk.TOP, fill=tk.X, padx=6)
        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(prog_row, variable=self._prog_var,
                                          maximum=100, mode="determinate",
                                          style="Horizontal.TProgressbar")
        self._prog_lbl = tk.Label(prog_row, text="", bg=C["BG"], fg=C["FG2"],
                                  font=("Helvetica", 9))
        # Hidden until processing starts
        self._prog_visible = False

        # ── Colour legend strip ───────────────────────────────────────────────
        legend = tk.Frame(self, bg=C["CARD"], pady=3, padx=8)
        legend.pack(side=tk.TOP, fill=tk.X)
        tk.Label(legend, text="Legend:", bg=C["CARD"], fg=C["FG2"],
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0,8))
        for label, hex_col in [("Aorta outline", "#00DC00"),
                                ("Plaque (dashed)", "#0020DC"),
                                ("Trunk", "#0050C8"),
                                ("Left arm", "#00C8C8"),
                                ("Right arm", "#B400C8")]:
            dot = tk.Canvas(legend, width=12, height=12, bg=C["CARD"],
                            highlightthickness=0)
            dot.create_oval(1,1,11,11, fill=hex_col, outline="")
            dot.pack(side=tk.LEFT)
            tk.Label(legend, text=label, bg=C["CARD"], fg=C["FG2"],
                     font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(2,12))

        # ── Main pane ─────────────────────────────────────────────────────────
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # File list panel
        list_frame = tk.Frame(main, bg=C["CARD"], width=170)
        main.add(list_frame, weight=0)

        tk.Label(list_frame, text="Images", bg=C["CARD"], fg=C["ACCENT"],
                 font=("Helvetica", 12, "bold")).pack(pady=(6,2))

        lb_frame = tk.Frame(list_frame, bg=C["CARD"])
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self._listbox = tk.Listbox(lb_frame, selectmode=tk.SINGLE,
                                   font=("Courier", 10),
                                   bg=C["PANEL"], fg=C["FG"],
                                   selectbackground=C["SEL"],
                                   selectforeground=C["FG"],
                                   activestyle="none",
                                   highlightthickness=0, bd=0)
        lb_sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL,
                              command=self._listbox.yview)
        self._listbox.config(yscrollcommand=lb_sb.set)
        lb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Status dots per image (processed = green dot prefix)
        tk.Label(list_frame, text="✓ = analysed", bg=C["CARD"], fg=C["FG2"],
                 font=("Helvetica", 8)).pack(pady=(0,4))

        # Image panels
        img_frame = tk.Frame(main, bg=C["BG"])
        main.add(img_frame, weight=4)

        panels = tk.Frame(img_frame, bg=C["BG"])
        panels.pack(fill=tk.BOTH, expand=True)

        def _img_col(parent, title):
            col = tk.Frame(parent, bg=C["BG"])
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(col, text=title, bg=C["BG"], fg=C["ACCENT"],
                     font=("Helvetica", 11, "bold")).pack(pady=(4,2))
            canvas = tk.Canvas(col, bg=C["PANEL"], highlightthickness=1,
                               highlightbackground="#334455")
            canvas.pack(fill=tk.BOTH, expand=True)
            return col, canvas

        orig_col,  self._orig_canvas = _img_col(panels, "Original")
        ann_col,   self._ann_canvas  = _img_col(panels, "Annotated")

        # Burden label under annotated panel
        self._burden_var = tk.StringVar(value="")
        ttk.Label(ann_col, textvariable=self._burden_var,
                  style="Burden.TLabel").pack(pady=4)

        # Welcome overlay (shown before any folder is loaded)
        self._welcome_label = tk.Label(
            self._orig_canvas,
            text="📁  Click  'Open Folder'  to get started\n\n"
                 "Supported formats: JPG · PNG · TIF\n\n"
                 "Then click  'Process All'  to analyse images",
            bg=C["PANEL"], fg=C["FG2"],
            font=("Helvetica", 13), justify=tk.CENTER,
            wraplength=320,
        )
        self._welcome_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Double-click to zoom
        self._orig_canvas.bind("<Double-Button-1>", self._zoom_orig)
        self._ann_canvas.bind("<Double-Button-1>",  self._zoom_ann)

        # ── Footer ────────────────────────────────────────────────────────────
        tk.Label(self, text="Miguelyser — Developed by P.Valiathan 2026",
                 bg=C["BG"], fg="#445566", font=("Helvetica", 8)).pack(side=tk.BOTTOM)

        # ── Results table ─────────────────────────────────────────────────────
        tbl_outer = tk.Frame(self, bg=C["BG"])
        tbl_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(2,4))

        tk.Label(tbl_outer, text="Results Table", bg=C["BG"], fg=C["ACCENT"],
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W)

        tbl_frame = tk.Frame(tbl_outer, bg=C["BG"])
        tbl_frame.pack(fill=tk.X)

        cols = ("File", "Aorta px", "Plaque px", "Total %",
                "Trunk px", "Trunk plq", "Trunk %",
                "L-Arm px", "L-Arm plq", "L-Arm %",
                "R-Arm px", "R-Arm plq", "R-Arm %")
        self._tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=5)
        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=80, anchor=tk.E)
        self._tree.column("File", width=140, anchor=tk.W)
        self._tree.tag_configure("odd",  background=C["PANEL"])
        self._tree.tag_configure("even", background=C["CARD"])

        vsb = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL,   command=self._tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._tree.pack(fill=tk.X)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, text, color=None):
        self._status_var.set(text)
        self._status_lbl.config(fg=color or C["ACCENT"])

    def _show_progress(self, pct, label=""):
        if not self._prog_visible:
            self._prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
            self._prog_lbl.pack(side=tk.LEFT)
            self._prog_visible = True
        self._prog_var.set(pct)
        self._prog_lbl.config(text=label)

    def _hide_progress(self):
        if self._prog_visible:
            self._prog_bar.pack_forget()
            self._prog_lbl.pack_forget()
            self._prog_visible = False

    def _set_busy(self, busy):
        self._busy = busy
        btns = [self._btn_folder, self._btn_all, self._btn_sel,
                self._btn_xls, self._btn_edit]
        for b in btns:
            b.set_enabled(not busy)
        self._btn_cancel.set_enabled(busy)

    def _request_cancel(self):
        self._cancel.set()
        self._set_status("Cancelling…", C["WARN"])

    # ── Folder / file management ─────────────────────────────────────────────

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing aorta images")
        if not folder:
            return
        self._folder        = folder
        self._output_folder = os.path.join(folder, "output")
        os.makedirs(self._output_folder, exist_ok=True)
        self._files   = sorted([
            f for f in glob.glob(os.path.join(folder, "*"))
            if os.path.splitext(f)[1].lower() in self.IMG_EXTS
            and "_annotated" not in os.path.basename(f)
        ])
        self._cache   = {}
        self._results = {}
        self._orig_bgr = {}
        self._listbox.delete(0, tk.END)
        for f in self._files:
            self._listbox.insert(tk.END, f"  {os.path.basename(f)}")
        n = len(self._files)
        if n == 0:
            self._set_status("No supported images found in that folder.", C["WARN"])
            messagebox.showwarning("No Images Found",
                                   f"No JPEG/PNG/TIF images were found in:\n{folder}\n\n"
                                   "Please select a folder that contains image files.")
            return
        self._tree.delete(*self._tree.get_children())
        self._sel_idx = -1
        self._welcome_label.place_forget()
        self._ann_canvas.delete("all")
        self._burden_var.set("")
        self._set_status(f"Loaded {n} image{'s' if n!=1 else ''} from: {os.path.basename(folder)}")
        # Auto-select first image
        self._listbox.selection_set(0)
        self._sel_idx = 0
        self._display_index(0)

    # ── Image display ─────────────────────────────────────────────────────────

    def _show_image_on_canvas(self, canvas, bgr):
        canvas.update_idletasks()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 500, 600
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        small  = cv2.resize(bgr, (nw, nh))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas._tk_img = tk_img
        canvas.delete("all")
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        canvas.create_image(ox, oy, anchor=tk.NW, image=tk_img)

    def _display_index(self, idx):
        if idx < 0 or idx >= len(self._files):
            return
        path = self._files[idx]
        img  = self._orig_bgr.get(path)
        if img is None:
            img = cv2.imread(path)
            if img is None:
                self._set_status(f"Cannot read: {os.path.basename(path)}", C["ERR"])
                return
            self._orig_bgr[path] = img
        self._show_image_on_canvas(self._orig_canvas, img)
        if path in self._cache:
            ann   = self._cache[path][0]
            stats = self._cache[path][7]
            self._show_image_on_canvas(self._ann_canvas, ann)
            pct   = stats["plaque_pct"]
            color = C["ERR"] if pct > 20 else C["WARN"] if pct > 10 else C["ACCENT"]
            self._burden_var.set(f"Total plaque burden: {pct:.1f}%")
        else:
            self._ann_canvas.delete("all")
            self._ann_canvas.create_text(
                self._ann_canvas.winfo_width()//2 or 250,
                self._ann_canvas.winfo_height()//2 or 300,
                text="Not yet analysed\nClick 'Process Selected'",
                fill=C["FG2"], font=("Helvetica", 12), justify=tk.CENTER)
            self._burden_var.set("")

    def _on_list_select(self, _=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        self._sel_idx = sel[0]
        self._display_index(self._sel_idx)

    def _nav(self, delta):
        if not self._files:
            return
        new_idx = (self._sel_idx + delta) % len(self._files)
        self._sel_idx = new_idx
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(new_idx)
        self._listbox.see(new_idx)
        self._display_index(new_idx)

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def _zoom_orig(self, _=None):
        if self._sel_idx < 0:
            return
        path = self._files[self._sel_idx]
        img  = self._orig_bgr.get(path)
        if img is not None:
            _ZoomWindow(self, img, f"Original — {os.path.basename(path)}")

    def _zoom_ann(self, _=None):
        if self._sel_idx < 0:
            return
        path = self._files[self._sel_idx]
        if path in self._cache:
            ann = self._cache[path][0]
            _ZoomWindow(self, ann, f"Annotated — {os.path.basename(path)}")

    # ── Processing ────────────────────────────────────────────────────────────

    def _run_process(self, paths):
        total = len(paths)
        processed = 0
        for i, path in enumerate(paths):
            if self._cancel.is_set():
                break
            fname = os.path.basename(path)
            self.after(0, lambda f=fname, i=i, t=total: (
                self._set_status(f"Processing {i+1}/{t}: {f}", C["WARN"]),
                self._show_progress(100 * i / t, f"{i+1} / {t}"),
            ))
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Cannot read image: {fname}")
                params = DEFAULT_PARAMS.copy()
                result = process_image(img, params)
                # result = (annotated, aorta_mask, plaque_mask, trunk, la, ra, bif_xy, stats)
                self._cache[path]   = result + (params,)
                self._orig_bgr[path] = img
                stats = result[7]
                self._results[path] = stats

                stem     = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(self._output_folder, stem + "_annotated.jpg")
                cv2.imwrite(out_path, result[0])
                processed += 1

                # Mark in list (safe UI update) — capture path as default arg to avoid closure bug
                idx = self._files.index(path)
                fname_display = os.path.basename(path)
                self.after(0, lambda i2=idx, fn=fname_display, s=stats, p=path: (
                    self._listbox.delete(i2),
                    self._listbox.insert(i2, f"✓ {fn}"),
                    self._listbox.itemconfig(i2, fg=C["ACCENT"]),
                    self._update_table_row(p, s),
                ))
                # Refresh display if this is the selected image
                cur_idx = self._sel_idx
                if cur_idx == self._files.index(path):
                    self.after(0, lambda i2=cur_idx: self._display_index(i2))

            except Exception as exc:
                self.after(0, lambda e=str(exc): self._set_status(f"Error: {e}", C["ERR"]))

        cancelled = self._cancel.is_set()
        self._cancel.clear()
        self.after(0, lambda: self._on_process_done(processed, total, cancelled))

    def _on_process_done(self, processed, total, cancelled):
        self._set_busy(False)
        self._hide_progress()
        if cancelled:
            self._set_status(f"Cancelled — {processed}/{total} images processed.", C["WARN"])
        else:
            self._set_status(f"Done — {processed} image{'s' if processed!=1 else ''} analysed.", C["ACCENT"])
        if 0 <= self._sel_idx < len(self._files):
            self._display_index(self._sel_idx)

    def _process_all(self):
        if not self._files:
            messagebox.showinfo("No Folder Selected", "Please open a folder first.")
            return
        self._set_busy(True)
        self._cancel.clear()
        t = threading.Thread(target=self._run_process, args=(self._files,), daemon=True)
        t.start()

    def _process_selected(self):
        if self._sel_idx < 0 or self._sel_idx >= len(self._files):
            messagebox.showinfo("Nothing Selected", "Click an image in the list first.")
            return
        self._set_busy(True)
        self._cancel.clear()
        t = threading.Thread(target=self._run_process,
                             args=([self._files[self._sel_idx]],), daemon=True)
        t.start()

    # ── Results table ─────────────────────────────────────────────────────────

    def _update_table_row(self, path, stats):
        fname = os.path.basename(path)
        for row in self._tree.get_children():
            if self._tree.item(row)["values"][0] == fname:
                self._tree.delete(row)
                break
        tag = "odd" if len(self._tree.get_children()) % 2 == 0 else "even"
        vals = (
            fname,
            stats["aorta_area_px"],   stats["plaque_area_px"],   f"{stats['plaque_pct']:.2f}",
            stats["trunk_area_px"],   stats["trunk_plaque_px"],   f"{stats['trunk_plaque_pct']:.2f}",
            stats["left_arm_area_px"],stats["left_arm_plaque_px"],f"{stats['left_arm_plaque_pct']:.2f}",
            stats["right_arm_area_px"],stats["right_arm_plaque_px"],f"{stats['right_arm_plaque_pct']:.2f}",
        )
        self._tree.insert("", tk.END, values=vals, tags=(tag,))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_excel(self):
        if not self._results:
            messagebox.showinfo("No Data Yet",
                                "Analyse some images first, then click Export Excel.")
            return
        rows = []
        for path, stats in self._results.items():
            row = {"filename": os.path.basename(path)}
            row.update(stats)
            rows.append(row)
        df = pd.DataFrame(rows, columns=[
            "filename",
            "aorta_area_px", "plaque_area_px", "plaque_pct",
            "trunk_area_px", "trunk_plaque_px", "trunk_plaque_pct",
            "left_arm_area_px", "left_arm_plaque_px", "left_arm_plaque_pct",
            "right_arm_area_px","right_arm_plaque_px","right_arm_plaque_pct",
        ])
        out = os.path.join(self._output_folder, "plaque_results.xlsx")
        try:
            df.to_excel(out, index=False, engine="openpyxl")
        except PermissionError:
            messagebox.showerror("Cannot Save",
                                 f"Could not write to:\n{out}\n\n"
                                 "The file may be open in Excel. Close it and try again.")
            return
        self._set_status("Saved: output/plaque_results.xlsx")
        messagebox.showinfo("Exported", f"Results saved to:\n{out}")

    # ── Edit Boundaries ───────────────────────────────────────────────────────

    def _edit_boundaries(self):
        if self._sel_idx < 0 or self._sel_idx >= len(self._files):
            messagebox.showinfo("Nothing Selected", "Click an image in the list first.")
            return
        path    = self._files[self._sel_idx]
        img_bgr = self._orig_bgr.get(path)
        if img_bgr is None:
            img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Cannot Read Image",
                                 f"Failed to load:\n{os.path.basename(path)}")
            return
        init_params = DEFAULT_PARAMS.copy()
        if path in self._cache and len(self._cache[path]) > 8:
            init_params = copy.deepcopy(self._cache[path][8])

        def on_apply(annotated, aorta_mask, plaque_mask, trunk, la, ra,
                     bif_xy, stats, params):
            self._cache[path]   = (annotated, aorta_mask, plaque_mask,
                                   trunk, la, ra, bif_xy, stats, params)
            self._results[path] = stats
            stem     = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(self._output_folder, stem + "_annotated.jpg")
            cv2.imwrite(out_path, annotated)
            self._update_table_row(path, stats)
            self._display_index(self._sel_idx)

        try:
            EditBoundariesWindow(self, img_bgr, init_params, on_apply)
        except Exception as exc:
            messagebox.showerror("Edit Boundaries Error",
                                 f"Could not open editor:\n{exc}")

    # ── Help dialogs ──────────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("Quick Start Guide")
        win.configure(bg=C["BG"])
        win.geometry("540x440")
        win.resizable(False, False)
        tk.Label(win, text="Quick Start Guide", bg=C["BG"], fg=C["ACCENT"],
                 font=("Helvetica", 16, "bold")).pack(pady=(20,8))
        steps = [
            ("1", "Open Folder", "Click 📁 Open Folder and select the folder\ncontaining your aorta JPEG/PNG images."),
            ("2", "Process All", "Click ⚡ Process All and wait for analysis\nto complete. A progress bar tracks progress."),
            ("3", "Review",      "Use ← → arrow keys to browse images.\nDouble-click any image panel to zoom in."),
            ("4", "Correct",     "If a result looks wrong, select the image\nand click ✏ Edit Boundaries to fix it."),
            ("5", "Export",      "Click 📊 Export Excel to save all results\nto plaque_results.xlsx in the image folder."),
        ]
        for num, title, desc in steps:
            row = tk.Frame(win, bg=C["CARD"], pady=6, padx=12)
            row.pack(fill=tk.X, padx=16, pady=4)
            tk.Label(row, text=num, bg="#1565C0", fg="white",
                     font=("Helvetica", 14, "bold"),
                     width=2, height=1).pack(side=tk.LEFT, padx=(0,10))
            info = tk.Frame(row, bg=C["CARD"])
            info.pack(side=tk.LEFT)
            tk.Label(info, text=title, bg=C["CARD"], fg=C["FG"],
                     font=("Helvetica", 11, "bold"), anchor=tk.W).pack(anchor=tk.W)
            tk.Label(info, text=desc, bg=C["CARD"], fg=C["FG2"],
                     font=("Helvetica", 9), anchor=tk.W, justify=tk.LEFT).pack(anchor=tk.W)
        tk.Button(win, text="Close", command=win.destroy,
                  bg=C["CARD2"], fg=C["FG"], relief=tk.FLAT,
                  font=("Helvetica", 11), padx=20, pady=6).pack(pady=16)

    def _show_shortcuts(self):
        win = tk.Toplevel(self)
        win.title("Keyboard Shortcuts")
        win.configure(bg=C["BG"])
        win.geometry("380x260")
        win.resizable(False, False)
        tk.Label(win, text="Keyboard Shortcuts", bg=C["BG"], fg=C["ACCENT"],
                 font=("Helvetica", 14, "bold")).pack(pady=(16,8))
        shortcuts = [("← Left arrow",  "Previous image"),
                     ("→ Right arrow", "Next image"),
                     ("Double-click",  "Zoom image panel"),
                     ("Escape",        "Close zoom / dialog")]
        for key, desc in shortcuts:
            row = tk.Frame(win, bg=C["CARD"], pady=6)
            row.pack(fill=tk.X, padx=20, pady=2)
            tk.Label(row, text=key, bg=C["CARD2"], fg=C["ACCENT"],
                     font=("Courier", 11, "bold"), padx=8, pady=4).pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=C["CARD"], fg=C["FG"],
                     font=("Helvetica", 10), padx=10).pack(side=tk.LEFT)
        tk.Button(win, text="Close", command=win.destroy,
                  bg=C["CARD2"], fg=C["FG"], relief=tk.FLAT,
                  font=("Helvetica", 11), padx=20, pady=6).pack(pady=14)

    def _show_about(self):
        icon_pil = _make_icon(64)
        win = tk.Toplevel(self)
        win.title("About")
        win.configure(bg=C["BG"])
        win.geometry("400x320")
        win.resizable(False, False)
        icon_tk = ImageTk.PhotoImage(icon_pil)
        win._icon = icon_tk
        tk.Label(win, image=icon_tk, bg=C["BG"]).pack(pady=(20,6))
        tk.Label(win, text="Atherosclerotic Plaque Analyzer",
                 bg=C["BG"], fg=C["ACCENT"], font=("Helvetica", 18, "bold")).pack()
        tk.Label(win, text="v2.0  ·  Oil Red O Staining  ·  Mouse Aorta",
                 bg=C["BG"], fg=C["FG2"], font=("Helvetica", 10)).pack()
        tk.Frame(win, bg="#334455", height=1).pack(fill=tk.X, padx=40, pady=12)
        tk.Label(win,
                 text="Automated segmentation and quantification of\n"
                      "Oil Red O-stained mouse aorta images.\n\n"
                      "Divides the Y-shaped aorta into Trunk,\n"
                      "Left Arm and Right Arm regions.",
                 bg=C["BG"], fg=C["FG2"], font=("Helvetica", 10), justify=tk.CENTER).pack()
        tk.Button(win, text="Close", command=win.destroy,
                  bg=C["CARD2"], fg=C["FG"], relief=tk.FLAT,
                  font=("Helvetica", 11), padx=20, pady=6).pack(pady=16)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AortaAnalyzer()
    app.mainloop()
